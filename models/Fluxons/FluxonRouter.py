import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Entmax 1.5 (α=1.5) 实现：数值稳定，适合做稀疏门控 ----------
def _entmax_threshold_and_support(probs, alpha: float = 1.5, dim: int = -1, n_iter: int = 50, tol: float = 1e-6):
    assert alpha > 1.0
    x = probs
    # 防溢出平移：max 到 0
    max_val = x.max(dim=dim, keepdim=True).values
    x = x - max_val                   # max(x) = 0

    inv = 1.0 / (alpha - 1.0)

    # --- 关键：构造“包根区间” [left, right]，使 f(left) > 0, f(right) < 0 ---
    right = torch.zeros_like(max_val)           # = 0
    # 初始左端点给个负值
    left = -torch.ones_like(max_val)            # = -1

    def f(tau):
        p = torch.clamp(x - tau, min=0) ** inv
        return p.sum(dim=dim, keepdim=True) - 1.0

    # 自适应扩大左端点，直到 f(left) > 0 或者扩大到一定次数
    val_left = f(left)
    expand = 0
    while torch.any(val_left <= 0) and expand < 20:
        left = left * 2.0                       # -1, -2, -4, ...
        val_left = f(left)
        expand += 1
    # 若仍不满足，直接回退到 softmax-like 的安全值（几乎不会触发）
    # 也可把 +1 改成 +eps 防止数值问题
    if torch.any(val_left <= 0):
        tau = right
        support = torch.clamp(x - tau, min=0) ** inv
        return tau, support

    # --- 正常二分 ---
    for _ in range(n_iter):
        mid = (left + right) / 2.0
        val_mid = f(mid)
        left  = torch.where(val_mid > 0, mid, left)
        right = torch.where(val_mid <= 0, mid, right)
        if val_mid.numel() > 0 and torch.max(torch.abs(val_mid)) < tol:
            break

    tau = (left + right) / 2.0
    support = torch.clamp(x - tau, min=0) ** inv
    return tau, support


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Entmax α=1.5 的常用变体。输出在指定维度上非负且和为1，并天然稀疏。
    """
    assert logits.numel() > 0, "Empty tensor passed to entmax15!"
    # 平移不变性
    z = logits - logits.max(dim=dim, keepdim=True).values
    # 直接求阈值与支持
    _, p = _entmax_threshold_and_support(z, alpha=1.5, dim=dim)
    # 归一化（数值上应已和为1，这里做一次保险）
    p = p / (p.sum(dim=dim, keepdim=True) + 1e-12)
    return p


def topk_select(probs: torch.Tensor, k: int, dim: int = -1):
    """
    对每行取 Top-k 概率并重归一。
    输入:
      probs: [B_valid, K] 非负且和为1的分布
      k:     选取的 top-k（会自动 clip 到 <= K）
    返回:
      idx: [B_valid, k]  —— 每个样本选中的流子索引（按概率降序）
      w:   [B_valid, k]  —— 归一化后的权重
    """
    k = min(k, probs.size(dim))
    values, indices = torch.topk(probs, k, dim=dim, largest=True, sorted=True)
    weights = values / (values.sum(dim=dim, keepdim=True) + 1e-12)
    return indices, weights


class FluxonRouter(nn.Module):
    def __init__(self,
                 in_dim: int,       # 输入 [h_fast || h_slow] 的维度
                 state_dim: int,    # A 的每行维度 d
                 num_fluxons,       # K
                 mode: str = "linear",  # "linear" | "cosine"
                 k_select: int = 3,
                 tau_start: float = 2.0,
                 tau_end: float = 0.5,
                 total_steps: int = 1000,
                 device: str = "cpu"):
        super().__init__()
        self.W_Q = nn.Linear(in_dim, state_dim, bias=False).to(device)
        self.W_K = nn.Linear(state_dim, state_dim, bias=False).to(device)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)

        self.num_fluxons = num_fluxons
        self.k_select = k_select

        # tau scheduler 参数
        self.mode = mode
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.step_count = 0

    def _update_tau(self):
        """根据调度策略更新 tau"""
        self.step_count += 1
        progress = min(self.step_count / self.total_steps, 1.0)

        if self.mode == "linear":
            tau = self.tau_start - progress * (self.tau_start - self.tau_end)
        elif self.mode == "exp":
            tau = self.tau_end + (self.tau_start - self.tau_end) * (0.95 ** self.step_count)
        elif self.mode == "cosine":
            tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1 + math.cos(math.pi * progress))
        else:
            tau = self.tau_start
        return tau

    def forward(self, h_concat: torch.Tensor, A_states: torch.Tensor):
        """
        h_concat: (B_valid, in_dim) —— [h_fast || h_slow]
        A_states: (K, state_dim) —— 流子状态矩阵 A
        return:
            idx:     [B_valid, k]                   —— 每个样本选中的流子索引（按概率降序）
            wight:   [B_valid, k]                   —— 归一化后的权重
            tau:     float                    —— 当前温度
        """
        # B = h_concat.size(0)
        tau = self._update_tau()
        # queries
        q = self.W_Q(h_concat)  # [B_valid, state_dim]
        # keys: W_K A
        A_states = A_states.to(h_concat.device)
        K = self.W_K(A_states)  # [K, state_dim]
        # 打分：内积 / tau
        scores = (q @ K.t()) / max(1e-8, tau)  # [B_valid, K]
        # print(scores)
        probs = entmax15(scores, dim=-1)  # [B_valid, K]
        # probs = F.softmax(scores / 1.0, dim=-1)  # [B_valid, K]
        # print(probs.shape)
        # print(probs)
        # 筛选topK
        idx, weight = topk_select(probs, k=self.k_select, dim=-1)
        # print(idx)
        # print(weight)

        return idx, weight, tau
