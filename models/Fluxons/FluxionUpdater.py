import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os
class FluxonUpdater(nn.Module):
    """
    按路由结果聚合 batch 消息并更新流子中心:
      输入:
        h_fast:  [B_valid, D]
        h_slow:  [B_valid, D]
        idx:     [B_valid, k]   路由选中的流子索引
        weight:  [B_valid, k]   对应门控权重 (已归一化)
        fluxon:  需提供 get_all_fluxon() 和 set_all_fluxon(tensor) 接口；states 形状 [K, D]
      超参:
        state_dim = D     (默认与 fast/slow 维度对齐)
        ema_momentum      若 >0 则在 GRU 更新后再做一次 EMA 融合，增强稳定
    """
    def __init__(self, in_dim: int, state_dim: int, ema_momentum: float = 0.5, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        # W_m: 将 [h_fast||h_slow] 映射为与 state_dim 对齐的消息向量
        self.W_m = nn.Linear(in_dim, state_dim, bias=False).to(device)
        nn.init.xavier_uniform_(self.W_m.weight)
        # nn.init.zeros_(self.W_m.bias)
        # GRUCell: input_size=hidden_size=state_dim
        self.center_gru = nn.GRUCell(input_size=state_dim, hidden_size=state_dim).to(device)
        self.ema_m = float(ema_momentum)

    @torch.no_grad()
    def _ema_blend(self, new_s: torch.Tensor, old_s: torch.Tensor, mask: torch.Tensor):
        """
        对被更新的流子 (mask==True) 做 EMA 融合: s = (1-m)*old + m*new
        mask: [K, 1] bool/float
        """
        if self.ema_m <= 0:
            return new_s
        m = self.ema_m
        blended = old_s * (1.0 - m) + new_s * m
        # 仅在被使用的流子位置应用 EMA，其他保持 old_s
        return torch.where(mask, blended, old_s)

    def forward(self,
                h_fast: torch.Tensor,
                h_slow: torch.Tensor,
                idx: torch.Tensor,        # [B_valid, k]
                weight: torch.Tensor,     # [B_valid, k]
                A_states) -> torch.Tensor:
        """
        返回: updated_states [K, D] （已写回 fluxon）
        """
        device = h_fast.device
        B, D = h_fast.shape
        assert D == self.state_dim, f"state_dim({self.state_dim}) 应与 h_fast/h_slow 维度一致 ({D})"
        assert h_slow.shape == h_fast.shape
        assert idx.shape == weight.shape and idx.dim() == 2

        # 1) 计算每个样本的消息向量 m_i = W_m([h_fast || h_slow])  ->  [B_valid, D]
        x = torch.cat([h_fast, h_slow], dim=-1)              # [B_valid, 2D]
        m_per_sample = self.W_m(x)                           # [B_valid, D]

        # 2) 将 batch 消息按路由加权并聚合到各流子
        #    agg[k] = Σ_i w_i[k] * m_i
        K_total = A_states.shape[0]           # K
        flat_idx = idx.reshape(-1).to(device).long()         # [B_valid*k]
        flat_w = weight.reshape(-1, 1).to(device)          # [B_valid*k,1]
        # 展开 m_i 与 w_i[k] 对齐
        m_rep = m_per_sample.unsqueeze(1).expand(B, idx.size(1), D).reshape(-1, D)  # [B_valid*k, D]
        contrib = flat_w * m_rep                              # [B_valid*k, D]

        # 2a) 聚合消息
        # agg = torch.zeros(K_total, D, device=device)
        # agg.index_add_(0, flat_idx, contrib)                 # 对相同 fluxon 累加
        # # 2b) 统计每个流子的总权重（可做均值或做 mask）
        # wsum = torch.zeros(K_total, 1, device=device)
        # wsum.index_add_(0, flat_idx, flat_w)                 # [K,1]
        N = flat_idx.numel()
        M = torch.zeros(N, K_total, device=device)
        M.scatter_(1, flat_idx.view(-1, 1), 1.0)  # 构造稠密 one-hot
        agg = M.t() @ contrib  # [K, D]
        wsum = M.t() @ flat_w  # [K, 1]

        # 改成加权均值
        agg_mean = agg / (wsum + 1e-9)
        used_mask = (wsum > 0.0)                             # [K,1] bool

        # 检查点
        # stats_path = Path("./fluxon_stats/updated_mask.npy")
        # stats_path.parent.mkdir(parents=True, exist_ok=True)
        # # 本次更新到的 fluxion 掩码（[K,1] -> [K] -> numpy.bool_）
        # updated_now = used_mask.squeeze(-1).detach().cpu().numpy().astype(np.bool_)
        # # 读取历史掩码（如果不存在就用全 False）
        # if stats_path.exists():
        #     try:
        #         updated_total = np.load(stats_path, allow_pickle=False)
        #         updated_total = updated_total.astype(np.bool_)
        #     except Exception:
        #         # 文件损坏或不可读，重置
        #         updated_total = np.zeros((int(K_total),), dtype=np.bool_)
        # else:
        #     updated_total = np.zeros((int(K_total),), dtype=np.bool_)
        # # 并集：历史 | 本次
        # updated_total |= updated_now
        # # 覆盖保存（等价于“删除原先的，保存现在的总的”）
        # np.save(stats_path, updated_total)  # 如果已存在，会被覆盖
        # # 统计覆盖率
        # ever_cnt = int(updated_total.sum())
        # coverage = (ever_cnt / float(K_total)) if K_total > 0 else 0.0
        # print(f"[Fluxion] ever-updated: {ever_cnt}/{int(K_total)} = {coverage * 100:.2f}%")

        # 3) 用 GRU 更新中心: ŝ_k = GRU(m_k, s_k)
        old_states = A_states.to(device)      # [K, D]
        # 对所有流子批量计算一次 GRU，然后用 mask 决定是否采用
        new_all = self.center_gru(agg_mean, old_states)           # [K, D]
        updated = torch.where(used_mask, new_all, old_states)  # 未被选中的保持不变

        # 4) 可选：EMA 融合（仅对 used 的位置）
        updated = self._ema_blend(updated, old_states, used_mask)

        return updated
