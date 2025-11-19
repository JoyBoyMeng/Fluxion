import torch
import torch.nn as nn
import torch.nn.functional as F

class FluxonRouterCos(nn.Module):
    """
    单选路由（argmax/argmin）：
      - metric='cosine'  ：按余弦相似度最大选择
      - metric='euclidean'：按欧氏距离（平方）最小选择
    其它特性：
      - 不使用投影、温度、softmax、entmax
      - 权重恒为 1（[B,1]），tau 固定 1.0 用于兼容旧接口
      - 要求 h 与 A 的最后一维一致

    Args:
        metric: 'cosine' | 'euclidean'
        eps:    数值稳定用的极小值（归一化 & 负零截断）
    """
    def __init__(self, metric: str = "cosine", eps: float = 1e-8):
        super().__init__()
        metric = metric.lower()
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric}")
        self.metric = metric
        self.eps = float(eps)

    @torch.no_grad()
    def _check_shapes(self, h: torch.Tensor, A: torch.Tensor):
        if h.dim() != 2 or A.dim() != 2:
            raise ValueError(f"h and A must be 2D, got {h.shape=} {A.shape=}")
        if h.size(-1) != A.size(-1):
            raise ValueError(
                f"Dim mismatch: h D={h.size(-1)} != A D={A.size(-1)}; "
                f"本实现要求二者最后一维一致。"
            )

    def _cosine_scores(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 归一化后做余弦，值域约 [-1, 1]
        h_n = F.normalize(h, p=2, dim=-1, eps=self.eps)   # [B, D]
        A_n = F.normalize(A, p=2, dim=-1, eps=self.eps)   # [K, D]
        return h_n @ A_n.t()                              # [B, K]

    def _euclidean_sq_dists(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 使用平方欧氏距离，避免开方（与欧氏距离同序）
        # dist2 = ||h||^2 + ||A||^2 - 2 h A^T
        h2 = (h * h).sum(dim=-1, keepdim=True)            # [B, 1]
        A2 = (A * A).sum(dim=-1, keepdim=True).t()        # [1, K]
        dist2 = h2 + A2 - 2.0 * (h @ A.t())               # [B, K]
        return dist2.clamp_min(0.0)                       # 数值稳定：去掉极小负值

    def forward(self, h: torch.Tensor, A: torch.Tensor):
        """
        h: [B, D]   memory 表征
        A: [K, D]   fluxon 状态矩阵

        Returns:
            idx:    [B, 1]  选中的 fluxon 索引
        """
        self._check_shapes(h, A)

        if self.metric == "cosine":
            scores = self._cosine_scores(h, A)            # [B, K], bigger is better
            idx = scores.argmax(dim=1, keepdim=True)      # [B, 1]
        else:  # 'euclidean'
            dist2 = self._euclidean_sq_dists(h, A)        # [B, K], smaller is better
            idx = dist2.argmin(dim=1, keepdim=True)       # [B, 1]

        return idx

