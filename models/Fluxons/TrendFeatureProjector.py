import torch
import torch.nn as nn

class TrendFeatureProjector(nn.Module):
    """
    将 item 的慢统计特征(如重复率、独立用户数、增长率、爆发度等)映射为趋势向量:
      输入: slow_feat  [B, F_slow]  (你现有是 7 维, F_slow=7)
      输出: trend_vec  [B, D_trend] (与 [fast;slow] 对齐使用)
    """
    def __init__(self, slow_feat_dim: int, trend_dim: int, hidden: int = None, device: str = 'cpu'):
        super().__init__()
        hidden = hidden or max(64, trend_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(slow_feat_dim),
            nn.Linear(slow_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, trend_dim),
        ).to(device)

    def forward(self, slow_feat: torch.Tensor) -> torch.Tensor:
        return self.net(slow_feat)
