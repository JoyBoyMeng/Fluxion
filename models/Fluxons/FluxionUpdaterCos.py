import torch
import torch.nn as nn
import torch.nn.functional as F


class FluxonUpdaterCos(nn.Module):
    """
    适配“单选路由”（idx:[B,1]，无 weight）的流子更新器

    输入:
      h_fast:  [B, D]
      h_slow:  [B, D]
      idx:     [B, 1]     每个样本选中的 fluxion 索引（单选）
      A_states:[K, 2D]     流子中心矩阵（Tensor）

    超参:
      in_dim      = 2D（如果是 [h_fast||h_slow]，则应为 2*D）
      state_dim   = 2D （与 A_states 的最后一维一致）
      ema_momentum ∈ [0,1]；>0 时在 GRU 后再做一次 EMA 融合增强稳定
    """
    def __init__(self, in_dim: int, state_dim: int, ema_momentum: float = 0.5, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.state_dim = int(state_dim)

        # GRUCell: input_size=hidden_size=state_dim
        self.center_gru = nn.GRUCell(input_size=state_dim, hidden_size=state_dim).to(device)

        self.ema_m = float(ema_momentum)

    def _ema_blend_sel(self, new_sel: torch.Tensor, old_sel: torch.Tensor):
        """仅对被选中的条目做 EMA 融合；未选中的不在此函数中处理。"""
        if self.ema_m <= 0:
            return new_sel
        m = self.ema_m
        return old_sel * (1.0 - m) + new_sel * m

    def forward(self,
                h_fast: torch.Tensor,
                h_slow: torch.Tensor,
                idx: torch.Tensor,          # [B, 1]
                A_states: torch.Tensor      # [K, D]
                ) -> torch.Tensor:
        """
        返回: updated_states [K, 2D] （仅把本次路由命中的 fluxion 更新）
        """
        device = h_fast.device
        B, D = h_fast.shape
        assert 2*D == self.state_dim, f"state_dim({self.state_dim}) 应与 h_fast+h_slow 维度一致 ({2*D})"
        assert h_slow.shape == h_fast.shape, "h_fast 与 h_slow 形状必须一致"
        assert idx.dim() == 2 and idx.size(0) == B and idx.size(1) == 1, f"idx 需为 [B,1]，实际 {idx.shape}"
        assert A_states.dim() == 2 and A_states.size(1) == 2*D, f"A_states 需为 [K,{2*D}]，实际 {A_states.shape}"

        # 1) 每个样本的消息向量 m_i = W_m([h_fast || h_slow])  ->  [B, 2D]
        message = torch.cat([h_fast, h_slow], dim=-1)                  # [B, 2D]
        # m_per_sample = self.W_m(x)                               # [B, 2D]

        # 2) 将样本按其选中的 fluxion 分组并做均值
        #    flat_idx: [B], 取唯一索引并聚合
        flat_idx = idx.view(-1).to(device).long()                # [B]
        uniq, inv = torch.unique(flat_idx, return_inverse=True)  # uniq:[U], inv:[B] 指示每个样本属于 uniq 中哪个组

        # 聚合 Σ m_i 与计数
        # U = uniq.size(0)
        # agg = torch.zeros(U, 2*D, device=device, dtype=message.dtype)   # [U, 2D]
        # cnt = torch.zeros(U, 1, device=device, dtype=message.dtype)   # [U, 1]
        # ones = torch.ones(B, 1, device=device, dtype=message.dtype)   # [B, 1]

        # agg.index_add_(0, inv, message)    # 对每个唯一 fluxion 的桶累加消息
        # cnt.index_add_(0, inv, ones)            # 对应计数
        # m_mean = agg / (cnt + 1e-9)             # [U, 2D] 分组均值
        # 1) 排序后相同 inv 连续
        sorted_inv, perm = torch.sort(inv, stable=True)  # [B]
        msg_sorted = message[perm]  # [B, 2D]
        # 2) 计算前缀和
        csum = torch.cumsum(msg_sorted, dim=0)  # [B, 2D]
        # 3) 找每一段的起止位置
        change = torch.ones(B, dtype=torch.bool, device=device)
        change[1:] = sorted_inv[1:] != sorted_inv[:-1]
        starts = torch.nonzero(change, as_tuple=False).squeeze(1)  # [U]
        ends = torch.cat([starts[1:], torch.tensor([B], device=device)])  # [U]
        # 4) 分段和（end 累加 - start-1 累加）
        end_csum = csum[ends - 1]  # [U, 2D]
        start_csum = torch.zeros_like(end_csum)
        valid = starts > 0
        start_csum[valid] = csum[starts[valid] - 1]
        agg = end_csum - start_csum  # [U, 2D]
        # 5) 分段计数
        cnt = (ends - starts).unsqueeze(1).to(message.dtype)  # [U, 1]
        m_mean = agg / (cnt + 1e-9)  # [U, 2D] 分组均值

        # 3) 只对被命中的 fluxion 做 GRU 更新
        old_states = A_states.to(device)        # [K, 2D]
        updated = old_states.clone()

        old_sel = old_states[uniq]              # [U, 2D]
        new_sel = self.center_gru(m_mean, old_sel)  # [U, 2D]

        # 4) 可选 EMA 融合（仅对选中的行）
        new_sel = self._ema_blend_sel(new_sel, old_sel)

        # 5) 写回被命中的条目，切片是可以传导梯度的
        updated[uniq] = new_sel
        # updated = torch.index_copy(old_states, 0, uniq, new_sel)

        return updated
