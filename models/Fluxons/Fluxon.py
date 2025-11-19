import torch
import torch.nn as nn
import numpy as np
import math
from collections import deque

class Fluxon(nn.Module):
    def __init__(self, num_fluxons: int, state_dim: int,
                 half_life_short: float = 21.0, half_life_long: float = 90.0,
                 eps: float = 1e-6, init_type: str = 'zero', device="cpu"):
        """
        A Fluxon class to store group-level embeddings and popularity statistics.
        :param num_fluxons: 流子数量 (K)
        :param state_dim: 每个流子的 embedding 维度
        :param half_life_short: 短期 EMA 半衰期（天为单位，可换算为秒）
        :param half_life_long: 长期 EMA 半衰期
        """
        super().__init__()
        self.num_fluxons = num_fluxons
        self.state_dim = state_dim
        self.device = device
        self.init_type = init_type

        # 流子状态向量 (K, d)
        if self.init_type == 'zero':
            self.states = nn.Parameter(torch.zeros((num_fluxons, state_dim), device=device),
                                       requires_grad=False)
        elif self.init_type == 'ball':
            self.states = nn.Parameter(
                torch.empty((num_fluxons, state_dim), device=device),
                requires_grad=False
            )
        else:
            print('init type missing, exit')
            exit()
        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        if self.init_type == 'zero':
            self.states.data.zero_()
        elif self.init_type == 'ball':
            with torch.no_grad():
                g = torch.Generator(device=self.device)  # 局部 RNG
                g.manual_seed(42)  # 固定种子
                # 单位球面归一化
                # 高斯采样：为每个流子生成一个随机方向（打破对称）
                w = torch.randn(self.num_fluxons, self.state_dim, device=self.device)
                # 行归一化到单位球面：只保留“方向”信息，模长=1
                # 这样 K= W_K A 的尺度由 W_K 决定，更稳；A 提供方向多样性
                w = w / (w.norm(dim=1, keepdim=True) + 1e-12)
                # 选择范数尺度：由于后面还有 W_K（建议 Xavier 初始化），此处保持 1.0 最稳，
                # 避免早期分数过大/过小导致 entmax 饱和或全均匀
                target_norm = 1.0
                w = w * target_norm
                # 复制到参数：作为 A 的初始状态（K 个“趋势中心”）
                self.states.copy_(w)
        else:
            print('init type missing, exit')
            exit()
    def get_all_fluxon(self):
        """
        get all fluxons
        :return: (K, state_dim)
        """
        return self.states

    def set_all_fluxon(self, idx, updated):
        dev = self.states.device
        D = self.states.size(1)
        # 全量覆盖
        if idx is None:
            assert updated.shape == self.states.shape, \
                f"updated 形状应为 {tuple(self.states.shape)}，但得到 {tuple(updated.shape)}"
            self.states.copy_(updated.to(dev))
            return
        # 规范化 idx：支持 list / numpy / torch
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx)
        idx = idx.to(dev)
        # 整型索引
        idx = idx.long().view(-1)
        assert updated.size(0) == idx.numel() and updated.size(1) == D, \
            f"updated 形状应为 [{idx.numel()}, {D}]，但得到 {tuple(updated.shape)}"
        # 边界检查（可选）
        if torch.any(idx < 0) or torch.any(idx >= self.states.size(0)):
            raise IndexError("idx 中存在越界的流子索引")
        # 用 index_copy_ 支持重复索引，最后一次写入生效
        self.states.index_copy_(0, idx, updated.to(dev))

    def detach_memory_bank(self):
        self.states.detach()

    def backup_memory_bank(self):
        return self.states.data.clone()

    def reload_memory_bank(self, backup_memory_bank):
        self.states.data = backup_memory_bank.clone()