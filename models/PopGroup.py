import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict

from utils.utils import NeighborSampler
from models.modules import TimeEncoder
from models.Fluxons.Fluxon import Fluxon
from models.Fluxons.FluxonRouter import FluxonRouter
from models.Fluxons.FluxonRouterCos import FluxonRouterCos
from models.Fluxons.FluxionUpdater import FluxonUpdater
from models.Fluxons.FluxionUpdaterCos import FluxonUpdaterCos
from models.modules import TimeEncoder
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from models.Fluxons.TrendFeatureProjector import TrendFeatureProjector


class PopGroup(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 time_feat_dim: int, model_name: str = '', num_layers: int = 2,
                 num_heads: int = 2, dropout: float = 0.1,
                 device: str = 'cpu', item_nums: int = 0, user_nums: int = 0,
                 mode: str = "linear", k_select: int = 3, tau_start: float = 2.0,
                 tau_end: float = 0.5, total_steps: int = 1000, router_type: str = 'cosine',
                 distance_type: str = 'cosine', fluxion_init_type: str = 'zero', fluxion_ema: float = 0.3,
                 fluxion_size: int = 16):
        """
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param model_name: str, name of memory-based models, could be TGN, DyRep or JODIE
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(PopGroup, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        self.model_name = model_name
        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]
        self.num_items = item_nums
        self.memory_dim = self.node_feat_dim
        # since models use the identity function for message encoding, message dimension is 2 * memory_dim + time_feat_dim + edge_feat_dim
        self.message_dim = self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # message module (models use the identity function for message encoding, hence, we only create MessageAggregator)
        self.message_aggregator = MessageAggregator()

        # memory modules
        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)
        self.memory_updater = GRUMemoryUpdater(memory_bank=self.memory_bank, message_dim=self.message_dim,
                                               memory_dim=self.memory_dim)

        # slow memory modules
        self.slow_memory_bank = nn.Parameter(torch.zeros((self.num_items, self.memory_dim), device=self.device), requires_grad=False)
        self.slow_memory_updater = SlowMemoryUpdater(memory_bank=self.slow_memory_bank,
                                                     memory_dim=self.memory_dim,
                                                     mlp_hidden=2*self.memory_dim,
                                                     device=self.device)
        self.num_users = user_nums + 1
        assert self.num_nodes == self.num_users + self.num_items, '数据处理错误，数据集有问题'

        # Fluxon modules
        self.mode = mode
        self.k_select = k_select
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.router_type = router_type
        self.fluxion_init_type = fluxion_init_type
        self.num_fluxions = self.num_items//fluxion_size
        print(f'fluxion数量为:{self.num_fluxions}')
        self.fluxon_bank = Fluxon(num_fluxons=self.num_fluxions,
                                  state_dim=self.memory_dim*2,
                                  init_type=self.fluxion_init_type,
                                  device=self.device)
        if router_type == 'cosine':
            self.router = FluxonRouterCos(metric=distance_type, eps=1e-8)
        else:
            self.router = FluxonRouter(in_dim=self.memory_dim * 2,
                                       state_dim=self.memory_dim * 2,
                                       num_fluxons=self.num_fluxions,
                                       mode=self.mode,
                                       k_select=self.k_select,
                                       tau_start=self.tau_start,
                                       tau_end=self.tau_end,
                                       total_steps=self.total_steps,
                                       device=self.device)
        if router_type == 'cosine':
            self.fluxon_updater = FluxonUpdaterCos(in_dim=self.memory_dim * 2,
                                                   state_dim=self.memory_dim * 2,
                                                   ema_momentum=fluxion_ema,
                                                   device=self.device)
        else:
            self.fluxon_updater = FluxonUpdater(in_dim=self.memory_dim * 2,
                                                state_dim=self.memory_dim,
                                                ema_momentum=fluxion_ema,
                                                device=self.device)
        self.reduce_fluxion_dim = FluxionLinearReduce(in_dim=self.memory_dim * 2, out_dim=self.memory_dim, bias=False)


        # Fluxion aggregation modules
        self.history_len = 5
        # 空位用 self.num_fluxions 填充，对应一个全0表征
        self.pad_flux_index = self.num_fluxions
        # 对 item 节点建历史（大小 = num_items × N），N中的值范围为[0,self.num_fluxions]
        self.register_buffer(
            "item_flux_hist",
            torch.full((self.num_items, self.history_len), self.pad_flux_index, dtype=torch.long)
        )
        # 绝对时间历史：初始 0，绝对时间要多存一个，相对时间才正确。
        self.register_buffer(
            "item_time_hist",
            torch.zeros(self.num_items, self.history_len+1, dtype=torch.float32)
        )
        self.time_encoder_fluxion = TimeEncoder(time_dim=self.memory_dim * 2)
        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.memory_dim * 2, num_heads=self.num_heads,
                               dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
        self.output_layer = nn.Linear(in_features=self.memory_dim * 2,
                                      out_features=self.memory_dim, bias=True)

    def compute_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                             node_interact_times: np.ndarray, edge_ids: np.ndarray,
                                             dst_slow_feature: np.ndarray, valid_index: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param valid_index: with label
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :param dst_slow_feature: ndarray, shape (batch_size, 5, 7)
        :return:
        """
        # Tensor, shape (2 * batch_size, )
        node_ids = np.concatenate([src_node_ids, dst_node_ids])

        # compute new raw messages for source and destination nodes
        unique_src_node_ids, new_src_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=src_node_ids,
                                                                                            dst_node_ids=dst_node_ids,
                                                                                            node_interact_times=node_interact_times,
                                                                                            edge_ids=edge_ids)
        unique_dst_node_ids, new_dst_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=dst_node_ids,
                                                                                            dst_node_ids=src_node_ids,
                                                                                            node_interact_times=node_interact_times,
                                                                                            edge_ids=edge_ids)

        # store new raw messages for source and destination nodes
        self.memory_bank.store_node_raw_messages(node_ids=unique_src_node_ids, new_node_raw_messages=new_src_node_raw_messages)
        self.memory_bank.store_node_raw_messages(node_ids=unique_dst_node_ids, new_node_raw_messages=new_dst_node_raw_messages)

        assert edge_ids is not None
        # if the edges are positive, update the memories for source and destination nodes (since now we have new messages for them)
        self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)
        # clear raw messages for source and destination nodes since we have already updated the memory using them
        self.memory_bank.clear_node_raw_messages(node_ids=node_ids)
        fast_memory = self.memory_bank.get_memories(dst_node_ids)

        if valid_index[0].size == 0:
            dst_node_embeddings = None
            return dst_node_embeddings

        # 以下是slow memory过程
        slow_memory_old = self.slow_memory_bank[torch.from_numpy(dst_node_ids-self.num_users)]
        dst_slow_feature = torch.from_numpy(dst_slow_feature).float().to(self.device)
        dst_slow_feature = dst_slow_feature[:, -1, :]   # shape (batch_size, 7)
        # 更新
        slow_memory = self.slow_memory_updater(slow_memory_old, dst_slow_feature, torch.from_numpy(dst_node_ids))
        # 写回
        valid_slow_memory = slow_memory[valid_index]
        valid_dst_node_ids = dst_node_ids[valid_index] - self.num_users
        self.slow_memory_bank[valid_dst_node_ids] = valid_slow_memory
        # 再次检查double check valid_index is right
        ok = dst_slow_feature[valid_index].abs().sum(dim=-1) != 0
        if not ok.all():
            print(dst_slow_feature[valid_index])
            exit()

        # 以下是fluxion过程
        # 准备trend
        dst_trend_embeddings = torch.concatenate((fast_memory, slow_memory), dim=-1)
        valid_dst_trend_embeddings = dst_trend_embeddings[valid_index]
        valid_fast_memory = fast_memory[valid_index]
        valid_slow_memory = slow_memory[valid_index]
        assert torch.equal(valid_dst_trend_embeddings, torch.concatenate((valid_fast_memory, valid_slow_memory), dim=-1)), 'valid process error'
        # 路由
        A_states = self.fluxon_bank.get_all_fluxon().detach().clone()
        if self.router_type == 'cosine':
            idx = self.router(valid_dst_trend_embeddings, A_states)     # [B_valid, 1]
        else:
            idx, weight, tau = self.router(valid_dst_trend_embeddings, A_states)
        # 更新，产生梯度
        if self.router_type == 'cosine':
            updated_fluxon_bank = self.fluxon_updater(valid_fast_memory, valid_slow_memory, idx, A_states)
        else:
            updated_fluxon_bank = self.fluxon_updater(valid_fast_memory, valid_slow_memory, idx, weight, A_states)
        # 写回，copy
        with torch.no_grad():
            self.fluxon_bank.set_all_fluxon(None, updated_fluxon_bank)
        # 取出
        if self.router_type == 'cosine':
            sel = idx.squeeze(-1)   # [B_valid]
            fluxion_memory = updated_fluxon_bank[sel]   # [B_valid, D]  取出对应的 fluxion 表征
            fluxion_embedding = self.reduce_fluxion_dim(fluxion_memory)
        else:
            # 按索引拿到每个样本选中的 k 个中心
            # idx: [B_valid, k] (LongTensor), weight: [B_valid, k]
            picked = updated_fluxon_bank[idx]  # [B_valid, k, D] 高级索引
            # 加权求和得到 z
            fluxion_embedding = (weight.unsqueeze(-1) * picked).sum(dim=1)  # [B_valid, D]

        # 以下是fluxion轨迹聚合过程
        item_indices = dst_node_ids[valid_index] - self.num_users
        item_indices = torch.from_numpy(item_indices)   # [B_valid]
        flux_indices = idx.squeeze(-1)  # [B_valid]
        K, D = updated_fluxon_bank.shape
        pad_row = torch.zeros(1, D, device=self.device) # [1, D]
        updated_fluxon_bank = torch.cat([updated_fluxon_bank, pad_row], dim=0)  # [K+1, D]
        node_interact_times_all = torch.from_numpy(node_interact_times).float().to(self.device)  # [B]
        event_times = node_interact_times_all[valid_index] # [B_valid]
        # 调用轨迹聚合函数，得到轨迹表征
        traj_emb = self.aggregate_fluxion_trajectory(
            item_indices=item_indices,  # [B_valid]
            flux_indices=flux_indices,  # [B_valid]
            updated_fluxon_bank=updated_fluxon_bank,  # [K+1, D]，最后一行是 dummy=0
            event_times=event_times  # [B_valid]
        )

        dst_node_embeddings = torch.concatenate(
            (valid_fast_memory, valid_slow_memory, fluxion_embedding, traj_emb), dim=-1
        )

        # dst_node_embeddings = torch.concatenate((valid_fast_memory, valid_slow_memory, fluxion_embedding), dim=-1)
        return dst_node_embeddings

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        update memories for nodes in node_ids
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(node_ids=node_ids,
                                                                                                                   node_raw_messages=node_raw_messages)

        # update the memories with the aggregated messages
        self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
                                            unique_node_timestamps=unique_node_timestamps)

    def compute_new_node_raw_messages(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                      node_interact_times: np.ndarray, edge_ids: np.ndarray):
        """
        compute new raw messages for nodes in src_node_ids
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param dst_node_embeddings: Tensor, shape (batch_size, node_feat_dim)
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        src_node_memories = self.memory_bank.get_memories(node_ids=src_node_ids)
        dst_node_memories = self.memory_bank.get_memories(node_ids=dst_node_ids)

        # Tensor, shape (batch_size, )
        src_node_delta_times = torch.from_numpy(node_interact_times).float().to(self.device) - \
                               self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        # Tensor, shape (batch_size, time_feat_dim)
        src_node_delta_time_features = self.time_encoder(src_node_delta_times.unsqueeze(dim=1)).reshape(len(src_node_ids), -1)

        # Tensor, shape (batch_size, edge_feat_dim)
        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]

        # Tensor, shape (batch_size, message_dim = memory_dim + memory_dim + time_feat_dim + edge_feat_dim)
        new_src_node_raw_messages = torch.cat([src_node_memories, dst_node_memories, src_node_delta_time_features, edge_features], dim=1)

        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        new_node_raw_messages = defaultdict(list)
        # ndarray, shape (num_unique_node_ids, )
        unique_node_ids = np.unique(src_node_ids)

        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append((new_src_node_raw_messages[i], node_interact_times[i]))

        return unique_node_ids, new_node_raw_messages


    def aggregate_fluxion_trajectory(
            self,
            item_indices: torch.Tensor,  # [B_valid]  item 行号 (0..num_items-1)
            flux_indices: torch.Tensor,  # [B_valid]  本次路由命中的 fluxion index
            updated_fluxon_bank: torch.Tensor,  # [K+1, D]     最新的 fluxion memory
            event_times: torch.Tensor  # [B_valid]  本次事件的绝对时间（float）
    ) -> torch.Tensor:
        """
           根据每个 item 的历史 fluxion 轨迹 + 相对时间，做 Transformer 聚合
           历史结构:
              - item_flux_hist: [num_items, N]
                最近 N 次事件的 flux index，0 是最新，N-1 是最旧，pad_flux_index 表示无效
              - item_time_hist: [num_items, N+1]
                最近 N+1 次事件的绝对时间，0 是最新，对应 item_flux_hist[:,0]，
                1 对应 flux_hist[:,1]，…，N 对应“比最旧 flux 更早的一次事件”，用于算最后一个间隔
            相对时间（间隔）定义:
              Δt_i = t_i - t_{i+1}, i = 0..N-1
              每个 Δt_i 对应 flux_hist[:, i]（这次事件与前一次之间的间隔）
           返回:
               traj_emb: [B_valid, D]  轨迹聚合表征
        """
        K = int(self.num_fluxions)
        N = int(self.history_len)
        item_indices = item_indices.to(device=self.device, dtype=torch.long)  # [B_valid]
        flux_indices = flux_indices.to(device=self.device, dtype=torch.long)  # [B_valid]
        event_times = event_times.to(device=self.device, dtype=torch.float32)  # [B_valid]
        B_valid = item_indices.size(0)
        assert flux_indices.shape == (B_valid,), f"flux_indices 应为 [B_valid], got {flux_indices.shape}"
        assert event_times.shape == (B_valid,), f"event_times 应为 [B_valid], got {event_times.shape}"
        # 1. 从历史 buffer 取出该 batch 对应的轨迹
        # 每行是某个 item 最近 N 次事件的 flux 索引（含 dummy 索引）
        hist_flux = self.item_flux_hist[item_indices]  # [B_valid, N]
        # 每行是某个 item 最近 N+1 次绝对时间
        hist_time = self.item_time_hist[item_indices]  # [B_valid, N+1]
        # 2. 右移一位，把历史往后挪，新事件插到最前面
        # 原来的第 i 位（0..N-1）挪到 i+1 位，原来最后一位挪到 0
        hist_flux = torch.roll(hist_flux, shifts=1, dims=1)
        hist_time = torch.roll(hist_time, shifts=1, dims=1)
        # 把本次路由得到的 flux index 写到最前面位置
        hist_flux[:, 0] = flux_indices
        hist_time[:, 0] = event_times
        # 3. 把更新后的历史写回 buffer，供下次使用
        self.item_flux_hist[item_indices] = hist_flux
        self.item_time_hist[item_indices] = hist_time
        # 4. 根据历史 flux 索引取出对应的 fluxion memory
        # 对每个 item、每个历史位置拿到对应 fluxion 的 embedding
        hist_flux_mem = updated_fluxon_bank[hist_flux]  # [B_valid, N, D]
        # 5. 用 N+1 个绝对时间计算 N 个时间间隔
        assert torch.all(hist_time[:, :-1] >= hist_time[:, 1:]), "hist_time 不是从大到小排序的"
        # t_cur = hist_time[:, :N]
        # t_prev = hist_time[:, 1:]
        # # 相邻事件的时间差；负值裁剪成 0
        # dt = (t_cur - t_prev).clamp_min(0.0)  # [B_valid, N]
        ref_t = hist_time[:, 0:1]
        t_targets = hist_time[:, :N]
        dt = (ref_t - t_targets).clamp_min(0.0)  # [B_valid, N]
        # 时间编码
        time_emb = self.time_encoder_fluxion(dt)  # [B_valid, N, time_dim=D]
        seq = hist_flux_mem + time_emb  # [B_valid, N, D]
        for transformer in self.transformers:
            seq = transformer(seq)  # [B_valid, N, D]
        traj_emb = seq[:, 0, :]  # [B_valid, D]
        traj_emb = self.output_layer(traj_emb)  # [B_valid, D/2]
        return traj_emb

    def reset_item_history(self):
        """
        每个 epoch 开头调用，清空 item 的 fluxion 和时间历史
        """
        # flux 轨迹用 pad_flux_index 填充
        self.item_flux_hist.fill_(self.pad_flux_index)
        # 时间轨迹直接置 0
        self.item_time_hist.zero_()

# Message-related Modules
class MessageAggregator(nn.Module):

    def __init__(self):
        """
        Message aggregator. Given a batch of node ids and corresponding messages, aggregate messages with the same node id.
        """
        super(MessageAggregator, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        given a list of node ids, and a list of messages of the same length,
        aggregate different messages with the same node id (only keep the last message for each node)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids = np.unique(node_ids)
        unique_node_messages, unique_node_timestamps, to_update_node_ids = [], [], []

        for node_id in unique_node_ids:
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])

        # ndarray, shape (num_unique_node_ids, ), array of unique node ids
        to_update_node_ids = np.array(to_update_node_ids)
        # Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if len(unique_node_messages) > 0 else torch.Tensor([])
        # ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        unique_node_timestamps = np.array(unique_node_timestamps)

        return to_update_node_ids, unique_node_messages, unique_node_timestamps


# Memory-related Modules
class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        # Parameter, last updated time of nodes, shape (num_nodes, )
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)
        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        self.node_raw_messages = defaultdict(list)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids)]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories

    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

        return self.node_memories.data.clone(), self.node_last_updated_times.data.clone(), cloned_node_raw_messages

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        self.node_memories.data, self.node_last_updated_times.data = backup_memory_bank[0].clone(), backup_memory_bank[1].clone()

        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[2].items():
            self.node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()

        # Detach all stored messages
        for node_id, node_raw_messages in self.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append((node_raw_message[0].detach(), node_raw_message[1]))

            self.node_raw_messages[node_id] = new_node_raw_messages

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        """
        store raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param new_node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        """
        clear raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        """
        get last updated times for nodes in unique_node_ids
        :param unique_node_ids: ndarray, (num_unique_node_ids, )
        :return:
        """
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]

    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])


class MemoryUpdater(nn.Module):

    def __init__(self, memory_bank: MemoryBank):
        """
        Memory updater.
        :param memory_bank: MemoryBank
        """
        super(MemoryUpdater, self).__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                        unique_node_timestamps: np.ndarray):
        """
        update memories for nodes in unique_node_ids
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, return without updating operations
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_unique_node_ids, memory_dim)
        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        # Tensor, shape (num_unique_node_ids, memory_dim)
        updated_node_memories = self.memory_updater(unique_node_messages, node_memories)
        # update memories for nodes in unique_node_ids
        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)

        # update last updated times for nodes in unique_node_ids
        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

    def get_updated_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                             unique_node_timestamps: np.ndarray):
        """
        get updated memories based on unique_node_ids, unique_node_messages and unique_node_timestamps
        (just for computation), but not update the memories
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, directly return node_memories and node_last_updated_times without updating
        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.data.clone(), self.memory_bank.node_last_updated_times.data.clone()

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_nodes, memory_dim)
        updated_node_memories = self.memory_bank.node_memories.data.clone()
        updated_node_memories[torch.from_numpy(unique_node_ids)] = self.memory_updater(unique_node_messages,
                                                                                       updated_node_memories[torch.from_numpy(unique_node_ids)])

        # Tensor, shape (num_nodes, )
        updated_node_last_updated_times = self.memory_bank.node_last_updated_times.data.clone()
        updated_node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

        return updated_node_memories, updated_node_last_updated_times


class GRUMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(GRUMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)

class RNNMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        RNN-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(RNNMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)

class SlowMemoryUpdater(nn.Module):
    """
    用 dst_slow_feature 更新 slow_memory_old，并写回全局 slow_memory_bank
    - 输入:
        slow_memory_old: (B, D)
        dst_slow_feature: (B, 7)
        dst_node_ids: (B,)  LongTensor
    - 过程:
        1) dst_slow_feature -> MLP -> (B, D)
        2) GRUCell(input=(B, D), hidden=slow_memory_old) -> updated (B, D)
        3) 将 updated 写回 slow_memory_bank[dst_node_ids]
           * 若 dst_node_ids 有重复，取最后出现的
    - 输出:
        updated: (B, D)  # 与 slow_memory_old 对齐的逐样本更新后内存
    """
    def __init__(self, memory_bank: nn.Parameter, memory_dim: int, mlp_hidden: int = 128, device: str = 'cpu'):
        super().__init__()
        assert isinstance(memory_bank, nn.Parameter), "memory_bank 必须是 nn.Parameter"
        self.memory_bank = memory_bank          # 形状: (N_items, D)，requires_grad=False
        self.memory_dim = memory_dim
        self.device = device
        # 7 -> D 的投影（可改为更深/不同激活）
        self.mlp = nn.Sequential(
            nn.Linear(7, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, memory_dim),
        ).to(self.device)
        # GRUCell: 输入维度=memory_dim，隐状态维度=memory_dim
        self.gru = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim).to(self.device)

    def forward(self, slow_memory_old: torch.Tensor,
                dst_slow_feature: torch.Tensor,
                dst_node_ids: torch.Tensor) -> torch.Tensor:
        """
        slow_memory_old: (B, D)
        dst_slow_feature: (B, 7)
        dst_node_ids: (B,)  LongTensor（若是 numpy 请先外部转换）
        """
        assert slow_memory_old.dim() == 2 and slow_memory_old.size(1) == self.memory_dim
        assert dst_slow_feature.shape[0] == slow_memory_old.shape[0] and dst_slow_feature.shape[1] == 7
        assert dst_node_ids.shape[0] == slow_memory_old.shape[0]
        # 1) 特征投影到 memory 维度
        inp = self.mlp(dst_slow_feature)  # (B, D)
        # 2) GRUCell 更新
        updated = self.gru(inp, slow_memory_old)  # (B, D)
        return updated

class FluxionLinearReduce(nn.Module):
    """一层线性：in_dim -> out_dim"""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)
        # 可选初始化（Xavier 更稳）
        # nn.init.xavier_uniform_(self.proj.weight)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs
