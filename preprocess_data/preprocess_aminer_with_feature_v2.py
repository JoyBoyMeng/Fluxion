# downstream_build_features.py
import numpy as np
import pandas as pd

# ---- 路径设置（与你的工程一致） ----
CSV_TRAIN = '../processed_data/aminer/pp_aminer_half_year_train.csv'
CSV_VAL = '../processed_data/aminer/pp_aminer_half_year_val.csv'
CSV_TEST = '../processed_data/aminer/pp_aminer_half_year_test.csv'
CSV_ALL = '../processed_data/aminer/pp_aminer_half_year_all.csv'

FEAT_TRAIN = '../processed_data/aminer/pp_aminer_half_year_train_feature.npy'
FEAT_VAL = '../processed_data/aminer/pp_aminer_half_year_val_feature.npy'
FEAT_TEST = '../processed_data/aminer/pp_aminer_half_year_test_feature.npy'
FEAT_ALL = '../processed_data/aminer/pp_aminer_half_year_all_feature.npy'

# ---- 半年窗口设定 & 特征维度 ----
DAY = 60 * 60 * 24
front_half_year_seconds = DAY * (31 + 28 + 31 + 30 + 31 + 30)  # 1-6 月
behind_half_year_seconds = DAY * (31 + 31 + 30 + 31 + 30 + 31)  # 7-12 月
SEQ_L = 5
P_DIM = 7  # [交互数, 用户数, 重复用户数, 重复交互数, 间隔天数, 历史共同用户数, 历史共同用户交互数]


def build_features_all(data_all: pd.DataFrame) -> np.ndarray:
    """
    为 data_all 的每一行生成按行对齐的 (5,7) 特征：
      - 默认全 0
      - 若该行 label!=0，写入对应预测点（本半年度末）的最近 5×半年统计
    返回：features_all，shape=(len(data_all), 5, 7)
    """
    N = len(data_all)
    feats = np.zeros((N, SEQ_L, P_DIM), dtype=np.float32)

    # 为了对齐原始行：取每个 item 的行号索引
    ts_all = data_all['ts'].to_numpy()
    u_all = data_all['u'].to_numpy()
    i_all = data_all['i'].to_numpy()
    label_all = data_all['label'].to_numpy()

    # 全局时间边界：用 min/max（避免依赖排序）
    start_time_all = int(data_all['ts'].min())
    end_time_all = int(data_all['ts'].max())

    # 分组迭代：对每个 item 复现“第二个算法”的半年统计
    groups = data_all.groupby('i', sort=False).groups  # dict: item_id -> Int64Index(行号)
    for item_id, idxs_pd in groups.items():
        idxs = np.asarray(idxs_pd, dtype=np.int64)  # 该 item 在 data_all 中的行号
        timestamps = ts_all[idxs]
        users = u_all[idxs]
        labels = label_all[idxs]

        # —— 关键：组内按时间排序，但保留到原行号的映射 —— #
        ord_ = np.argsort(timestamps, kind='mergesort')
        t_s = timestamps[ord_]
        u_s = users[ord_]
        y_s = labels[ord_]
        row_s = idxs[ord_]  # 排序后的位置 → 原 data_all 行号

        # 滑动交替半年窗口
        half_year_count = 0
        start_time = start_time_all
        from numpy import unique, intersect1d

        # 历史唯一用户集合（用于统计“共同用户”）
        before_users = np.array([], dtype=np.int64)

        # 特征序列缓存（最近 SEQ_L 个半年度特征）
        feats_seq = []

        while start_time + front_half_year_seconds <= end_time_all:
            end_time = start_time + (front_half_year_seconds if (half_year_count % 2 == 0)
                                     else behind_half_year_seconds)
            if half_year_count == 0:
                # (start, end] 左开右闭
                start_time = start_time - 1

            # 本半年度内的索引（在排序后的序列中找）
            mask = (t_s > start_time) & (t_s <= end_time)
            win_idxs = np.where(mask)[0]

            if win_idxs.size > 0:
                # 该半年度末最后一条交互（按时间的最后一个）
                valid_pos = win_idxs[-1]

                # 统计 7 项特征
                current_popularity = int(win_idxs.size)
                current_users = u_s[win_idxs]
                unique_users, counts = unique(current_users, return_counts=True)
                current_num_users = int(unique_users.size)
                dup_mask = counts > 1
                current_num_duplicate_users = int(np.sum(dup_mask))
                current_num_duplicates = int(np.sum(counts[dup_mask] - 1))
                days_gap = float((end_time - t_s[valid_pos]) / DAY)

                if before_users.size > 0 and current_num_users > 0:
                    common_users = intersect1d(unique_users, before_users, assume_unique=False)
                    num_common_users = int(common_users.size)
                    if num_common_users > 0:
                        user_count_map = dict(zip(unique_users.tolist(), counts.tolist()))
                        total_occurrences = int(sum(user_count_map[u_] for u_ in common_users))
                    else:
                        total_occurrences = 0
                else:
                    num_common_users = 0
                    total_occurrences = 0

                # 更新历史用户集合
                before_users = np.union1d(before_users, unique_users)

                new_row = np.array([
                    current_popularity,
                    current_num_users,
                    current_num_duplicate_users,
                    current_num_duplicates,
                    days_gap,
                    num_common_users,
                    total_occurrences
                ], dtype=np.float32)
            else:
                # 空半年：七维全 0
                new_row = np.zeros((P_DIM,), dtype=np.float32)

            # 维护最近 SEQ_L 半年度特征
            feats_seq.append(new_row)
            if len(feats_seq) < SEQ_L:
                # 左侧补零
                pad = [np.zeros((P_DIM,), dtype=np.float32) for _ in range(SEQ_L - len(feats_seq))]
                feats_tail = np.stack(pad + feats_seq, axis=0)  # (5,7)
            else:
                feats_tail = np.stack(feats_seq[-SEQ_L:], axis=0)  # (5,7)

            # 若该半年度末对应的行 label!=0，则把 feats_tail 写回该行
            if win_idxs.size > 0:
                valid_pos = win_idxs[-1]
                if y_s[valid_pos] != 0:
                    row_id = row_s[valid_pos]  # 回到 data_all 的行号
                    feats[row_id] = feats_tail

            start_time = end_time
            half_year_count += 1

    return feats


def save_splits_and_check(data_all: pd.DataFrame, features_all: np.ndarray):
    """
    保存 train/val/test/all 的特征 .npy，并做检查：
      - 行数与 CSV 一致
      - label==0  ↔ 特征全 0
      - label!=0  ↔ 特征非全 0
      - valid edge num（label!=0 的计数）与“非全 0 特征”计数一致
    仅输出检查结果（无其他冗余日志）
    """
    # 读取 splits（直接读取上游已保存的 CSV）
    df_train = pd.read_csv(CSV_TRAIN)
    df_val = pd.read_csv(CSV_VAL)
    df_test = pd.read_csv(CSV_TEST)
    df_all = data_all

    # 用 CSV 的时间上界来构造掩码，确保与上游划分一致
    ts_all = data_all['ts'].to_numpy()
    t_train_max = df_train['ts'].max() if len(df_train) > 0 else -np.inf
    t_val_max = df_val['ts'].max() if len(df_val) > 0 else -np.inf

    mask_train = (ts_all <= t_train_max)
    mask_val = (ts_all > t_train_max) & (ts_all <= t_val_max)
    mask_test = (ts_all > t_val_max)

    # 保存 .npy
    np.save(FEAT_TRAIN, features_all[mask_train])
    np.save(FEAT_VAL, features_all[mask_val])
    np.save(FEAT_TEST, features_all[mask_test])
    np.save(FEAT_ALL, features_all)

    # ---- 检查函数（仅打印检查结果）----
    def check(split_name, df_split, feats_split, show_k: int = 10):
        ok = True
        # 1) 行数一致
        if len(df_split) != len(feats_split):
            print(f"[{split_name}] 行数不一致: csv={len(df_split)} vs npy={len(feats_split)}")
            ok = False

        labels = df_split['label'].to_numpy()
        flat = feats_split.reshape(feats_split.shape[0], -1) if len(feats_split) > 0 else np.zeros((0, SEQ_L * P_DIM))
        all_zero_mask = (flat == 0).all(axis=1) if len(flat) > 0 else np.array([], dtype=bool)

        # 2) label==0 ↔ 特征全 0
        nz_label = (labels != 0)
        z_label = ~nz_label
        nz_feat = ~all_zero_mask
        z_feat = all_zero_mask

        a = np.sum(z_label)
        b = np.sum(z_feat)
        c = np.sum(nz_label)
        d = np.sum(nz_feat)

        if a != b:
            print(f"[{split_name}] label==0 数={a} 与 全0特征数={b} 不一致")
            ok = False
        if c != d:
            print(f"[{split_name}] label!=0 数={c} 与 非全0特征数={d} 不一致")
            ok = False

        # 3) valid edge num 一致
        print(f"[{split_name}] OK={ok} | rows={len(df_split)} | "
              f"label==0={a} ↔ zero_feats={b} | "
              f"label!=0={c} ↔ nonzero_feats={d}")

        # ====== 逐行一致性（无 row_id，直接打印行号/行下标）======
        # 要求：label==0 ⇔ 特征全0；label!=0 ⇔ 特征非全0
        per_row_ok = (z_label == z_feat)
        bad_idx = np.where(~per_row_ok)[0]
        if bad_idx.size > 0:
            print(f"[{split_name}] 逐行不一致 {bad_idx.size} 条（显示前{min(show_k, bad_idx.size)}个下标）："
                  f"{bad_idx[:show_k].tolist()}")
            ok = False

        return ok, c, d

    ok1, c1, d1 = check("train", df_train, np.load(FEAT_TRAIN, allow_pickle=False))
    ok2, c2, d2 = check("val", df_val, np.load(FEAT_VAL, allow_pickle=False))
    ok3, c3, d3 = check("test", df_test, np.load(FEAT_TEST, allow_pickle=False))
    ok4, c4, d4 = check("all", df_all, np.load(FEAT_ALL, allow_pickle=False))

    # 额外输出各 split 的 valid edge num（应与非全 0 特征数一致）
    print(f"[valid edge num] train={c1}(labels) / {d1}(feats) | "
          f"val={c2}/{d2} | test={c3}/{d3} | all={c4}/{d4}")


def main():
    # 读取 ALL（其余三个 CSV 只用于检查/确定掩码）
    data_all = pd.read_csv(CSV_ALL)
    # 若不存在 label 列（理论上不会），补一个
    if 'label' not in data_all.columns:
        data_all['label'] = 0

    # 生成全量按行对齐特征
    features_all = build_features_all(data_all)

    # 保存拆分并做检查（仅输出检查结果）
    save_splits_and_check(data_all, features_all)


if __name__ == "__main__":
    main()
