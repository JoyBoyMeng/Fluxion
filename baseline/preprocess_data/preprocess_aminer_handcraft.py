import pandas as pd
import numpy as np

data_train = pd.read_csv('../../processed_data/aminer/pp_aminer_half_year_train.csv')
data_val = pd.read_csv('../../processed_data/aminer/pp_aminer_half_year_val.csv')
data_test = pd.read_csv('../../processed_data/aminer/pp_aminer_half_year_test.csv')
data_all = pd.concat([data_train, data_val, data_test])
print(f'train data edge num is {len(data_train)}')
print(f'val data edge num is {len(data_val)}')
print(f'test data edge num is {len(data_test)}')
print(f'all data edge num is {len(data_all)}')

start_time_all = data_all['ts'].values[0]
end_time_all = data_all['ts'].values[-1]

front_half_year_seconds = 60 * 60 * 24 * (31+28+31+30+31+30)
behind_half_year_seconds = 60 * 60 * 24 * (31+31+30+31+30+31)
squence_l = 5

features_list = []
labels_list = []
time_list = []
idx_list = []
# 分别对每个 item 进行处理
for item_id, group in data_all.groupby('i'):
    # group = group.reset_index()  # 保留原始索引
    timestamps = group['ts'].values  # 当前 item 的所有交互时间（按顺序）
    users = group['u'].values   # # 当前 item 的所有交互user（按顺序、有重复）
    labels = group['label'].values   # # 当前 item 的所有交互label（按顺序、有0）
    idxs = group['idx'].values
    print(f'Processing the {item_id}-th item, with a total of {len(group)} interactions.')
    start_time = start_time_all
    half_year_count = 0
    before_users = np.array([], dtype=int)
    features = np.zeros((squence_l, 7))
    while start_time + front_half_year_seconds <= end_time_all:
        # 判定是上半年还是下半年
        if half_year_count % 2 == 0:
            end_time = start_time + front_half_year_seconds
        else:
            end_time = start_time + behind_half_year_seconds
        # 最后半年没数据
        if half_year_count == 15:
            start_time = end_time
            continue
        # 第一个半年开始时间不能是第一天，因为是左开右闭区间
        if half_year_count == 0:
            start_time = start_time - 1
        half_year_mask = (timestamps > start_time) & (timestamps <= end_time)  # 布尔数组，标记哪些交互在此半年内
        half_year_indices = np.where(half_year_mask)[0]  # 这些布尔位置的具体索引（在 group 内的位置）
        # 当前半年有交互，意味着最后一位肯定是valid edge
        if len(half_year_indices) != 0:
            valid_pos = half_year_indices[-1]
            # assert labels[valid_pos] != 0, f'final interact label is 0 in {half_year_count+1} half year'
            # 统计并获取这半年数据
            # 当前半年的交互数
            current_popularity = len(half_year_indices)
            # 当前半年的用户数
            current_users = users[half_year_indices]  # 当前半年每次交互的用户(有重复)
            current_users_unique = set(current_users)  # 当前半年唯一化用户（无重复）
            current_num_users = len(current_users_unique)  # 用户数（个数）
            # 当前半年用户中重复交互的用户数
            unique, counts = np.unique(current_users, return_counts=True)  # 统计每个用户的出现次数
            assert len(unique) == current_num_users, f'在第{half_year_count}半年，用户数统计过程中前后不一致'
            duplicate_mask = counts > 1  # 找出出现次数 > 1的用户（即重复的用户）
            current_num_duplicate_users = np.sum(duplicate_mask)  # 有重复的 users 的个数
            # 当前半年交互中重复发生的交互数
            current_num_duplicates = np.sum(counts[duplicate_mask] - 1)
            # 当前半年最后一次交互到现在的时间间隔
            days = (end_time - timestamps[valid_pos]) / (3600 * 24)
            if before_users.size > 0:
                # 当前半年用户中之前交互过的用户数
                user_count_dict = dict(zip(unique, counts))
                common_users = np.intersect1d(current_users, before_users)
                num_common_users = len(common_users)
                if num_common_users > 0:
                    # 当前半年交互中之前发生过的交互数
                    total_occurrences = sum(user_count_dict[user] for user in common_users)
                else:
                    total_occurrences = 0
            else:
                num_common_users = 0
                total_occurrences = 0
            # 将此半年的交互加入历史
            before_users = np.concatenate((before_users, np.array(list(current_users_unique))))
        else:
            current_popularity = 0
            current_num_users = 0
            current_num_duplicate_users = 0
            current_num_duplicates = 0
            days = 0
            num_common_users = 0
            total_occurrences = 0

        # 将此半年的特征写入
        new_row = np.array([current_popularity, current_num_users,
                            current_num_duplicate_users, current_num_duplicates,
                            days, num_common_users, total_occurrences])
        # print(f'half year num : {half_year_count+1}')
        features = np.vstack([features, new_row])

        # 有popularity真值，构造一条数据
        if len(half_year_indices) != 0:
            valid_pos = half_year_indices[-1]
            if labels[valid_pos] != 0:
                valid_pos = half_year_indices[-1]
                t = timestamps[valid_pos]
                l = labels[valid_pos]
                edge_idx = idxs[valid_pos]
                features_list.append(features[-5:, :])
                labels_list.append(l)
                time_list.append(t)
                idx_list.append(edge_idx)
                # print(features[-5:, :])
                # print(f'next year popularity {l}')

        start_time = end_time
        half_year_count += 1
    # exit()

print(f'train data edge num is {len(data_train)}')
print(f'val data edge num is {len(data_val)}')
print(f'test data edge num is {len(data_test)}')
print(f'all data edge num is {len(data_all)}')

# 打包、排序
sorted_all = sorted(zip(idx_list, features_list, labels_list, time_list), key=lambda x: x[0])
# 解包排序结果
idx_list_sorted, features_list_sorted, labels_list_sorted, time_list_sorted = zip(*sorted_all)
# 转成 list（因为 zip 输出的是元组）
idx_list = list(idx_list_sorted)
features_list = list(features_list_sorted)
labels_list = list(labels_list_sorted)
time_list = list(time_list_sorted)
# 检查时间是否升序
assert all(time_list[i] <= time_list[i + 1] for i in range(len(time_list) - 1)), "time_list is not sorted"

features_array = np.stack(features_list, axis=0)
labels_array = np.array(labels_list)
time_array = np.array(time_list)
print(f'feature的shape为{features_array.shape}')
print(f'label的shape为{labels_array.shape}')
print(f'time的shape为{time_array.shape}')
train_end_time = start_time_all + (front_half_year_seconds + behind_half_year_seconds) * 6 + front_half_year_seconds
val_end_time = train_end_time + behind_half_year_seconds

# 拆分条件
train_mask = time_array <= train_end_time
val_mask = (time_array > train_end_time) & (time_array <= val_end_time)
test_mask = time_array > val_end_time
# 拆分特征与标签
x_train = features_array[train_mask]
y_train = labels_array[train_mask]
x_val = features_array[val_mask]
y_val = labels_array[val_mask]
x_test = features_array[test_mask]
y_test = labels_array[test_mask]
print(f'Train set: {x_train.shape}, {y_train.shape}')
print(f'Val set: {x_val.shape}, {y_val.shape}')
print(f'Test set: {x_test.shape}, {y_test.shape}')


# 保存特征
np.save('../processed_data/aminer/features_train.npy', x_train)
np.save('../processed_data/aminer/features_val.npy', x_val)
np.save('../processed_data/aminer/features_test.npy', x_test)
# 保存标签
np.save('../processed_data/aminer/labels_train.npy', y_train)
np.save('../processed_data/aminer/labels_val.npy', y_val)
np.save('../processed_data/aminer/labels_test.npy', y_test)