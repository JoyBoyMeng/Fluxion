import pandas as pd
import numpy as np

data_all = pd.read_csv('../DG_data/aminer/ml_aminer.csv')
start_time_all = data_all['ts'].values[0]
end_time_all = data_all['ts'].values[-1]
print(start_time_all)
print(end_time_all)
print((end_time_all - start_time_all) / 3600 / 24)
print(f'edge num is {len(data_all)}')

front_half_year_seconds = 60 * 60 * 24 * (31+28+31+30+31+30)
behind_half_year_seconds = 60 * 60 * 24 * (31+31+30+31+30+31)
valid_edge = 0

# 分别对每个 item 进行处理
for item_id, group in data_all.groupby('i'):
    group = group.reset_index()  # 保留原始索引
    timestamps = group['ts'].values  # 当前 item 的所有交互时间（按顺序）
    original_indices = group['index'].values  # 这些行在原始 DataFrame 中的行号
    print(f'Processing the {item_id}-th item, with a total of {len(group)} interactions.')
    half_year_count = 0
    current_pos = -1
    start_time = start_time_all
    # 主循环：计算本个半年的交互次数，存入上一次计算 popularity 的时间点，并计算下一次写入 popularity 的时间点
    while start_time + front_half_year_seconds <= end_time_all:
        # 判定是上半年还是下半年
        if half_year_count % 2 == 0:
            end_time = start_time + front_half_year_seconds
        else:
            end_time = start_time + behind_half_year_seconds
        # 第一个半年开始时间不能是第一天，因为是左开右闭区间
        if half_year_count == 0:
            start_time = start_time - 1
        half_year_mask = (timestamps > start_time) & (timestamps <= end_time)  # 布尔数组，标记哪些交互在此半年内
        half_year_indices = np.where(half_year_mask)[0]  # 这些布尔位置的具体索引（在 group 内的位置）
        # 不需要写入
        if current_pos == -1:
            # 下一半年需要写入
            if len(half_year_indices) != 0:
                current_pos = half_year_indices[-1]
            print('no popularity this half year')
        # 需要写入
        else:
            # 写入值，计算下一个半年需要写入的位置
            if len(half_year_indices) != 0:
                popularity = len(half_year_indices)
                data_all.loc[original_indices[current_pos], 'label'] = popularity  # 把这次“未来半年”统计窗口的 label 写入原始 DataFrame
                current_pos = half_year_indices[-1]
                print(f'popularity is {popularity}')
                valid_edge += 1
            # 写入0，下一半年不需要写入
            else:
                current_pos = -1
                print('no popularity this half year')
        # 推移时间
        start_time = end_time
        half_year_count += 1
        print('----------------------------')
    print('======================================')
print(f'edge num is {len(data_all)}')
print(f'valid edge num is {valid_edge}')
train_end_time = start_time_all + (front_half_year_seconds + behind_half_year_seconds) * 6 + front_half_year_seconds
val_end_time = train_end_time + behind_half_year_seconds
data_train = data_all[data_all['ts'] <= train_end_time]
data_val = data_all[(data_all['ts'] > train_end_time) & (data_all['ts'] <= val_end_time)]
data_test = data_all[data_all['ts'] > val_end_time]
print(f'train data edge num is {len(data_train)}')
print(f'val data edge num is {len(data_val)}')
print(f'test data edge num is {len(data_test)}')
print(f'all data edge num is {len(data_all)}')

valid_data_train = data_train[data_train['label'] != 0]
print(f'train data valid edge num is {len(valid_data_train)}')
valid_data_val = data_val[data_val['label'] != 0]
print(f'val data valid edge num is {len(valid_data_val)}')
valid_data_test = data_test[data_test['label'] != 0]
print(f'test data valid edge num is {len(valid_data_test)}')
valid_data_all = data_all[data_all['label'] != 0]
print(f'all data valid edge num is {len(valid_data_all)}')

data_train.to_csv('../processed_data/aminer/pp_aminer_half_year_train.csv', index=False)
data_val.to_csv('../processed_data/aminer/pp_aminer_half_year_val.csv', index=False)
data_test.to_csv('../processed_data/aminer/pp_aminer_half_year_test.csv', index=False)
data_all.to_csv('../processed_data/aminer/pp_aminer_half_year_all.csv', index=False)