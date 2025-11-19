import pandas as pd
import numpy as np

half_year_seconds = 60 * 60 * 24 * 30 * 6
one_season_seconds = 60 * 60 * 24 * 30 * 3

df = pd.read_csv('../DG_data/aminer/ml_aminer.csv')
df['label'] = 0
end_time = df['ts'].values[-1]
num = 0

# 分别对每个 item 进行处理
for item_id, group in df.groupby('i'):
    group = group.reset_index()     # 保留原始索引
    timestamps = group['ts'].values     # 当前 item 的所有交互时间（按顺序）
    original_indices = group['index'].values    # 这些行在原始 DataFrame 中的行号
    print(f'Processing the {item_id}-th item, with a total of {len(group)} interactions.')
    # 从当前 item 的第一个交互开始
    current_pos = 0
    # 主循环：只处理“未来半年”的首个交互
    while current_pos < len(timestamps):
        print(f'Current position is {current_pos}---->{len(timestamps)}')
        current_time = timestamps[current_pos]  # 本轮要计算 popularity 的时间点
        window_end = current_time + half_year_seconds   # 当前交互的“未来半年”的结束时间点
        if window_end > end_time:     #最后时间不足空窗期的直接舍弃掉
            break
        # print(f'[{current_time},{window_end})')
        future_mask = (timestamps >= current_time) & (timestamps < window_end)  # 布尔数组，标记哪些交互在未来半年内
        future_indices = np.where(future_mask)[0]   # 这些布尔位置的具体索引（在 group 内的位置）
        # print(future_indices)
        # print(f'real time[{timestamps[future_indices[0]]},{timestamps[future_indices[-1]]}])')
        if len(future_indices) == 0:
            print('Something went wrong!!!')
            exit()
        elif len(future_indices) == 1:
            print('this period has only one interaction')
            popularity = 1
            print(popularity)
            # if current_pos == 0:
            #     print('Node first appear, disregard')
            # else:
            #     df.loc[original_indices[current_pos], 'label'] = popularity  # 把这次“未来半年”统计窗口的 label 写入原始 DataFrame
            current_pos = current_pos + 1
        else:
            popularity = len(future_indices)    # 未来半年内的交互次数（即你要预测的标签）
            print(popularity)
            if current_pos == 0:
                print('Node first appear, disregard')
            else:
                df.loc[original_indices[current_pos], 'label'] = popularity     # 把这次“未来半年”统计窗口的 label 写入原始 DataFrame
            # current_pos = future_indices[-1]    # 未来半年中最后一次交互
            current_pos = future_indices[-1] + 1  # 未来半年后第一次交互
        if current_pos != 0:
            num = num + 1
        print('----------------------------')
    print('======================================')


# df.to_csv('../processed_data/aminer/pp_aminer_one_season.csv')
df.to_csv('../processed_data/aminer/pp_aminer_half_year.csv')
print(f'valid edge num is {num}')