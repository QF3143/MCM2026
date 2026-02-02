import pandas as pd
import numpy as np

def merge_mcm_data(fanvote_path, raw_data_path, output_path):
    # 1. 读取数据
    df_fanvote = pd.read_csv(fanvote_path)
    df_raw = pd.read_csv(raw_data_path)

    # 2. 预处理原始数据 (从宽格式转换为长格式)
    # 我们需要提取选手的静态特征（如行业、年龄、舞伴）以及每一周的裁判打分
    
    # 定义静态特征列
    static_cols = [
        'celebrity_name', 'season', 'ballroom_partner', 'celebrity_industry', 
        'celebrity_age_during_season', 'celebrity_homestate', 
        'celebrity_homecountry/region', 'results', 'placement'
    ]

    long_data_list = []

    # 假设最多有11周（根据数据列名判断）
    # 遍历每一周，提取当周的裁判分数并计算总分
    for week_num in range(1, 12):
        # 构建当前周的裁判分数列名 (weekX_judgeY_score)
        judge_cols = [f'week{week_num}_judge{j}_score' for j in range(1, 5)]
        
        # 检查这些列是否存在于原始数据中
        existing_judge_cols = [c for c in judge_cols if c in df_raw.columns]
        
        if not existing_judge_cols:
            continue
            
        # 提取相关数据
        temp_df = df_raw[static_cols + existing_judge_cols].copy()
        
        # 计算当周裁判总分 (忽略NaN，如果全为NaN则结果为0或NaN)
        # min_count=1 确保如果整行都是空值，总分也是NaN
        temp_df['total_judge_score'] = temp_df[existing_judge_cols].sum(axis=1, min_count=1)
        
        # 添加'week'列
        temp_df['week'] = week_num
        
        # 只保留静态特征、周数和总分
        temp_df = temp_df[static_cols + ['week', 'total_judge_score']]
        
        long_data_list.append(temp_df)

    # 将所有周的数据拼接在一起
    df_raw_long = pd.concat(long_data_list, ignore_index=True)

    # 过滤掉裁判总分为0或空的行（这通常意味着选手在该周已淘汰或未参赛）
    df_raw_long = df_raw_long[df_raw_long['total_judge_score'] > 0]

    # 3. 合并数据
    # 为了合并，我们需要统一列名。df_fanvote中的'name'对应df_raw中的'celebrity_name'
    df_fanvote = df_fanvote.rename(columns={'name': 'celebrity_name'})

    # 确保用于合并的关键列（Key）数据类型一致
    key_cols = ['season', 'week', 'celebrity_name']
    for col in ['season', 'week']:
        df_fanvote[col] = df_fanvote[col].astype(int)
        df_raw_long[col] = df_raw_long[col].astype(int)

    # 执行合并 (使用inner join，确保每一行既有粉丝投票估算，又有原始背景信息)
    df_merged = pd.merge(df_fanvote, df_raw_long, on=key_cols, how='inner')

    # 4. 保存结果
    df_merged.to_csv(output_path, index=False)
    print(f"合并完成！文件已保存至: {output_path}")
    print(f"合并后数据形状: {df_merged.shape}")
    print("包含列名:", df_merged.columns.tolist())
    
    return df_merged

# 使用示例 (请确保文件名与你上传的一致)
if __name__ == "__main__":
    merged_df = merge_mcm_data(
        fanvote_path='QF‘s solution/Bayes_Elo/real_figures/real_fan_vote_estimates_weekly.csv', 
        raw_data_path='2026_MCM_Problem_C_Data.csv', 
        output_path='QF‘s solution/Q3_bayes/merged_data_for_q3.csv'
    )