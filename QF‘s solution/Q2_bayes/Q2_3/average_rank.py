import pandas as pd
import numpy as np

# 数据导入：加载估算结果与原始竞赛数据
estimates = pd.read_csv('QF‘s solution/Bayes_Elo/real_figures/fan_vote_estimates_weekly.csv')
raw_data = pd.read_csv('2026_MCM_Problem_C_Data.csv')

# 1. 执行每周内的秩变换
# 在每个赛季的每一周内，根据裁判得分百分数与粉丝投票百分数进行排名
# 采用 'min' 策略处理并列情况，排名 1 代表表现最优
estimates['judge_rank'] = estimates.groupby(['season', 'week'])['judge_pct'].rank(
    ascending=False, method='min'
)
estimates['fan_rank'] = estimates.groupby(['season', 'week'])['est_fan_pct'].rank(
    ascending=False, method='min'
)

# 2. 聚合个体表现指标
# 计算每位选手在参赛期间的平均排名，作为其技术水平与人气水平的度量标准
contestant_metrics = estimates.groupby(['season', 'name']).agg(
    avg_judge_rank=('judge_rank', 'mean'),
    avg_fan_rank=('fan_rank', 'mean'),
    total_weeks=('week', 'count')
).reset_index()

# 3. 关联原始终名次 (Placement)
# 提取原始数据中的名次信息，用于后续的一致性验证（Consistency Analysis）
final_placements = raw_data[['celebrity_name', 'season', 'placement']].drop_duplicates()
summary_table = pd.merge(
    contestant_metrics, 
    final_placements, 
    left_on=['name', 'season'], 
    right_on=['celebrity_name', 'season'], 
    how='left'
).drop(columns=['celebrity_name'])

# 4. 结果持久化
# 按照赛季与裁判排名排序以增强表格可读性
summary_table = summary_table.sort_values(by=['season', 'avg_judge_rank'])
summary_table.to_csv('QF‘s solution/Q2_bayes/Q2_3/contestant_rankings_summary.csv', index=False)

# 输出前 5 行样本以供验证
print(summary_table.head())