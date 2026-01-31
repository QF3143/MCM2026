import pandas as pd
import numpy as np

# 1. 加载数据
df_main_raw = pd.read_csv('2026_MCM_Problem_C_Data.csv')
df_fan_share = pd.read_csv('QF‘s solution/Q3_Q1v2/Problem_C_Solution_Advanced.csv')

# --- 步骤 1: 将主表转换为长表 (Wide to Long) ---
# 我们需要提取每周的裁判得分并汇总
id_vars = [
    'celebrity_name', 'ballroom_partner', 'celebrity_industry', 
    'celebrity_age_during_season', 'season'
]

# 提取所有包含 'judge' 和 'score' 的列
score_cols = [col for col in df_main_raw.columns if 'judge' in col and 'score' in col]

# 使用 melt 转换
df_long = df_main_raw.melt(
    id_vars=id_vars,
    value_vars=score_cols,
    var_name='raw_week_info',
    value_name='judge_score'
)

# 从 'week1_judge1_score' 中提取周数 (Week)
df_long['week'] = df_long['raw_week_info'].str.extract('week(\d+)').astype(int)

# 汇总每周的总分 (因为原表有 3-4 个裁判的分数)
df_main = df_long.groupby(id_vars + ['week'])['judge_score'].sum().reset_index()
df_main = df_main.rename(columns={'judge_score': 'total_judge_score'})

# --- 步骤 2: 准备并合并新表格 ---
# 统一列名以匹配主表
df_fan_share = df_fan_share.rename(columns={
    'Contestant': 'celebrity_name',
    'Season': 'season',
    'Week': 'week',
    'Est_Fan_Share': 'est_fan_share'
})

# 确保数据类型一致 (int)
df_main['season'] = df_main['season'].astype(int)
df_main['week'] = df_main['week'].astype(int)
df_fan_share['season'] = df_fan_share['season'].astype(int)
df_fan_share['week'] = df_fan_share['week'].astype(int)

# 执行合并
df = pd.merge(
    df_main, 
    df_fan_share[['celebrity_name', 'season', 'week', 'est_fan_share']], 
    on=['celebrity_name', 'season', 'week'], 
    how='left'
)

# --- 步骤 3: 核心特征工程 ---

# A. 职业舞伴能力指数 (Pro Efficacy Index)
# 计算该舞伴在历史上带出的平均分
df['Pro_Efficacy'] = df.groupby('ballroom_partner')['total_judge_score'].transform('mean')

# B. 行业降维 (Industry Grouping)
def map_industry(ind):
    if pd.isna(ind): return 'Other'
    ind = str(ind).lower()
    if any(x in ind for x in ['actor', 'actress', 'singer', 'musician', 'comedian', 'magician']):
        return 'Performing Arts'
    elif any(x in ind for x in ['athlete', 'nfl', 'nba', 'olympian', 'sports']):
        return 'Sports'
    elif any(x in ind for x in ['tv', 'reality', 'anchor', 'host', 'media']):
        return 'TV/Media'
    else:
        return 'Other'

df['Industry_Group'] = df['celebrity_industry'].apply(map_industry)

# C. 年龄非线性项
df['Age_Squared'] = df['celebrity_age_during_season'] ** 2

# D. 缺失值处理 (针对没有模拟数据的周)
df['est_fan_share'] = df['est_fan_share'].fillna(df['est_fan_share'].median())

# --- 步骤 4: 数据编码与输出 ---
df_encoded = pd.get_dummies(df, columns=['Industry_Group'], prefix='Ind')

# 最终特征列
features = [
    'celebrity_age_during_season', 'Age_Squared',
    'Pro_Efficacy', 'est_fan_share', 'season', 'week'
]
encoded_cols = [c for c in df_encoded.columns if c.startswith('Ind_')]
X = df_encoded[features + encoded_cols]

# 保存结果供后续建模使用
df_encoded.to_csv('QF‘s solution/Q3_Q1v2/Processed_Feature_Matrix.csv', index=False)
print("Merge successful. Processed data saved to 'Processed_Feature_Matrix.csv'.")