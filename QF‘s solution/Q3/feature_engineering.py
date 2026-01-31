import pandas as pd
import numpy as np

# 1. 加载合并后的数据
df = pd.read_csv('QF‘s solution/Q3/merged_data_for_q3.csv')

# --- 核心特征工程 ---

# A. 职业舞伴能力指数 (Pro Efficacy Index)
# 逻辑：计算该舞伴在所有赛季中的平均裁判得分 (Mean Judge Score)
# 注意：为了更严谨，可以使用 "Leave-One-Out" 编码（即计算除此之外的比赛均分），
# 但在此处，全局均分也是一个强有力的代理变量。
pro_efficacy = df.groupby('ballroom_partner')['total_judge_score'].transform('mean')
df['Pro_Efficacy'] = pro_efficacy

# B. 行业降维 (Industry Grouping)
# 将杂乱的行业归类为宏观组别，以便发现规律
def map_industry(ind):
    if pd.isna(ind): return 'Other'
    ind = str(ind).lower()
    
    # 定义映射规则
    if any(x in ind for x in ['actor', 'actress', 'singer', 'musician', 'comedian', 'magician', 'producer', 'rapper']):
        return 'Performing Arts' # 舞台表现力强
    elif any(x in ind for x in ['athlete', 'nfl', 'nba', 'olympian', 'racing', 'fitness', 'sports']):
        return 'Sports' # 体能好但可能僵硬
    elif any(x in ind for x in ['tv', 'reality', 'anchor', 'host', 'media', 'journalist', 'radio']):
        return 'TV/Media' # 脸熟，观众缘
    elif any(x in ind for x in ['model', 'beauty', 'fashion']):
        return 'Model/Fashion' # 外形优势
    else:
        return 'Other' # 政治家、企业家等

df['Industry_Group'] = df['celebrity_industry'].apply(map_industry)

# C. 年龄的非线性处理
# 假设：年龄的影响不是线性的（太老跳不动，太年轻没阅历）。
# 添加年龄的平方项，捕捉 "倒U型" 或 "U型" 关系
df['Age_Squared'] = df['celebrity_age_during_season'] ** 2

# --- 数据编码 (Encoding) ---

# D. 对行业组别进行 One-Hot 编码
df_encoded = pd.get_dummies(df, columns=['Industry_Group'], prefix='Ind')

# --- 准备最终模型输入 (X 和 Y) ---

# 定义特征列表
features = [
    'celebrity_age_during_season', 'Age_Squared',  # 人口统计学特征
    'Pro_Efficacy',                                # 舞伴效应 (关键!)
    'season', 'week'                               # 时间特征
] 
# 加入所有行业特征
features += [c for c in df_encoded.columns if 'Ind_' in c]

# 定义目标变量
target_judge = 'total_judge_score'  # 裁判看重技术
target_fan = 'est_fan_pct'          # 粉丝看重人气

# 创建最终用于建模的 DataFrame
X = df_encoded[features].copy()
y_judge = df_encoded[target_judge]
y_fan = df_encoded[target_fan]

# 检查结果
print("特征工程完成！")
print(f"特征矩阵 X 形状: {X.shape}")
print("特征列表:", features)
print("\n前5行预览:")
print(X.head())

# 保存处理好的特征矩阵（可选）
df_encoded.to_csv('QF‘s solution/Q3/feature_engineered_data.csv', index=False)