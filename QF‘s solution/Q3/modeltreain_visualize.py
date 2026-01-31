import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

# --- 0. 设置学术绘图风格 ---
plt.style.use('seaborn-v0_8-whitegrid')
# 如果没有安装 seaborn-v0_8-whitegrid，可以使用 'ggplot' 或 'seaborn'
# plt.style.use('ggplot') 
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体，更像学术论文
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
colors_judge = '#2E86C1' # 学术蓝
colors_fan = '#E67E22'   # 学术橙

# --- 1. 加载数据 ---
# 确保文件在当前目录下
df = pd.read_csv('QF‘s solution/Q3/feature_engineered_data.csv')

# 填补空值 (GBR能处理，但为了安全起见)
features = ['celebrity_age_during_season', 'Pro_Efficacy', 'Age_Squared'] + [c for c in df.columns if 'Ind_' in c]
X = df[features].fillna(0)
y_judge = df['total_judge_score']
y_fan = df['est_fan_pct']

# --- 2. 训练模型以获取特征重要性 ---
model_judge = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y_judge)
model_fan = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y_fan)

# 计算并归一化重要性
imp_df = pd.DataFrame({
    'Feature': features,
    'Judge': model_judge.feature_importances_,
    'Fan': model_fan.feature_importances_
})
imp_df['Judge'] = imp_df['Judge'] / imp_df['Judge'].sum() * 100
imp_df['Fan'] = imp_df['Fan'] / imp_df['Fan'].sum() * 100
imp_df = imp_df.sort_values('Judge') # 按裁判重要性排序

imp_df.to_csv("QF‘s solution/Q3/importance.csv")

# --- 3. 绘制蝴蝶图 (Butterfly Chart) ---
fig, ax = plt.subplots(figsize=(12, 7))
y = np.arange(len(imp_df))
height = 0.4

# 绘制水平条形图
rects1 = ax.barh(y + height/2, imp_df['Judge'], height, label='Impact on Judge Score (Technical)', color=colors_judge, alpha=0.85)
rects2 = ax.barh(y - height/2, imp_df['Fan'], height, label='Impact on Fan Votes (Popularity)', color=colors_fan, alpha=0.85)

# 装饰图表
ax.set_yticks(y)
# 美化标签名称
clean_labels = imp_df['Feature'].str.replace('Ind_', 'Industry: ').str.replace('celebrity_', '').str.replace('during_season', '').str.replace('_', ' ')
ax.set_yticklabels(clean_labels, fontsize=11)
ax.set_xlabel('Relative Feature Importance (%)', fontsize=12, fontweight='bold')
ax.set_title('Contrast of Drivers: Judges vs. Fans', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', frameon=True, fontsize=11)
ax.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('contrast_butterfly_chart.png', dpi=300)
plt.show()

# --- 4. 绘制年龄效应双轴图 (Age Effect Analysis) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# 数据聚合：计算每个年龄的平均分和平均票
# 为了曲线平滑，可以过滤掉极端稀疏的年龄点(如>80岁或<16岁)
age_data = df[df['celebrity_age_during_season'].between(16, 75)]
age_groups = age_data.groupby('celebrity_age_during_season')[['total_judge_score', 'est_fan_pct']].mean().reset_index()

# 设置双轴
ax2 = ax1.twinx()

# 绘制拟合曲线 (Order=2 代表二次多项式拟合，捕捉非线性)
sns.regplot(x='celebrity_age_during_season', y='total_judge_score', data=age_groups, ax=ax1, 
            scatter_kws={'alpha':0.4, 's':30}, line_kws={'color': colors_judge, 'linewidth': 2.5}, 
            order=2, label='Judge Score (Trend)', color=colors_judge, ci=None)

sns.regplot(x='celebrity_age_during_season', y='est_fan_pct', data=age_groups, ax=ax2, 
            scatter_kws={'alpha':0.4, 's':30}, line_kws={'color': colors_fan, 'linewidth': 2.5}, 
            order=2, label='Fan Vote (Trend)', color=colors_fan, ci=None)

# 装饰
ax1.set_xlabel('Celebrity Age', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Judge Score', color=colors_judge, fontsize=12, fontweight='bold')
ax2.set_ylabel('Estimated Fan Vote Share', color=colors_fan, fontsize=12, fontweight='bold')

# 设置刻度颜色
ax1.tick_params(axis='y', labelcolor=colors_judge)
ax2.tick_params(axis='y', labelcolor=colors_fan)

# 添加图例 (由于是双轴，需要手动添加)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# 注意：regplot默认不返回handle，这里简单处理标题
plt.title('Non-Linear Age Dynamics: Technical Decline vs. Fan Sympathy', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('age_dynamics_contrast.png', dpi=300)
plt.show()

print("图表已生成：'contrast_butterfly_chart.png' 和 'age_dynamics_contrast.png'")