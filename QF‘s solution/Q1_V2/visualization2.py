import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载与预处理
df = pd.read_csv('QF‘s solution/Q1_V2/Problem_C_Solution_Advanced.csv')

# 计算置信区间宽度 (CI Width) 作为不确定性的直观度量
df['CI_Width'] = df['Fan_Share_CI_Upper'] - df['Fan_Share_CI_Lower']

# 设置绘图风格，保持学术严谨性 (Academic Style)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# -----------------------------------------------------------
# 子图 1: 不确定性分布 (稳定性评价)
# -----------------------------------------------------------
sns.histplot(df['Fan_Share_Std'], kde=True, color='#2c3e50', ax=axes[0, 0], alpha=0.6)
axes[0, 0].set_title(r'$\bf{Distribution\ of\ Model\ Uncertainty}$ (Standard Deviation)', fontsize=12)
axes[0, 0].set_xlabel('Fan Share Standard Deviation')
axes[0, 0].set_ylabel('Frequency')
# 添加辅助线：中位数
med_std = df['Fan_Share_Std'].median()
axes[0, 0].axvline(med_std, color='#e74c3c', linestyle='--', label=f'Median Std: {med_std:.3f}')
axes[0, 0].legend()

# -----------------------------------------------------------
# 子图 2: 估计值与误差范围的一致性 (一致性评价)
# -----------------------------------------------------------
sns.scatterplot(x='Est_Fan_Share', y='CI_Width', data=df, alpha=0.4, color='#2980b9', ax=axes[0, 1], s=15)
# 添加趋势线
sns.regplot(x='Est_Fan_Share', y='CI_Width', data=df, scatter=False, color='#e74c3c', ax=axes[0, 1])
axes[0, 1].set_title(r'$\bf{Consistency\ Check:\ Estimate\ vs.\ CI\ Width}$', fontsize=12)
axes[0, 1].set_xlabel('Estimated Fan Share')
axes[0, 1].set_ylabel('95% CI Width')

# -----------------------------------------------------------
# 子图 3: 跨赛季的鲁棒性 (跨域稳定性)
# -----------------------------------------------------------
# 筛选前15个赛季以避免拥挤，或者每隔几个赛季采样
seasons_to_plot = df['Season'].unique()[:20] 
df_subset = df[df['Season'].isin(seasons_to_plot)]

sns.boxplot(x='Season', y='Fan_Share_Std', data=df_subset, color='#95a5a6', ax=axes[1, 0], fliersize=1)
axes[1, 0].set_title(r'$\bf{Robustness\ Across\ Seasons}$ (Volatility Boxplot)', fontsize=12)
axes[1, 0].set_xlabel('Season')
axes[1, 0].set_ylabel('Fan Share Standard Deviation')

# -----------------------------------------------------------
# 子图 4: 模拟收敛性诊断
# -----------------------------------------------------------
sns.scatterplot(x='Valid_Simulations', y='Fan_Share_Std', data=df, alpha=0.4, color='#27ae60', ax=axes[1, 1], s=15)
axes[1, 1].set_title(r'$\bf{Algorithmic\ Stability:\ Sample\ Size\ vs.\ Uncertainty}$', fontsize=12)
axes[1, 1].set_xlabel('Number of Valid Simulations')
axes[1, 1].set_ylabel('Fan Share Standard Deviation')

# 保存图片
plt.savefig('QF‘s solution/Q1_V2/Visualization/Model_Stability_Consistency_Analysis.png', dpi=300, bbox_inches='tight')

# 不使用 plt.show()，直接保存