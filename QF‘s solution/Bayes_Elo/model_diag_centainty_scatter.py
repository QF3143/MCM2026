import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 加载数据
df = pd.read_csv('QF‘s solution/Bayes_Elo/real_figures/real_fan_vote_estimates_weekly.csv')

# 2. 聚合每周统计数据
weekly_stats = df.groupby(['season', 'week']).agg(
    n_remaining=('name', 'count'),
    certainty=('certainty_score', 'mean')
).reset_index()

# 剔除空值
weekly_stats = weekly_stats.dropna(subset=['certainty'])

# 3. 设置学术绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# 4. 创建可视化图像
fig, ax = plt.subplots(figsize=(10, 6))

# 对 Y 轴进行对数变换处理 (Log10)
# 这能拉伸底部密集的数据点，使模型在不同人数下的预测灵敏度对比更明显
weekly_stats['log_certainty'] = np.log10(weekly_stats['certainty'])

# 绘制带有趋势线的回归图
sns.regplot(
    x='n_remaining', 
    y='log_certainty', 
    data=weekly_stats, 
    ax=ax,
    scatter_kws={'alpha':0.4, 's':40, 'color': '#2E86C1'},
    line_kws={'color': '#E67E22', 'linewidth': 2.5},
    label='Log-Linear Trend'
)

# 5. 美化纵轴标签：将对数刻度映射回原始物理数值，增强可读性
yticks = [-4, -3, -2, -1, 0]
ax.set_yticks(yticks)
ax.set_yticklabels(['0.0001', '0.001', '0.01', '0.1', '1.0'])

# 6. 图表装饰
ax.set_xlabel('Number of Remaining Contestants ($n$)', fontsize=12, fontweight='bold')
ax.set_ylabel('Certainty Score (Logarithmic Scale)', fontsize=12, fontweight='bold')
ax.set_title('Inference Precision Dynamics: Contestant Density vs. Certainty', fontsize=14, fontweight='bold', pad=15)

# 添加皮尔逊相关系数标注 (在对数空间下的相关性)
corr = weekly_stats['n_remaining'].corr(weekly_stats['log_certainty'])
ax.text(0.05, 0.05, f'Pearson $r$ (log-scale) = {corr:.3f}', 
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('QF‘s solution/Bayes_Elo/Model_diag_figure/certainty_vs_n_log_scale.png', dpi=300)
plt.show()