import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载与预处理
df = pd.read_csv('cleaned_weekly_data.csv')
# 过滤掉得分为0的数据（通常代表未参赛或已出局），以便更准确地观察竞技表现
active_df = df[df['avg_score'] > 0].copy()

# 2. 配色与风格设置
# 定义纯色：深蓝色和金黄色
blue_solid = "#005a9c"
yellow_solid = '#ffcc00'

# 设置Seaborn主题风格
sns.set_style("whitegrid", {'axes.grid': False})

# 创建 2x2 的画布
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')

# --- 子图 1: 选手得分频率分布 (直方图) ---
sns.histplot(active_df['avg_score'], bins=20, kde=False, ax=axes[0,0], 
             color=blue_solid, alpha=0.9, edgecolor='black')
axes[0,0].set_title('Weekly Average Score Distribution', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Average Score')
axes[0,0].set_ylabel('Frequency')

# --- 子图 2: 随比赛周次的得分提升趋势 (折线图) ---
sns.lineplot(data=active_df, x='week', y='avg_score', ax=axes[0,1], 
             color=blue_solid, marker='o', markerfacecolor=yellow_solid, 
             markeredgecolor='black', markersize=10, linewidth=3, errorbar=None)
axes[0,1].set_title('Mean Score Progression Over Weeks', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Week')
axes[0,1].set_ylabel('Average Score')
axes[0,1].set_xticks(range(1, 12))

# --- 子图 3: 各赛季得分表现对比 (蓝黄交替箱线图) ---
# 创建蓝黄交替的调色板
seasons = sorted(active_df['season'].unique())
season_colors = [blue_solid if i % 2 == 0 else yellow_solid for i in range(len(seasons))]

sns.boxplot(data=active_df, x='season', y='avg_score', ax=axes[1,0],
            palette=season_colors, fliersize=1)
axes[1,0].set_title('Score Distribution Across Seasons', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Season')
axes[1,0].set_ylabel('Average Score')
# 赛季较多，每5个赛季显示一个标签
axes[1,0].set_xticks(range(0, 35, 5))
axes[1,0].set_xticklabels([f'S{i}' for i in range(1, 36, 5)])

# --- 子图 4: 评委排名与得分的对应关系 (纯色箱线图) ---
sns.boxplot(data=active_df, x='judge_rank', y='avg_score', ax=axes[1,1],
            color=yellow_solid, fliersize=1)
# 给黄色箱体添加黑色轮廓增强质感
for patch in axes[1,1].patches:
    patch.set_edgecolor('black')

axes[1,1].set_title('Judge Rank vs. Average Score', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Judge Rank')
axes[1,1].set_ylabel('Average Score')

# 3. 全局细节优化
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

plt.tight_layout(pad=4.0)
plt.savefig('QF‘s solution/Clean_Data_visualization/solid_blue_yellow_analysis.png', dpi=300)