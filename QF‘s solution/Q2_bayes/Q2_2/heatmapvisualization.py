import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# 1. 加载数据 (请确保路径正确，注意文件名中的引号字符)
file_path = 'QF‘s solution/Q2_bayes/Q2_2/simulation_results_4_rules.csv'
try:
    df_results = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    df_results = pd.DataFrame() # 兜底逻辑

# 2. 目标人物与赛制定义
target_people = [
    ('Jerry Rice', 2, 'Sim_Rank_NoSave'),
    ('Bristol Palin', 11, 'Sim_Pct_NoSave'),
    ('Bobby Bones', 27, 'Sim_Pct_NoSave')
]

# 这里调整为 1x3 重点对比的三个核心维度
rules_map = {
    'Sim_Rank_Save': 'Rank (+ Judges\' Save)', 
    'Sim_Rank_NoSave': 'Rank (No Save)',       
    'Sim_Pct_NoSave': 'Percentage (No Save)'   
}

season_max_weeks = {2: 10, 11: 10, 27: 10}
colors = ['#E74C3C', '#2ECC71', '#BDC3C7'] # 红色:淘汰, 绿色:晋级, 灰色:赛季结束
cmap = sns.color_palette(colors)

# 3. 创建画布
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 4. 绘图循环
for i, (name, season, real_rule_col) in enumerate(target_people):
    ax = axes[i]
    if df_results.empty: continue
    
    row = df_results[df_results['Name'] == name].iloc[0]
    max_w = season_max_weeks.get(season, 10)
    
    matrix_data = []
    for col_name in rules_map.keys():
        survived_until = row[col_name]
        matrix_data.append([1 if w <= survived_until else 0 if w <= max_w else 2 for w in range(1, 11)])
    
    matrix_df = pd.DataFrame(matrix_data, index=rules_map.values(), columns=range(1, 11))
    
    sns.heatmap(matrix_df, ax=ax, cmap=cmap, cbar=False, linewidths=1.5, linecolor='white', vmin=0, vmax=2)
    
    # 动态标记历史真实规则 (蓝色边框)
    historical_label = rules_map[real_rule_col]
    for idx, label in enumerate(list(rules_map.values())):
        if label == historical_label:
            ax.get_yticklabels()[idx].set_weight('bold')
            ax.get_yticklabels()[idx].set_color('blue')
            rect = plt.Rectangle((0, idx), 10, 1, fill=False, edgecolor='blue', lw=3.5, clip_on=False)
            ax.add_patch(rect)
    
    ax.set_title(f"{name} (S{season})", fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('Week', fontsize=12)

# 5. 图例设计 (关键部分)
legend_patches = [
    mpatches.Patch(color=colors[1], label='Survived (晋级)'),
    mpatches.Patch(color=colors[0], label='Eliminated (淘汰)'),
    mpatches.Patch(color=colors[2], label='Season Ended (赛季结束)'),
    mpatches.Patch(edgecolor='blue', facecolor='none', linewidth=2, label='Actual Historical Rule (历史真实规则)')
]

# 使用 fig.legend 确保图例在所有子图正下方居中
fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=4, fontsize=12, frameon=True, facecolor='white', edgecolor='lightgrey')

# 调整整体布局，为底部的 legend (0.02) 留出空间 (bottom=0.15)
plt.suptitle("Survival Impact Analysis: Rules Comparison & Judges' Save Influence", 
             fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.08, 1, 0.93]) 

# 保存并展示
plt.savefig('survival_heatmap_1x3_with_legend.png', dpi=300, bbox_inches='tight')
plt.show()