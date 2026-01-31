import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# 1. 加载数据 (假设你已有包含模拟结果的 DataFrame: df_results)
# 必须包含列: 'Name', 'Sim_Rank_Save', 'Sim_Rank_NoSave', 'Sim_Pct_NoSave'
file_path = 'simulation_results_4_rules.csv' 
try:
    df_results = pd.read_csv(file_path)
except FileNotFoundError:
    # 模拟数据仅用于绘图逻辑展示
    data = {
        'Name': ['Jerry Rice', 'Billy Ray Cyrus', 'Bristol Palin', 'Bobby Bones'],
        'Sim_Rank_Save': [8, 5, 6, 4],    # 模拟存活周数
        'Sim_Rank_NoSave': [8, 4, 5, 3],
        'Sim_Pct_NoSave': [8, 8, 10, 9]   # 历史实际存活周数
    }
    df_results = pd.DataFrame(data)

# 2. 定义目标人物、赛季及其对应的历史真实规则
target_people = [
    ('Jerry Rice', 2, 'Sim_Rank_NoSave'),       # S1-2 使用 Rank 且无 Save
    ('Billy Ray Cyrus', 4, 'Sim_Pct_NoSave'),    # S4 使用 Percentage
    ('Bristol Palin', 11, 'Sim_Pct_NoSave'),    # S11 使用 Percentage
    ('Bobby Bones', 27, 'Sim_Pct_NoSave')       # S27 使用 Percentage
]

# 定义要展示的三种核心投票机制
rules_map = {
    'Sim_Rank_Save': "Rank (+Judges Save)", 
    'Sim_Rank_NoSave': 'Rank',       
    'Sim_Pct_NoSave': 'Percentage'   
}

# 根据数据手册校准的赛季最大周数
season_max_weeks = {2: 8, 4: 10, 11: 10, 27: 9}
colors = ['#F39C12', '#2980B9', '#D6DBDF'] # 红色:淘汰, 绿色:晋级, 灰色:赛季结束
cmap = sns.color_palette(colors)

# 3. 创建 2x2 画布
sns.set_theme(style="white")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4. 绘图循环
for i, (name, season, real_rule_col) in enumerate(target_people):
    row_idx = i // 2
    col_idx = i % 2
    ax = axes[row_idx, col_idx]
    
    # 提取该名人的模拟生存数据
    person_row = df_results[df_results['Name'] == name].iloc[0]
    max_w = season_max_weeks.get(season, 10)
    
    matrix_data = []
    for rule_key in rules_map.keys():
        survived_until = person_row[rule_key]
        # 构建生存矩阵: 1-晋级, 0-淘汰, 2-赛季外
        matrix_row = [1 if w <= survived_until else 0 if w <= max_w else 2 for w in range(1, 11)]
        matrix_data.append(matrix_row)
    
    matrix_df = pd.DataFrame(matrix_data, index=rules_map.values(), columns=range(1, 11))
    
    # 绘制热力图
    sns.heatmap(matrix_df, ax=ax, cmap=cmap, cbar=False, linewidths=2, linecolor='white', vmin=0, vmax=2)
    
    # 动态标记并框选该名人在历史上的真实赛制 (蓝色边框)
    historical_label = rules_map[real_rule_col]
    for idx, label in enumerate(list(rules_map.values())):
        if label == historical_label:
            ax.get_yticklabels()[idx].set_weight('bold')
            ax.get_yticklabels()[idx].set_color('#1F77B4')
            rect = plt.Rectangle((0, idx), 10, 1, fill=False, edgecolor='#1F77B4', lw=4, clip_on=False)
            ax.add_patch(rect)
    
    ax.set_title(f"{name} (Season {season})", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Week', fontsize=12)
    if col_idx == 1: ax.set_ylabel('') # 优化内部标签显示

# 5. 图例与整体布局
legend_patches = [
    mpatches.Patch(color=colors[1], label='Survived'),
    mpatches.Patch(color=colors[0], label='Eliminated'),
    mpatches.Patch(color=colors[2], label='Season Finished'),
    mpatches.Patch(edgecolor='#1F77B4', facecolor='none', linewidth=2, label='Historical Rule Used')
]

fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0.03),
           ncol=4, fontsize=14, frameon=True, facecolor='white', edgecolor='lightgrey')

plt.suptitle("Survival Sensitivity Analysis: Impact of Voting Systems on Controversial Contestants", 
             fontsize=24, fontweight='bold', y=0.98)

# 为底部的图例留出空间
plt.tight_layout(rect=[0, 0.08, 1, 0.95]) 

# 保存结果
plt.savefig('survival_heatmap_2x2', dpi=300, bbox_inches='tight')