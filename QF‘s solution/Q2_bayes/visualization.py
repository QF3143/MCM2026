import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# 1. 加载模拟结果数据
df_results = pd.read_csv('QF‘s solution/Q2_bayes/Q2_2/simulation_results_4_rules.csv')

# 2. 定义目标人物
target_people = [
    ('Jerry Rice', 2, 'Sim_Rank_NoSave'),
    ('Bristol Palin', 11, 'Sim_Pct_NoSave'),
    ('Bill Engvall', 17, 'Sim_Pct_NoSave'),
    ('Bobby Bones', 27, 'Sim_Pct_NoSave')
]

rules_map = {
    'Sim_Rank_NoSave': 'Rank (No Save)',
    'Sim_Pct_NoSave': 'Percentage (No Save)',
    'Sim_Rank_Save': 'Rank (+ Judges\' Save)',
    'Sim_Pct_Save': 'Percentage (+ Judges\' Save)'
}

season_max_weeks = {2: 10, 11: 10, 17: 11, 27: 10}

# 3. 设置绘图风格
sns.set_theme(style="white")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 稍微缩小宽度，避免拉伸

# 减少子图间距
plt.subplots_adjust(hspace=0.25, wspace=0.45)  # ← 关键：减小空白

colors = ['#E74C3C', '#2ECC71', '#BDC3C7']
cmap = sns.color_palette(colors)

# 4. 绘制每个热力图
for i, (name, season, real_rule_col) in enumerate(target_people):
    ax = axes[i // 2, i % 2]
    row = df_results[df_results['Name'] == name].iloc[0]
    max_w = season_max_weeks.get(season, 11)
    
    matrix_data = []
    for col_name in rules_map.keys():
        survived_until = row[col_name]
        row_status = []
        for w in range(1, 12):
            if w > max_w:
                row_status.append(2)
            elif w <= survived_until:
                row_status.append(1)
            else:
                row_status.append(0)
        matrix_data.append(row_status)
    
    matrix_df = pd.DataFrame(matrix_data, index=rules_map.values(), columns=range(1, 12))
    
    # 绘制热力图：关闭 square 以更好适应空间
    sns.heatmap(
        matrix_df, ax=ax, cmap=cmap, cbar=False,
        linewidths=1.5, linecolor='white',
        square=False,  # ← 允许非正方形单元格，更紧凑
        vmin=0, vmax=2,
        annot=False
    )
    
    # 标注实际规则行
    real_rule_label = rules_map[real_rule_col]
    for idx, label in enumerate(list(rules_map.values())):
        if label == real_rule_label:
            ax.get_yticklabels()[idx].set_weight('bold')
            ax.get_yticklabels()[idx].set_color('blue')
            rect = plt.Rectangle((0, idx), 11, 1, fill=False, edgecolor='blue', lw=2, clip_on=False)
            ax.add_patch(rect)
    
    # 设置标题（减小 padding）
    ax.set_title(f"{name} (S{season})", fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel('')
    ax.set_xlabel('Week' if i >= 2 else '')
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

# 5. 图例（放在图下方，避免占用顶部空间）
legend_patches = [
    mpatches.Patch(color=colors[1], label='Survived'),
    mpatches.Patch(color=colors[0], label='Eliminated'),
    mpatches.Patch(color=colors[2], label='Season Ended'),
    mpatches.Patch(edgecolor='blue', facecolor='none', linewidth=2, label='Actual Historical Rule')
]

# 将图例放在底部，留出顶部给主标题
fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=4, fontsize=10, frameon=False)

# 主标题（降低 y 位置，避免与子图重叠）
plt.suptitle("Counterfactual Survival Under Different Voting Rules", 
             fontsize=16, fontweight='bold', y=0.96)

# 保存（注意 bbox_inches='tight' 会自动裁剪多余空白）
plt.savefig('QF‘s solution/Q2_bayes/Q2_2/survival_heatmap_2x2.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()
print("2x2 热力图已生成并保存为 survival_heatmap_2x2.png")