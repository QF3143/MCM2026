import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# --- 1. Academic Style Configuration ---
# 设置符合学术出版标准的字体和样式
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['figure.dpi'] = 300  # High resolution for paper

# --- 2. Data Loading & Preprocessing ---

# (A) Load Method 1: Percentage Rule
df_percent = pd.read_csv('QF‘s solution/Q2_QFv2/percent/percentage_pridict.csv')
df_rank = pd.read_csv('QF‘s solution/Q2_QFv2/rank/rank_pridict.csv')

# (B) Load Method 2: Rank Rule
# [CRITICAL USER ACTION]: Replace the following Simulation Block with your actual file loading
# Example: df_rank = pd.read_csv('rank_predict.csv')

# (C) Merge Dataframes for Comparative Analysis
# Aligning data based on Contestant, Season, and Week
common_cols = ['Contestant', 'Season', 'Week']
df_merged = pd.merge(
    df_percent[common_cols + ['Est_Fan_Share', 'Fan_Share_CI_Lower', 'Fan_Share_CI_Upper']],
    df_rank[common_cols + ['Est_Fan_Share', 'Fan_Share_CI_Lower', 'Fan_Share_CI_Upper']],
    on=common_cols,
    suffixes=('_Percent', '_Rank')
)

def assign_mechanism(season):
    if 1 <= season <= 2 or season >= 28:
        return 'Actual: Rank Rule'
    else:
        return 'Actual: Percentage Rule'
    
df_merged['Mechanism_Label'] = df_merged['Season'].apply(assign_mechanism)
# --- 3. Visualization ---

# 定义颜色映射：蓝色代表排名制，黄色代表百分比制
color_palette = {
    'Actual: Rank Rule': '#3498db',        # Academic Blue
    'Actual: Percentage Rule': '#f1c40f'   # Academic Gold
}
# Plot 1: Methodological Agreement (Scatter Plot)
plt.figure(figsize=(8, 7))

# 定义颜色映射：蓝色代表排名制，黄色代表百分比制
color_palette = {
    'Actual: Rank Rule': '#3498db',        # Academic Blue
    'Actual: Percentage Rule': '#f1c40f'   # Academic Gold
}

# 绘制散点图
sns.scatterplot(
    data=df_merged, 
    x='Est_Fan_Share_Percent', 
    y='Est_Fan_Share_Rank', 
    hue='Mechanism_Label',
    palette=color_palette,
    alpha=0.7, 
    edgecolor='w',
    linewidth=0.5,
    s=45,
    style='Mechanism_Label' # 增加形状区分，确保黑白打印时依然可读
)

# 绘制对角线 (The Theoretical Identity Line)
# 如果模型估算完全一致，所有点应落在该线上
max_val = max(df_merged['Est_Fan_Share_Percent'].max(), df_merged['Est_Fan_Share_Rank'].max())
plt.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', linewidth=1.2, label='Perfect Agreement ($y=x$)')

# 图表精修
plt.title('Figure 1: Cross-Method Robustness & Historical Mechanism Comparison', fontweight='bold')
plt.xlabel('Estimated Fan Share $\hat{P}_{i,t}$ (Percentage Mapping)', fontsize=10)
plt.ylabel('Estimated Fan Share $\hat{R}_{i,t}$ (Rank Mapping)', fontsize=10)

# 优化图例
plt.legend(title='Voting Era', frameon=True, loc='upper left', fontsize='small')
plt.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('QF‘s solution/Q2_QFv2/visualization/Comparison_Scatter_Colored.png')
plt.show()

# Plot 2: Micro-Level Discrepancy with Error Bars (Bar Plot)
# Focusing on a specific week to show granular differences
target_season = 1
target_week = 2
df_subset = df_merged[(df_merged['Season'] == target_season) & (df_merged['Week'] == target_week)].copy()

if not df_subset.empty:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_subset))
    width = 0.35
    
    # Method 1 Bars
    plt.bar(x - width/2, df_subset['Est_Fan_Share_Percent'], width, 
            yerr=[df_subset['Est_Fan_Share_Percent'] - df_subset['Fan_Share_CI_Lower_Percent'], 
                  df_subset['Fan_Share_CI_Upper_Percent'] - df_subset['Est_Fan_Share_Percent']],
            label='Percentage Rule', color='#3498db', capsize=4, alpha=0.9, ecolor='black')
            
    # Method 2 Bars
    plt.bar(x + width/2, df_subset['Est_Fan_Share_Rank'], width, 
            yerr=[df_subset['Est_Fan_Share_Rank'] - df_subset['Fan_Share_CI_Lower_Rank'], 
                  df_subset['Fan_Share_CI_Upper_Rank'] - df_subset['Est_Fan_Share_Rank']],
            label='Rank Rule', color='#e67e22', capsize=4, alpha=0.9, ecolor='black')

    plt.xticks(x, df_subset['Contestant'], rotation=45, ha='right')
    plt.ylabel('Estimated Fan Vote Share')
    plt.title(f'Figure 2: Divergence in Fan Support Estimation (Season {target_season}, Week {target_week})')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('QF‘s solution/Q2_QFv2/visualization/Comparison_Bar_CI.png')
    plt.show()

# Plot 3: Distribution Density (KDE)
plt.figure(figsize=(8, 5))
sns.kdeplot(df_merged['Est_Fan_Share_Percent'], fill=True, color='#3498db', label='Percentage Rule', alpha=0.3)
sns.kdeplot(df_merged['Est_Fan_Share_Rank'], fill=True, color='#e67e22', label='Rank Rule', alpha=0.3)
plt.title('Figure 3: Probability Density Function of Estimated Shares')
plt.xlabel('Fan Vote Share')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('QF‘s solution/Q2_QFv2/visualization/Comparison_Distribution.png')
plt.show()