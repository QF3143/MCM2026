import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# --- 1. Academic Style Configuration ---
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 11
rcParams['figure.dpi'] = 300 

# --- 2. Data Loading & Merging ---

# 加载两个模型的结果
df_percent = pd.read_csv('QF‘s solution/Q2_bayes/Q2_1/percent_fan_vote_estimates_weekly.csv')
df_rank = pd.read_csv('QF‘s solution/Q2_bayes/Q2_1/rank_fan_vote_estimates_weekly.csv')

# 统一列名映射 (确保 name 字段一致)
for df in [df_percent, df_rank]:
    if 'name' in df.columns:
        df.rename(columns={'name': 'Contestant'}, inplace=True)

# 合并数据集进行对比分析
common_cols = ['Contestant', 'season', 'week']
val_cols = ['est_fan_pct', 'ci_95_lower', 'ci_95_upper']

df_merged = pd.merge(
    df_percent[common_cols + val_cols],
    df_rank[common_cols + val_cols],
    on=common_cols,
    suffixes=('_Percent', '_Rank')
)

# 标记真实的赛季历史机制
def assign_mechanism(season):
    if 1 <= season <= 2 or season >= 28:
        return 'Actual Era: Rank-based'
    else:
        return 'Actual Era: Percentage-based'
    
df_merged['Mechanism_Label'] = df_merged['season'].apply(assign_mechanism)

# --- 3. Visualization ---

# Plot 1: Methodological Agreement (交叉验证散点图)
plt.figure(figsize=(7, 6))
color_palette = {'Actual Era: Rank-based': '#3498db', 'Actual Era: Percentage-based': '#f39c12'}

sns.scatterplot(
    data=df_merged, x='est_fan_pct_Percent', y='est_fan_pct_Rank', 
    hue='Mechanism_Label', palette=color_palette,
    alpha=0.6, edgecolor='w', linewidth=0.3, s=35, style='Mechanism_Label'
)

# 绘制对角线 (理想一致性线)
max_val = max(df_merged['est_fan_pct_Percent'].max(), df_merged['est_fan_pct_Rank'].max())
plt.plot([0, max_val], [0, max_val], color='#c0392b', linestyle='--', linewidth=1, label='Identity Line ($y=x$)')

plt.title('Figure 1: Robustness of Fan Vote Estimations Across Mechanisms', fontweight='bold')
plt.xlabel('Estimated Share (via Percentage Rule Mapping)', fontsize=10)
plt.ylabel('Estimated Share (via Rank Rule Mapping)', fontsize=10)
plt.legend(title='Historical Era', frameon=True, loc='upper left', fontsize='small')
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('QF‘s solution/Q2_bayes/Q2_1/Mechanism_Cross_Validation_Scatter.png')
plt.show()

# Plot 2: Micro-Level Discrepancy with Error Bars (微观置信区间对比)
# 选取一个典型的竞争激烈的周进行展示
target_season, target_week = 1, 3 
df_subset = df_merged[(df_merged['season'] == target_season) & (df_merged['week'] == target_week)].copy()

if not df_subset.empty:
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df_subset))
    width = 0.35
    
    # 绘制 Percentage 规则下的估计
    plt.bar(x - width/2, df_subset['est_fan_pct_Percent'], width, 
            yerr=[df_subset['est_fan_pct_Percent'] - df_subset['ci_95_lower_Percent'], 
                  df_subset['ci_95_upper_Percent'] - df_subset['est_fan_pct_Percent']],
            label='Mapping: Percentage Rule', color='#3498db', capsize=3, alpha=0.8,ecolor='black')
            
    # 绘制 Rank 规则下的估计
    plt.bar(x + width/2, df_subset['est_fan_pct_Rank'], width, 
            yerr=[df_subset['est_fan_pct_Rank'] - df_subset['ci_95_lower_Rank'], 
                  df_subset['ci_95_upper_Rank'] - df_subset['est_fan_pct_Rank']],
            label='Mapping: Rank Rule', color='#e67e22', capsize=3, alpha=0.8,ecolor='black')

    plt.xticks(x, df_subset['Contestant'], rotation=30, ha='right')
    plt.ylabel('Estimated Fan Vote Share')
    plt.title(f'Figure 2: Estimation Divergence & 95% Confidence Intervals (S{target_season} W{target_week})')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('QF‘s solution/Q2_bayes/Q2_1/Micro_Granularity_Comparison.png')
    plt.show()

# Plot 3: Statistical Bias Density (偏差密度分布)
plt.figure(figsize=(7, 4.5))
df_merged['Bias'] = df_merged['est_fan_pct_Percent'] - df_merged['est_fan_pct_Rank']
sns.kdeplot(data=df_merged, x='Bias', hue='Mechanism_Label', fill=True, palette=color_palette, alpha=0.3)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title('Figure 3: Distribution of Systematic Bias between Rules')
plt.xlabel('Estimation Difference ($\Delta \hat{P}_{i,t}$)')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('QF‘s solution/Q2_bayes/Q2_1/Systematic_Bias_Distribution.png')
plt.show()

# Plot 4: Distribution Density (KDE)
plt.figure(figsize=(8, 5))
sns.kdeplot(df_merged['est_fan_pct_Percent'], fill=True, color='#3498db', label='Percentage Rule', alpha=0.3)
sns.kdeplot(df_merged['est_fan_pct_Rank'], fill=True, color='#e67e22', label='Rank Rule', alpha=0.3)
plt.title('Figure 3: Probability Density Function of Estimated Shares')
plt.xlabel('Fan Vote Share')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('QF‘s solution/Q2_bayes/Q2_1/Comparison_Distribution.png')
plt.show()