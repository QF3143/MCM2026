import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Step 1: Load data
# -----------------------------
df_sim = pd.read_csv('QF‘s solution/Q2_bayes/Q2_2/simulation_results_4_rules.csv')
df_cont = pd.read_csv('QF‘s solution/Q2_bayes/Q2_2/controversal_name_list.csv')

# -----------------------------
# Step 2: Align column names for merging
# -----------------------------
# simulation file: Name, Season
# controversal file: celebrity_name, season (lowercase)
df_cont.rename(columns={'celebrity_name': 'Name', 'season': 'Season'}, inplace=True)

# Now both have: Name, Season

# -----------------------------
# Step 3: Merge on Name and Season
# -----------------------------
df = pd.merge(
    df_sim,
    df_cont[['Name', 'Season', 'Normalized_Weighted_Index']],
    on=['Name', 'Season'],
    how='inner'
)

# -----------------------------
# Step 4: Select top 50 by Normalized_Weighted_Index
# -----------------------------
top50 = df.nlargest(50, 'Normalized_Weighted_Index')

# -----------------------------
# Step 5: Compute errors for four methods
# -----------------------------
methods = {
    'Sim_Rank_NoSave': 'Rank (No Save)',
    'Sim_Rank_Save': 'Rank (With Save)',
    'Sim_Pct_NoSave': 'Percentile (No Save)',
    'Sim_Pct_Save': 'Percentile (With Save)'
}

error_dict = {}
for col, label in methods.items():
    error = top50[col] - top50['Real_Survival_Week']
    error_dict[label] = error

# -----------------------------
# Step 6: Plot 2x2 violin plots
# -----------------------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (method_name, errors) in enumerate(error_dict.items()):
    ax = axes[idx]
    sns.violinplot(y=errors, ax=ax, inner='box', color='lightblue', width=0.8)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
    
    median_err = np.median(errors)
    mae = np.mean(np.abs(errors))
    ax.text(0.05, 0.95,
            f'Median: {median_err:.2f}\nMAE: {mae:.2f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            fontsize=10)
    
    ax.set_title(method_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (Simulated − Real Weeks)', fontsize=10)
    ax.set_xlabel('')

plt.suptitle('Simulation Error Distribution for Top 50 Most Controversial Celebrities\n'
             '(Error = Simulated Survival Week | Actual Survival Week)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()