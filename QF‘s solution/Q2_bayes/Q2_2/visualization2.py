import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df_sim = pd.read_csv('QF‘s solution/Q2_bayes/Q2_2/simulation_results_4_rules.csv')
df_cont = pd.read_csv('QF‘s solution/Q2_bayes/Q2_2/controversal_name_list.csv')

# 数据预处理
# 重命名 'controversal_name_list.csv' 中的列名以匹配 'simulation_results_4_rules.csv'
df_cont.rename(columns={'celebrity_name': 'Name', 'season': 'Season'}, inplace=True)

# 合并两个数据集
df_merged = pd.merge(df_sim, df_cont[['Name', 'Season', 'Normalized_Weighted_Index']], on=['Name', 'Season'], how='inner')

# 获取Normalized_Weighted_Index最高的50个人
top50 = df_merged.nlargest(50, 'Normalized_Weighted_Index')

# 定义四种方法的列名
methods = ['Sim_Rank_NoSave', 'Sim_Rank_Save', 'Sim_Pct_NoSave', 'Sim_Pct_Save']

# 创建一个图表
plt.figure(figsize=(15, 10))

for i, method in enumerate(methods, 1):
    # 计算误差
    top50[f'{method}_Error'] = top50[method] - top50['Real_Survival_Week']
    
    # 绘制散点图
    plt.subplot(2, 2, i)
    sns.scatterplot(x='Normalized_Weighted_Index', y=f'{method}_Error', data=top50)
    plt.title(f'Distribution of Errors for {method}')
    plt.xlabel('Normalized Weighted Index')
    plt.ylabel('Error (Simulated - Real Weeks)')
    plt.axhline(0, color='red', linestyle='--')  # 添加水平线表示无误差

plt.tight_layout()
plt.show()