import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
def calculate_method_bias(sim_file, raw_file, estimates_file):
    """
    计算不同投票机制下的决策一致性偏差 (Bias)
    """
    # 1. 加载仿真结果与基础数据
    sim_df = pd.read_csv(sim_file)
    raw_df = pd.read_csv(raw_file)
    fan_estimates = pd.read_csv(estimates_file)
    
    # 清洗姓名空格，确保匹配一致
    sim_df['Name'] = sim_df['Name'].str.strip()
    fan_estimates['name'] = fan_estimates['name'].str.strip()
    raw_df['celebrity_name'] = raw_df['celebrity_name'].str.strip()

    # 2. 获取每个赛季的总人数 (用于 Rank = Total - Week 公式)
    season_counts = raw_df.groupby('season')['celebrity_name'].nunique().reset_index()
    season_counts.columns = ['season', 'total_contestants']

    # 3. 计算每个选手的平均裁判排名与平均粉丝排名
    # 在每周内部进行排名 (1为最高)
    fan_estimates['j_rank'] = fan_estimates.groupby(['season', 'week'])['judge_pct'].rank(ascending=False)
    fan_estimates['f_rank'] = fan_estimates.groupby(['season', 'week'])['est_fan_pct'].rank(ascending=False)

    avg_ranks = fan_estimates.groupby(['season', 'name']).agg(
        avg_judge_rank=('j_rank', 'mean'),
        avg_fan_rank=('f_rank', 'mean')
    ).reset_index()

    # 4. 合并数据
    df = pd.merge(sim_df, season_counts, left_on='Season', right_on='season', how='inner')
    df = pd.merge(df, avg_ranks, left_on=['Name', 'Season'], right_on=['name', 'season'], how='inner')

    # 定义需要分析的模拟列（对应不同的淘汰周）
    methods = ['Sim_Rank_NoSave', 'Sim_Pct_NoSave', 'Sim_Rank_Save']
    
    bias_results = []

    # 5. 核心计算循环
    for s in df['Season'].unique():
        season_data = df[df['Season'] == s]
        if len(season_data) < 2:
            continue
            
        for method in methods:
            # 应用公式：Rank = Total - Week
            # 淘汰周越晚（数字越大），排名越靠前（数字越小）
            simulated_rank = season_data['total_contestants'] - season_data[method]
            
            # 计算裁判一致性 (Rho_Judge)
            rho_j, _ = spearmanr(season_data['avg_judge_rank'], simulated_rank)
            
            # 计算粉丝一致性 (Rho_Fan)
            rho_f, _ = spearmanr(season_data['avg_fan_rank'], simulated_rank)
            
            # 计算偏差
            # Bias = Rho_Judge - Rho_Fan
            bias_results.append({
                'Season': s,
                'Method': method,
                'Rho_Judge': rho_j,
                'Rho_Fan': rho_f,
                'Bias': rho_j - rho_f,
                'Abs_Bias': abs(rho_j - rho_f)
            })

    # 6. 汇总与输出
    bias_df = pd.DataFrame(bias_results)
    summary = bias_df.groupby('Method').agg(
        Mean_Abs_Bias=('Abs_Bias', 'mean'),
        Mean_Rho_Judge=('Rho_Judge', 'mean'),
        Mean_Rho_Fan=('Rho_Fan', 'mean')
    ).reset_index()

    # 保存结果
    bias_df.to_csv('season_method_bias_comparison.csv', index=False)
    summary.to_csv('final_method_bias_summary.csv', index=False)
    
    return summary

# ================= 运行脚本 =================
if __name__ == "__main__":
    summary_report = calculate_method_bias(
        'QF‘s solution/Q2_bayes/Q2_2/simulation_results_4_rules.csv', 
        '2026_MCM_Problem_C_Data.csv', 
        'QF‘s solution/Bayes_Elo/real_figures/fan_vote_estimates_weekly.csv'
    )
    print("各投票机制偏差汇总：")
    print(summary_report)

    # 设置风格
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(9, 6))

    # 绘图
    ax = sns.barplot(
        data=summary_report,
        x='Method',
        y='Mean_Abs_Bias',
        palette=['#2C7BB6']
    )

    # 添加数值标签
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    # 标题加粗
    plt.title('Comparison of Mean Absolute Bias Among Methods', fontsize=14, fontweight='bold', pad=20)

    plt.xlabel('Voting Mechanism', fontsize=12)
    plt.ylabel('Mean Absolute Bias', fontsize=12)
    plt.ylim(0, max(summary_report['Mean_Abs_Bias']) * 1.2) # 自动调整Y轴上限
    plt.xticks(rotation=15)
    plt.tight_layout()

    # 保存
    plt.savefig("QF‘s solution/Q2_bayes/Q2_3/bias_bar.png", dpi=300, bbox_inches='tight')