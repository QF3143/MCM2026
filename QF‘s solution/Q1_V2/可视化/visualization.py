import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_normalized_frontier():
    # 1. 加载数据
    # 你的模型估算结果
    results_df = pd.read_csv('QF‘s solution/Q1_V2/Problem_C_Solution_Advanced.csv')
    # 原始比赛数据 (用于获取裁判分和淘汰状态)
    raw_df = pd.read_csv('2026_MCM_Problem_C_Data.csv')
    
    # 清洗列名
    raw_df.columns = [c.lower().strip() for c in raw_df.columns]
    
    # 2. 构建辅助字典：快速查找裁判分和淘汰状态
    # 结构: (Season, Contestant) -> {Week: {'score': X, 'is_elim': T/F}}
    contestant_map = {}
    
    # 预计算每季的决赛周（避免误判）
    season_max_weeks = {}
    for s in raw_df['season'].unique():
        cols = [c for c in raw_df.columns if f'week' in c and 'judge1' in c]
        # 简单估算最大周数
        max_w = 0
        s_data = raw_df[raw_df['season'] == s]
        for col in cols:
            w_num = int(col.split('week')[1].split('_')[0])
            if s_data[col].sum() > 0:
                max_w = max(max_w, w_num)
        season_max_weeks[s] = max_w

    for idx, row in raw_df.iterrows():
        season = row['season']
        name = row['celebrity_name']
        res_text = str(row['results'])
        
        # 解析文本中的淘汰周
        text_elim_week = -1
        if 'Eliminated Week' in res_text:
            try:
                text_elim_week = int(res_text.split('Week')[1].strip())
            except: pass
            
        scores = {}
        # 遍历该选手所有周
        max_w = season_max_weeks.get(season, 10)
        
        for w in range(1, max_w + 1):
            # 获取当周裁判分
            j_score = 0
            for i in range(1, 5):
                c_name = f'week{w}_judge{i}_score'
                if c_name in raw_df.columns:
                    val = pd.to_numeric(row[c_name], errors='coerce')
                    if not pd.isna(val) and val > 0:
                        j_score += val
            
            # 简单判定淘汰：如果是文本中记录的淘汰周，则为 True
            # 注意：这里我们主要为了可视化“被淘汰者”的特征，用文本标签最稳妥
            is_elim = (w == text_elim_week)
            
            scores[w] = {'score': j_score, 'is_elim': is_elim}
            
        contestant_map[(season, name)] = scores

    # 3. 将原始信息合并到 Results DF 中
    def get_info(row):
        key = (row['Season'], row['Contestant'])
        week = row['Week']
        if key in contestant_map and week in contestant_map[key]:
            info = contestant_map[key][week]
            return pd.Series([info['score'], info['is_elim']])
        return pd.Series([0, False])

    results_df[['Judge_Score', 'Is_Eliminated']] = results_df.apply(get_info, axis=1)
    
    # 剔除裁判分为0的异常行 (可能退赛或数据缺失)
    results_df = results_df[results_df['Judge_Score'] > 0]

    # 4. 计算归一化指标 (Normalization)
    # 按 (Season, Week) 分组，获取当周总分和总人数
    groups = results_df.groupby(['Season', 'Week'])
    
    results_df['Week_N'] = groups['Contestant'].transform('count')
    results_df['Week_Judge_Total'] = groups['Judge_Score'].transform('sum')
    
    # 计算份额
    results_df['Judge_Share'] = results_df['Judge_Score'] / results_df['Week_Judge_Total']
    
    # 核心步骤：归一化 = Share * N
    results_df['Norm_Judge'] = results_df['Judge_Share'] * results_df['Week_N']
    results_df['Norm_Fan'] = results_df['Est_Fan_Share'] * results_df['Week_N']

    # 5. 绘图
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # 分组数据
    safe_data = results_df[results_df['Is_Eliminated'] == False]
    elim_data = results_df[results_df['Is_Eliminated'] == True]
    
    # 绘制散点
    # 安全者：蓝色圆点，透明度高一点以免遮挡
    plt.scatter(safe_data['Norm_Judge'], safe_data['Norm_Fan'], 
                c='#3498db', alpha=0.3, s=30, label='Safe (Survivors)')
    
    # 淘汰者：红色叉号，不透明，强调显示
    plt.scatter(elim_data['Norm_Judge'], elim_data['Norm_Fan'], 
                c='#e74c3c', alpha=0.9, s=50, marker='x', label='Eliminated')

    # 添加辅助参考线 (平均线)
    plt.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    # 标注区域
    plt.text(0.4, 0.3, 'DEATH ZONE\n(Low Judge + Low Fan)', 
             color='#c0392b', fontsize=10, ha='center', va='center', fontweight='bold', alpha=0.5)
    
    plt.text(0.4, 2.0, 'FAN FAVORITE ZONE\n(Low Judge + High Fan)', 
             color='#2980b9', fontsize=10, ha='center', va='center', fontweight='bold', alpha=0.5)

    # 装饰
    plt.title('Normalized Survival Frontier: Judge vs. Fan Support', fontsize=16, fontweight='bold')
    plt.xlabel('Normalized Judge Score (1.0 = Average)', fontsize=12)
    plt.ylabel('Normalized Fan Vote (1.0 = Average)', fontsize=12)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    
    # 限制坐标轴范围以防极值破坏视觉效果
    plt.xlim(0, 2.0)
    plt.ylim(0, 3.0)
    
    plt.tight_layout()
    plt.savefig('Normalized_Survival_Frontier.png', dpi=300)
    plt.show()
    
    print("图表已生成：Normalized_Survival_Frontier.png")

def plot_contestant_trajectory(season, name, path):
    df = pd.read_csv(path)
    data = df[(df['Season'] == season) & (df['Contestant'] == name)].sort_values('Week')
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['Week'], data['Est_Fan_Share'], 'o-', color='#e74c3c', label='Mean Estimate')
    
    # 填充 95% 置信区间
    plt.fill_between(data['Week'], 
                     data['Fan_Share_CI_Lower'], 
                     data['Fan_Share_CI_Upper'], 
                     color='#e74c3c', alpha=0.2, label='95% Confidence Interval')
    
    plt.xlabel('Week Number')
    plt.ylabel('Estimated Fan Share ($P_{fan}$)')
    plt.title(f'Fan Share Trajectory for {name} (Season {season})')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'Trajectory_{name.replace(" ", "_")}.png')
    
# 运行绘图函数
if __name__ == "__main__":
    plot_contestant_trajectory(19, "Alfonso Ribeiro", "QF‘s solution/Q1_V2/Problem_C_Solution_Advanced.csv")