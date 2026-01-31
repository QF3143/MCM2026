import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 配置参数
# ==========================================
SIMULATION_ROUNDS = 10000 # 增加模拟次数，因为全序约束更难满足，需要更多样本
ZIPF_PARAMETER = 1.0
RANDOM_SEED = 42

# ==========================================
# 2. 辅助函数
# ==========================================
def generate_zipf_shares(n, s=1.0):
    if n == 0: return []
    if s == 0: return np.ones(n) / n
    ranks = np.arange(1, n + 1)
    weights = 1 / np.power(ranks, s)
    return weights / np.sum(weights)

def get_ranks(scores, ascending=False):
    # ascending=False: 分数高(10) -> 排名小(1)
    if ascending:
        return np.argsort(np.argsort(scores)) + 1
    else:
        return np.argsort(np.argsort(-np.array(scores))) + 1

# ==========================================
# 3. 数据加载与处理
# ==========================================
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 1. 确定每季决赛周
    season_final_week = {}
    for season in df['season'].unique():
        s_df = df[df['season'] == season]
        max_w = 0
        for w in range(1, 20):
            cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            vals = []
            for c in cols:
                if c in s_df.columns:
                    vals.append(pd.to_numeric(s_df[c], errors='coerce').sum())
            if sum(vals) > 0:
                max_w = w
        season_final_week[season] = max_w

    long_data = []
    
    for idx, row in df.iterrows():
        season = row['season']
        name = row['celebrity_name']
        final_w = season_final_week.get(season, 10)
        
        # 解析名次 (Placement)
        try:
            placement = float(row['placement'])
        except:
            placement = 999
            
        # 提取分数
        scores_map = {}
        for w in range(1, final_w + 1):
            s_val = 0
            s_list = []
            for i in range(1, 5):
                col = f'week{w}_judge{i}_score'
                if col in row.index:
                    v = pd.to_numeric(row[col], errors='coerce')
                    if not pd.isna(v) and v > 0:
                        s_list.append(v)
            if s_list:
                s_val = sum(s_list)
            scores_map[w] = s_val
            
        # 判定淘汰周
        elim_week = -1
        # 常规周淘汰判定
        for w in range(1, final_w):
            if scores_map.get(w, 0) > 0 and scores_map.get(w+1, 0) == 0:
                elim_week = w
                break
                
        # 记录数据
        for w in range(1, final_w + 1):
            s_val = scores_map.get(w, 0)
            if s_val > 0:
                long_data.append({
                    'Season': season,
                    'Week': w,
                    'Contestant': name,
                    'JudgeScore': s_val,
                    'Placement': placement,
                    'IsEliminated': (w == elim_week), # 常规周的淘汰标记
                    'IsFinalWeek': (w == final_w)     # 决赛周标记
                })
                
    return pd.DataFrame(long_data), season_final_week

# ==========================================
# 4. 模拟核心 (引入全序约束)
# ==========================================
def simulate_week(week_df, rule_type, rounds, zipf_s):
    contestants = week_df['Contestant'].values
    judge_scores = week_df['JudgeScore'].values
    placements = week_df['Placement'].values
    is_eliminated_mask = week_df['IsEliminated'].values
    
    is_final_week = week_df['IsFinalWeek'].iloc[0]
    n = len(contestants)
    
    # 基础校验
    if n == 0: return None
    # 如果不是决赛周，且没发生淘汰，跳过
    if not is_final_week and not np.any(is_eliminated_mask):
        return None

    # 计算裁判指标
    if rule_type == 'percent':
        total = np.sum(judge_scores)
        judge_metrics = judge_scores / total
    else:
        judge_metrics = get_ranks(judge_scores, ascending=False)
        
    base_shares = generate_zipf_shares(n, s=zipf_s)
    valid_shares = []
    
    for _ in range(rounds):
        fan_shares = np.random.permutation(base_shares)
        
        # === 核心逻辑分支 ===
        
        if is_final_week:
            # --- 决赛周逻辑：全序约束 (Strict Ordering) ---
            
            # 计算综合结果
            if rule_type == 'percent':
                # 积分制：分数越高越好
                final_scores = judge_metrics + fan_shares
                # 我们的模拟排名：分数降序排列 (索引0是第一名)
                # argsort(argsort(-scores)) 得到 0-based rank (0=1st, 1=2nd...)
                sim_ranks_0based = np.argsort(np.argsort(-final_scores))
                sim_placement = sim_ranks_0based + 1
                
            else: # rank
                # 排名制：排名数值越小越好
                fan_ranks = get_ranks(fan_shares, ascending=False)
                total_ranks = judge_metrics + fan_ranks
                # 排名数值升序排列 (数值小是第一名)
                sim_ranks_0based = np.argsort(np.argsort(total_ranks))
                sim_placement = sim_ranks_0based + 1
            
            # 校验：模拟名次是否与真实名次完全一致？
            # 比较 sim_placement 和 placements
            # 注意：数据中 placements 可能不连续或有缺失（如有些只是Finalist），需只比对有明确名次的人
            # 简单起见，我们假设 data 里的 placement 是准确的 1, 2, 3...
            
            match = True
            for i in range(n):
                # 真实名次
                real_p = placements[i]
                if real_p < 99: # 忽略 999 或无效名次
                    if sim_placement[i] != real_p:
                        match = False
                        break
            
            if match:
                valid_shares.append(fan_shares)
                
        else:
            # --- 常规周逻辑：末位淘汰约束 (Lowest Score/Rank) ---
            
            if rule_type == 'percent':
                total_metric = judge_metrics + fan_shares
                # 找最低分
                min_val = np.min(total_metric)
                sim_elim_idx = np.where(total_metric == min_val)[0]
            else:
                fan_ranks = get_ranks(fan_shares, ascending=False)
                total_metric = judge_metrics + fan_ranks
                # 找最大排名值
                max_val = np.max(total_metric)
                sim_elim_idx = np.where(total_metric == max_val)[0]
            
            sim_elim_names = contestants[sim_elim_idx]
            actual_elim_names = contestants[is_eliminated_mask]
            
            if not set(actual_elim_names).isdisjoint(sim_elim_names):
                valid_shares.append(fan_shares)

    # 结果聚合
    if not valid_shares:
        return None
    
    #95%置信区间提取开始
    valid_shares_arr = np.array(valid_shares)
    # 计算 2.5% 和 97.5% 分位数作为 95% 置信区间
    ci_low = np.percentile(valid_shares_arr, 2.5, axis=0)
    ci_high = np.percentile(valid_shares_arr, 97.5, axis=0)
    
    return pd.DataFrame({
        'Contestant': contestants,
        'Est_Fan_Share': np.mean(valid_shares_arr, axis=0),
        'Fan_Share_CI_Lower': ci_low,
        'Fan_Share_CI_Upper': ci_high,
        'Fan_Share_Std': np.std(valid_shares_arr, axis=0),
        'Valid_Simulations': len(valid_shares_arr),
        'IsFinalWeek': is_final_week
    })

# ==========================================
# 5. 执行器
# ==========================================
def run_solver(filepath,rule="percent"):
    df_long, _ = load_data(filepath)
    results = []
    
    print("Running advanced simulations...")
    for (season, week), g in tqdm(df_long.groupby(['Season', 'Week'])):
        # if season <= 2 or season >= 28:
        #     rule = 'rank'
        # else:
        #     rule = 'percent'
            
        res = simulate_week(g, rule, SIMULATION_ROUNDS, ZIPF_PARAMETER)
        if res is not None:
            res['Season'] = season
            res['Week'] = week
            res['Rule_Used'] = rule
            results.append(res)
            
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

if __name__ == "__main__":
    out = run_solver('2026_MCM_Problem_C_Data.csv','rank')
    if not out.empty:
        out.to_csv('QF‘s solution/Q2_QFv2/rank/rank_pridict.csv', index=False)
        print("\nOptimization Complete.")