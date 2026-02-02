import pandas as pd
import numpy as np

class DancingSimulator:
    def __init__(self, data_path, estimates_path):
        """
        初始化：加载并预处理数据
        """
        # 1. 加载裁判分数
        self.df = pd.read_csv(data_path)
        # 清洗分数列
        score_cols = [c for c in self.df.columns if 'judge' in c and 'score' in c]
        for col in score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        self.df['celebrity_name'] = self.df['celebrity_name'].str.strip()
        
        # 2. 加载粉丝估算数据
        self.fan_df = pd.read_csv(estimates_path)
        self.fan_df['name'] = self.fan_df['name'].str.strip()

    def get_real_data_or_impute(self, season, week, name, history):
        """
        获取选手本周数据。
        如果现实中有数据 -> 返回真实数据。
        如果现实中无数据（已淘汰） -> 基于 history 计算平均值返回（复活逻辑）。
        """
        # A. 尝试获取真实裁判分
        season_data = self.df[self.df['season'] == season]
        contestant_row = season_data[season_data['celebrity_name'] == name]
        
        j_score = None
        f_pct = None
        status = 'Real'
        
        if not contestant_row.empty:
            # 获取当周裁判分列
            judge_cols = [f'week{week}_judge{i}_score' for i in range(1, 5)]
            # 检查是否有分 (sum > 0)
            score_sum = contestant_row[judge_cols].sum(axis=1).values[0]
            if score_sum > 0:
                j_score = score_sum
                
                # B. 如果有裁判分，尝试获取真实粉丝分
                fan_row = self.fan_df[
                    (self.fan_df['season'] == season) & 
                    (self.fan_df['week'] == week) & 
                    (self.fan_df['name'] == name)
                ]
                if not fan_row.empty:
                    f_pct = fan_row['est_fan_pct'].values[0]
                else:
                    f_pct = 0 # 极其罕见
            
        # C. 如果没有真实数据（现实已淘汰），进行插补 (Ghost Mode)
        if j_score is None:
            status = 'Ghost' # 标记为幽灵复活
            if len(history['judge']) > 0:
                j_score = np.mean(history['judge']) # 使用历史裁判平均分
                f_pct = np.mean(history['fan'])     # 使用历史粉丝平均分
            else:
                # 理论上不应发生（Week 1 肯定有分）
                j_score = 0
                f_pct = 0
                
        return j_score, f_pct, status

    def apply_rules(self, df_weekly, method='rank', judges_save=False):
        """
        规则引擎：计算排名并返回被淘汰者
        """
        df = df_weekly.copy()
        
        # --- 规则计算 ---
        if method == 'rank':
            # Rank制：数值越小越好。排名相加。
            df['judge_rank'] = df['weekly_judge_total'].rank(ascending=False, method='min')
            df['fan_rank'] = df['est_fan_pct'].rank(ascending=False, method='min')
            df['combined_score'] = df['judge_rank'] + df['fan_rank']
            # 排序：最差的在上面 (Combined Score 最大)
            df_sorted = df.sort_values(by=['combined_score', 'weekly_judge_total'], ascending=[False, True])

        elif method == 'percentage':
            # Percentage制：数值越大越好。比例相加。
            total_j = df['weekly_judge_total'].sum()
            df['judge_pct'] = df['weekly_judge_total'] / (total_j if total_j > 0 else 1)
            
            # 注意：粉丝百分比需要针对【当前的幸存者池】重新归一化
            total_f = df['est_fan_pct'].sum()
            df['fan_pct_norm'] = df['est_fan_pct'] / (total_f if total_f > 0 else 1)
            
            df['combined_score'] = 0.5 * df['judge_pct'] + 0.5 * df['fan_pct_norm']
            # 排序：最差的在上面 (Combined Score 最小)
            df_sorted = df.sort_values(by=['combined_score', 'weekly_judge_total'], ascending=[True, True])
            
        # --- 淘汰判定 ---
        if len(df_sorted) < 2:
            return df_sorted.iloc[0]['celebrity_name'], df_sorted
            
        bottom_2 = df_sorted.iloc[:2].copy()
        eliminated = None
        
        if judges_save:
            # 裁判拯救：Bottom 2 中裁判分低的走
            p1 = bottom_2.iloc[0]
            p2 = bottom_2.iloc[1]
            if p1['weekly_judge_total'] < p2['weekly_judge_total']:
                eliminated = p1['celebrity_name']
            elif p1['weekly_judge_total'] > p2['weekly_judge_total']:
                eliminated = p2['celebrity_name']
            else:
                eliminated = p1['celebrity_name'] # 平局按原规则
        else:
            eliminated = df_sorted.iloc[0]['celebrity_name']
            
        return eliminated, df_sorted

    def simulate_season(self, target_name, season, method, judges_save):
        print(f"\n{'#'*60}")
        print(f"FULL DYNAMIC SIMULATION: {target_name} | Season {season}")
        print(f"Mode: {method.upper()} | Judges' Save: {judges_save}")
        print(f"{'#'*60}")
        
        # 1. 初始化：获取该赛季 Week 1 所有参赛者
        week1_df = self.df[self.df['season'] == season]
        # 找出Week 1有分的人
        cols = [f'week1_judge{i}_score' for i in range(1, 5)]
        week1_df['total'] = week1_df[cols].sum(axis=1)
        starting_roster = week1_df[week1_df['total'] > 0]['celebrity_name'].unique().tolist()
        
        current_survivors = starting_roster.copy()
        # 历史记录字典：用于计算平均分
        contestant_history = {name: {'judge': [], 'fan': []} for name in starting_roster}
        
        # 2. 逐周模拟 (Max 15 weeks)
        for week in range(1, 16):
            if len(current_survivors) <= 1:
                print(f"🏆 Winner declared: {current_survivors[0]}")
                break
                
            # --- 构建本周参赛数据 (Roster Construction) ---
            weekly_data = []
            
            for name in current_survivors:
                # 获取数据（可能是真实的，也可能是 Imputed Ghost）
                j_score, f_pct, status = self.get_real_data_or_impute(
                    season, week, name, contestant_history[name]
                )
                
                # 如果是真实数据，更新历史记录（用于未来的平均值计算）
                # 注意：如果是Ghost数据，我们不将其加入历史，以免平均值发生人工偏移
                if status == 'Real':
                    contestant_history[name]['judge'].append(j_score)
                    contestant_history[name]['fan'].append(f_pct)
                    
                weekly_data.append({
                    'celebrity_name': name,
                    'weekly_judge_total': j_score,
                    'est_fan_pct': f_pct,
                    'status': status
                })
            
            df_weekly = pd.DataFrame(weekly_data)
            
            # --- 执行淘汰 ---
            elim_who, standings = self.apply_rules(df_weekly, method, judges_save)
            
            # --- 打印关键信息 ---
            target_info = ""
            if target_name in df_weekly['celebrity_name'].values:
                t_row = standings[standings['celebrity_name'] == target_name].iloc[0]
                rank_score = t_row['combined_score']
                target_info = f"| {target_name} ({t_row['status']}): Score={rank_score:.2f}"
            
            print(f"Week {week}: Eliminated -> {elim_who} {target_info}")
            
            # --- 更新幸存者池 ---
            if elim_who in current_survivors:
                current_survivors.remove(elim_who)
            
            # --- 判定目标人物命运 ---
            if elim_who == target_name:
                print(f">>> 🚨 {target_name} ELIMINATED in Week {week} under new rules! 🚨 <<<")
                return week
        
        if target_name in current_survivors:
            print(f"RESULT: {target_name} WON or reached Finals!")
            return "Finalist"
        else:
            return "Eliminated"

# ================= 使用示例 =================
sim = DancingSimulator('2026_MCM_Problem_C_Data.csv', 'QF‘s solution/Bayes_Elo/real_figures/real_fan_vote_estimates_weekly.csv')

