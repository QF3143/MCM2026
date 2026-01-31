import pandas as pd
import numpy as np
import re

class DancingSimulator:
    def __init__(self, data_path, estimates_path, list_path):
        """
        初始化模拟器
        """
        # 1. 加载数据
        self.df = pd.read_csv(data_path)
        self.fan_df = pd.read_csv(estimates_path)
        self.target_list = pd.read_csv(list_path)
        
        # 2. 数据清洗
        # 确保分数列为数值
        score_cols = [c for c in self.df.columns if 'judge' in c and 'score' in c]
        for col in score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
            
        # 去除名字空格
        self.df['celebrity_name'] = self.df['celebrity_name'].str.strip()
        self.fan_df['name'] = self.fan_df['name'].str.strip()
        self.target_list['celebrity_name'] = self.target_list['celebrity_name'].str.strip()

    def get_max_week(self, season):
        """获取某赛季的最大周数（用于处理决赛选手）"""
        season_df = self.df[self.df['season'] == season]
        max_w = 0
        for w in range(1, 20): # 假设最多20周
            col = f'week{w}_judge1_score'
            if col in self.df.columns:
                if season_df[col].sum() > 0:
                    max_w = w
        return max_w

    def get_real_elimination_week(self, name, season):
        """从原始数据中解析真实淘汰周数"""
        row = self.df[(self.df['celebrity_name'] == name) & (self.df['season'] == season)]
        if row.empty:
            return 0
        
        result_str = str(row.iloc[0]['results'])
        # 解析 "Eliminated Week X"
        match = re.search(r'Eliminated Week (\d+)', result_str)
        if match:
            return int(match.group(1))
        
        # 如果没有 "Eliminated Week X"，通常是冠军或亚军，返回该赛季最大周数
        return self.get_max_week(season)

    def get_real_data_or_impute(self, season, week, name, history):
        """
        获取选手本周表现：
        1. 真实存活 -> 返回真实裁判分和粉丝估算分
        2. 真实淘汰 -> 返回历史平均分 (Ghost Mode)
        """
        season_data = self.df[self.df['season'] == season]
        contestant_row = season_data[season_data['celebrity_name'] == name]
        
        j_score = None
        f_pct = None
        status = 'Real'
        
        # 尝试获取真实数据
        if not contestant_row.empty:
            judge_cols = [f'week{week}_judge{i}_score' for i in range(1, 5)]
            valid_cols = [c for c in judge_cols if c in self.df.columns]
            
            if valid_cols:
                score_sum = contestant_row[valid_cols].sum(axis=1).values[0]
                if score_sum > 0:
                    j_score = score_sum
                    # 获取对应的粉丝估算
                    fan_row = self.fan_df[
                        (self.fan_df['season'] == season) & 
                        (self.fan_df['week'] == week) & 
                        (self.fan_df['name'] == name)
                    ]
                    if not fan_row.empty:
                        f_pct = fan_row['est_fan_pct'].values[0]
                    else:
                        f_pct = 0
            
        # 如果没有真实数据（Ghost Mode），使用历史平均
        if j_score is None:
            status = 'Ghost'
            if len(history['judge']) > 0:
                j_score = np.mean(history['judge'])
                f_pct = np.mean(history['fan'])
            else:
                j_score = 0
                f_pct = 0
                
        return j_score, f_pct, status

    def apply_rules(self, df_weekly, method='rank', judges_save=False):
        """应用规则判定淘汰者"""
        df = df_weekly.copy()
        
        # === A. 排名制 (Rank) ===
        if method == 'rank':
            # 越小越好
            df['judge_rank'] = df['weekly_judge_total'].rank(ascending=False, method='min')
            df['fan_rank'] = df['est_fan_pct'].rank(ascending=False, method='min')
            df['combined_score'] = df['judge_rank'] + df['fan_rank']
            # 排序：Combined Score 降序 (最差的在顶端)
            df_sorted = df.sort_values(by=['combined_score', 'weekly_judge_total'], ascending=[False, True])

        # === B. 百分比制 (Percentage) ===
        elif method == 'percentage':
            # 越大越好
            total_j = df['weekly_judge_total'].sum()
            df['judge_pct'] = df['weekly_judge_total'] / (total_j if total_j > 0 else 1)
            
            total_f = df['est_fan_pct'].sum()
            df['fan_pct_norm'] = df['est_fan_pct'] / (total_f if total_f > 0 else 1)
            
            df['combined_score'] = 0.5 * df['judge_pct'] + 0.5 * df['fan_pct_norm']
            # 排序：Combined Score 升序 (最差的在顶端)
            df_sorted = df.sort_values(by=['combined_score', 'weekly_judge_total'], ascending=[True, True])
            
        # === 淘汰判定 ===
        if len(df_sorted) < 2:
            return df_sorted.iloc[0]['celebrity_name']
            
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
                eliminated = p1['celebrity_name']
        else:
            # 无拯救：最差的走
            eliminated = df_sorted.iloc[0]['celebrity_name']
            
        return eliminated

    def simulate_target(self, target_name, season, method, judges_save):
        """单次模拟核心逻辑"""
        # 1. 确定 Week 1 参赛名单
        week1_df = self.df[self.df['season'] == season]
        cols = [f'week1_judge{i}_score' for i in range(1, 5)]
        valid_cols = [c for c in cols if c in self.df.columns]
        
        if not valid_cols: return 0 
        
        week1_df['total'] = week1_df[valid_cols].sum(axis=1)
        starting_roster = week1_df[week1_df['total'] > 0]['celebrity_name'].unique().tolist()
        
        if target_name not in starting_roster:
            return 0 # 没参赛
            
        current_survivors = starting_roster.copy()
        contestant_history = {name: {'judge': [], 'fan': []} for name in starting_roster}
        
        max_week = self.get_max_week(season)
        
        # 2. 逐周模拟
        for week in range(1, max_week + 1):
            if len(current_survivors) <= 1:
                break 
                
            # 构建本周数据
            weekly_data = []
            for name in current_survivors:
                j, f, s = self.get_real_data_or_impute(season, week, name, contestant_history[name])
                
                # 仅记录真实数据进历史库
                if s == 'Real':
                    contestant_history[name]['judge'].append(j)
                    contestant_history[name]['fan'].append(f)
                    
                weekly_data.append({'celebrity_name': name, 'weekly_judge_total': j, 'est_fan_pct': f})
            
            df_weekly = pd.DataFrame(weekly_data)
            
            # 淘汰判定
            elim_who = self.apply_rules(df_weekly, method, judges_save)
            
            # 检查目标人物
            if elim_who == target_name:
                return week # 止步于此
            
            # 移除被淘汰者
            if elim_who in current_survivors:
                current_survivors.remove(elim_who)
                
        return max_week # 活到最后

    def run_batch_simulation(self):
        """批量运行并保存"""
        results = []
        print("Starting Batch Simulation...")
        
        for idx, row in self.target_list.iterrows():
            name = row['celebrity_name']
            season = int(row['season'])
            print(f"Processing: {name} (S{season})")
            
            # 1. 获取真实淘汰周
            real_week = self.get_real_elimination_week(name, season)
            
            # 2. 确定该赛季的真实规则 (Real Rule Config)
            # S1-2: Rank (NoSave)
            # S3-27: Pct (NoSave)
            # S28+: Rank (Save)
            real_rule_config = None
            if season <= 2:
                real_rule_config = ('rank', False)
            elif season <= 27:
                real_rule_config = ('percentage', False)
            else:
                real_rule_config = ('rank', True) 
            
            # 3. 四种模拟配置
            configs = [
                ('rank', False, 'Sim_Rank_NoSave'),
                ('percentage', False, 'Sim_Pct_NoSave'),
                ('rank', True, 'Sim_Rank_Save'),
                ('percentage', True, 'Sim_Pct_Save')
            ]
            
            entry = {
                'Name': name,
                'Season': season,
                'Real_Survival_Week': real_week
            }
            
            for method, save, col_name in configs:
                # 关键逻辑：如果配置与真实规则一致，直接填写真实结果，不模拟
                if (method, save) == real_rule_config:
                    entry[col_name] = real_week
                else:
                    sim_week = self.simulate_target(name, season, method, save)
                    entry[col_name] = sim_week
                    
            results.append(entry)
            
        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv('simulation_results_4_rules.csv', index=False)
        print("Done! Results saved to 'simulation_results_4_rules.csv'")
        return results_df

# === 运行脚本 ===
# 替换为你的文件名
data_file = '2026_MCM_Problem_C_Data.csv'
estimates_file = 'QF‘s solution/Bayes_Elo/real_figures/fan_vote_estimates_weekly.csv'
list_file = 'QF‘s solution/Q2_QFv2/Q2_2/controversal_name_list.csv'

sim = DancingSimulator(data_file, estimates_file, list_file)
df_res = sim.run_batch_simulation()
print(df_res.head())