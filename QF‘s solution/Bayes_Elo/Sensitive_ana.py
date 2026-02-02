import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from bayes_elo_V3 import BayesianEloEstimator

def preprocess_dwts_data(file_path):
    """
    将原始 2026_MCM_Problem_C_Data.csv 转换为模型所需的周报格式
    """
    df = pd.read_csv(file_path)
    rows = []
    
    for _, row in df.iterrows():
        name = row['celebrity_name']
        season = row['season']
        result_str = str(row['results'])
        
        # 解析淘汰周
        elim_week = -1
        status_final = 'Finalist'
        if 'Eliminated Week' in result_str:
            try:
                elim_week = int(result_str.split('Week ')[1])
                status_final = 'Eliminated'
            except: pass
        elif 'Withdrew' in result_str:
            status_final = 'Withdrew'
        
        for w in range(1, 15): # 遍历可能的周数
            j_cols = [f'week{w}_judge{i}_score' for i in range(1, 5)]
            scores = []
            for col in j_cols:
                if col in df.columns and pd.notnull(row[col]) and row[col] != 'N/A':
                    try: scores.append(float(row[col]))
                    except: pass
            
            if not scores or sum(scores) == 0: continue
            
            # 状态判定
            if status_final == 'Eliminated' and w == elim_week:
                status = 'Eliminated'
            elif status_final == 'Withdrew' and f'Week {w}' in result_str:
                status = 'Withdrew'
            elif status_final == 'Eliminated' and w > elim_week:
                status = 'Out'
            else:
                status = 'In'
            
            rows.append({
                'season': season, 'week': w, 'name': name,
                'judge_score': sum(scores), 'status': status
            })
            
    res_df = pd.DataFrame(rows)
    # 计算 judge_pct
    res_df['judge_pct'] = res_df.groupby(['season', 'week'])['judge_score'].transform(lambda x: x / x.sum())
    return res_df

class SensitivityAnalyzer:
    def __init__(self, data, baseline_params):
        self.data = data
        self.baseline_params = baseline_params
        self.results = []
        print(">>> 正在运行基准模型 (Baseline)...")
        self.baseline_rankings, self.baseline_consist = self._run_model(baseline_params)
        
    def _run_model(self, params):
        # 实例化模型
        model = BayesianEloEstimator(**params)
        # 运行推断
        output_df = model.run_inference(self.data)
        # 获取汇总排名表
        rank_df = model.get_final_rankings()
        
        # --- 修复 KeyError 的核心逻辑 ---
        # 1. 在汇总表中定位评分列 (可能是 'final_elo' 或 'elo_rating')
        rating_col = 'final_elo' if 'final_elo' in rank_df.columns else 'elo_rating'
        rank_series = rank_df.set_index('name')[rating_col]
        
        # 2. 从推断结果中提取一致性得分 (consistency_score)
        if 'consistency_score' in output_df.columns:
            # 过滤掉 NaN 值的周（如果有）
            valid_scores = output_df['consistency_score'].dropna()
            consistency = valid_scores.mean() if not valid_scores.empty else 0
        else:
            consistency = 0

        return rank_series, consistency

    def run_experiment(self, param_name, values):
        print(f">>> 测试参数敏感性: {param_name} = {values}")
        for val in values:
            test_params = self.baseline_params.copy()
            test_params[param_name] = val
            # 敏感性实验中可适当降低模拟次数以提升速度
            test_params['n_simulations'] = 500 
            
            elo_series, avg_consist = self._run_model(test_params)
            
            # 计算与基准排名的相关性
            common = self.baseline_rankings.index.intersection(elo_series.index)
            corr, _ = spearmanr(self.baseline_rankings.loc[common], elo_series.loc[common])
            
            self.results.append({
                'parameter': param_name, 'value': val,
                'spearman_corr': corr, 'consistency': avg_consist
            })

    def plot_report(self):
        df_res = pd.DataFrame(self.results)
        unique_params = df_res['parameter'].unique()
        
        fig, axes = plt.subplots(1, len(unique_params), figsize=(6*len(unique_params), 5))
        if len(unique_params) == 1: axes = [axes]
        
        for i, p in enumerate(unique_params):
            subset = df_res[df_res['parameter'] == p]
            axes[i].plot(subset['value'], subset['spearman_corr'], 'o-', label=r'Rank Stability ($\rho$)', color='#1f77b4', linewidth=2)
            axes[i].plot(subset['value'], subset['consistency'], 's--', label='Model Consistency', color='#ff7f0e')
            axes[i].set_title(f'Sensitivity: {p}', fontsize=14)
            axes[i].set_xlabel('Parameter Value')
            axes[i].grid(True, linestyle=':', alpha=0.6)
            axes[i].legend()
            
        plt.tight_layout()
        plt.savefig('QF‘s solution/Bayes_Elo/Model_diag_figure/sensitivity_analysis_report.png', dpi=300)
        print("\n✅ 分析完成！可视化结果已保存至: sensitivity_analysis_report.png")

if __name__ == "__main__":
    # 1. 加载并清洗数据
    raw_data_path = '2026_MCM_Problem_C_Data.csv'
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found.")
    else:
        df_clean = preprocess_dwts_data(raw_data_path)
        
        # 2. 定义基准参数 (Baseline)
        baseline = {
            'base_k_factor': 48.0,
            'temperature': 150.0,
            'n_simulations': 500, # 实验建议值
            'noise_std': 0.3,
            'judge_weight': 0.5
        }

        analyzer = SensitivityAnalyzer(df_clean, baseline)

        # 3. 执行参数波动实验
        analyzer.run_experiment('base_k_factor', [32, 40, 48, 56, 64])
        analyzer.run_experiment('temperature', [50, 100, 150, 200, 250])
        analyzer.run_experiment('noise_std', [0.1, 0.2, 0.3, 0.4, 0.5])

        # 4. 生成图表和日志
        analyzer.plot_report()
        pd.DataFrame(analyzer.results).to_csv('QF‘s solution/Bayes_Elo/Model_diag_figure/sensitivity_analysis_log.csv', index=False)