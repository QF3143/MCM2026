# evaluator_with_metrics.py

import json
import os
from collections import defaultdict
import numpy as np

# ✅ 从已有模块导入基础类（假设 Q1_Basic_Model.py 在同一目录）
from Q1_Basic_Model import InverseFanVoteEstimator


# ==========================================
# 扩展评估器：在原模型基础上增加稳定性指标计算
# ==========================================
class Evaluator(InverseFanVoteEstimator):
    """
    Extends the base estimator to run multiple global simulations
    and calculate Consistency (probability assigned to actual loser)
    and Certainty (standard deviation across runs).
    """

    def run_with_metrics(self, n_global_runs=20):
        print(f"Running stability analysis with {n_global_runs} global simulations...")
        
        all_runs_history = defaultdict(lambda: defaultdict(list))
        consistency_scores = defaultdict(list)
        
        for run_id in range(n_global_runs):
            if run_id % 5 == 0:
                print(f"  - Simulation run {run_id}/{n_global_runs}")

            # 重置 X 到均匀先验（关键！保证每次模拟独立）
            self.X = np.ones(len(self.contestants)) / len(self.contestants)
            
            # 模拟每周（根据数据自动推断最大周数）
            for week in range(1, self.max_data_week + 1):
                payload = self.get_week_data_and_target(week)  # ← 来自原类的方法
                if not payload:
                    break
                names, judges, elim_idx, global_idx = payload
                
                # 内层优化循环
                for _ in range(50):
                    current_priors = self.normalize(self.X[global_idx])
                    probs = self.simulate_step(judges, current_priors)
                    self.update_parameters(global_idx, probs, elim_idx)
                
                # 计算一致性：模型对实际淘汰者的预测概率
                final_probs = self.simulate_step(judges, self.normalize(self.X[global_idx]))
                if len(elim_idx) > 0:
                    actual_loser_prob = final_probs[elim_idx[0]]
                    consistency_scores[week].append(actual_loser_prob)
                else:
                    consistency_scores[week].append(1.0)  # 无淘汰视为完美一致
                
                # 记录本轮估计值（用于确定性计算）
                current_priors = self.normalize(self.X[global_idx])
                for name, score in zip(names, current_priors):
                    all_runs_history[week][name].append(score)
                
                # 时间演化（记忆机制）
                judge_share = self.normalize(judges)
                new_priors = self.alpha * current_priors + (1 - self.alpha) * judge_share
                self.X[global_idx] = new_priors
                self.X = self.normalize(self.X)

        return self._compile_report(all_runs_history, consistency_scores)

    def _compile_report(self, history, consistency):
        report = []
        for week in sorted(history.keys()):
            week_data = history[week]
            avg_consistency = float(np.mean(consistency[week]))
            
            contestants_stats = {}
            for name, scores in week_data.items():
                scores = np.array(scores)
                mean_score = float(np.mean(scores))
                std_dev = float(np.std(scores))
                cv = float(std_dev / mean_score) if mean_score > 0 else 0.0
                contestants_stats[name] = {
                    'mean_vote': mean_score,
                    'certainty_score': std_dev,
                    'cv': cv
                }
            
            report.append({
                'week': week,
                'consistency_index': avg_consistency,
                'contestants': contestants_stats
            })
        return report


# ==========================================
# 封装函数：接受路径参数，保存 JSON
# ==========================================
def evaluate_and_save(input_csv_path: str, output_json_path: str, season_id: int = 5, n_global_runs: int = 20):
    """
    运行多轮评估并保存结果。
    
    Parameters:
        input_csv_path: 输入 CSV 路径
        output_json_path: 输出 JSON 路径
        season_id: 赛季 ID
        n_global_runs: 模拟次数
    """
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    evaluator = Evaluator(
        data_path=input_csv_path,
        season_id=season_id,
        n_trials=1000,
        step_size=0.05,
        memory_alpha=0.7
    )
    
    metrics = evaluator.run_with_metrics(n_global_runs=n_global_runs)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_json_path}")
    return metrics


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 🔧 请根据你的实际路径修改以下两行
    INPUT_PATH = '/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/2026_MCM_Problem_C/2026_MCM_Problem_C_Data.csv'
    OUTPUT_PATH = '/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/2026_MCM_Problem_C/evaluation_metrics.json'

    results = evaluate_and_save(
        input_csv_path=INPUT_PATH,
        output_json_path=OUTPUT_PATH,
        season_id=5,
        n_global_runs=20
    )

    # 预览前两周
    print("\n🔍 Preview of first two weeks:")
    print(json.dumps(results[:2], indent=2))