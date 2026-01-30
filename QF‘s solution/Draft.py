import numpy as np
import pandas as pd
from collections import defaultdict
import json

# ==========================================
# Part 1: 你的基础估算器类 (Base Class)
# ==========================================
class InverseFanVoteEstimator:
    """
    Implements the Inverse Parameter Estimation utilizing Monte Carlo Simulation.
    Goal: Reconstruct latent fan voting distributions (X) based on elimination outcomes.
    """
    
    def __init__(self, data_path, season_id, n_trials=1000, step_size=0.05, memory_alpha=0.7):
        """
        :param season_id: The specific season to analyze (e.g., 5)
        :param n_trials: M, number of Monte Carlo samples per iteration
        :param step_size: n/lr, learning rate for parameter update
        :param memory_alpha: a, weight of historical popularity (0-1)
        """
        self.df = pd.read_csv(data_path)
        self.season_data = self.df[self.df['season'] == season_id].copy().reset_index(drop=True)
        self.n_trials = n_trials
        self.lr = step_size
        self.alpha = memory_alpha
        self.contestants = self.season_data['celebrity_name'].unique().tolist()
        
        # Initialize latent fan support X (Uniform Distribution)
        self.X = np.ones(len(self.contestants)) / len(self.contestants)
        
        # Pre-process elimination weeks
        self.season_data['elim_week_num'] = self.season_data['results'].apply(self._parse_week)

    def _parse_week(self, text):
        """Parses 'Eliminated Week 4' into integer 4. Returns 99 for finalists."""
        text = str(text)
        if "Eliminated Week" in text:
            return int(text.split("Week")[-1])
        return 99  # Safe value for winners/runner-ups

    def normalize(self, v):
        """Ensures vector sums to 1 (Simplex Constraint)."""
        return v / np.sum(v)

    def get_week_data(self, week):
        """Extracts active contestants and their judge scores for a given week."""
        active_mask = self.season_data['elim_week_num'] >= week
        score_cols = [c for c in self.season_data.columns if f'week{week}_judge' in c]
        
        # Calculate sum of judge scores
        current_scores = self.season_data.loc[active_mask, score_cols].fillna(0).sum(axis=1)
        
        # Filter out those who didn't perform (score=0)
        valid_idx = current_scores[current_scores > 0].index
        
        if len(valid_idx) == 0: return None
        
        names = self.season_data.loc[valid_idx, 'celebrity_name'].values
        scores = current_scores.loc[valid_idx].values
        
        # Identify who was actually eliminated this week
        elim_mask = self.season_data.loc[valid_idx, 'elim_week_num'] == week
        elim_indices = np.where(elim_mask)[0] # Local indices in the current arrays
        
        # Map active contestants back to global X vector indices
        global_indices = [self.contestants.index(n) for n in names]
        
        return names, scores, elim_indices, global_indices

    def simulate_step(self, judge_scores, fan_priors):
        """
        Performs Monte Carlo Simulation (M samples).
        Returns the probability of elimination for each contestant.
        """
        n_active = len(judge_scores)
        elim_counts = np.zeros(n_active)
        
        # Convert absolute scores to percentages
        judge_pct = judge_scores / np.sum(judge_scores)
        
        for _ in range(self.n_trials):
            # 1. Perturb Fan Votes (Simulate randomness)
            # Noise model: X_sample ~ X_prior * Uniform(0.9, 1.1)
            noise = np.random.uniform(1 - 0.1, 1 + 0.1, n_active) 
            sample_votes = fan_priors * noise
            fan_pct = sample_votes / np.sum(sample_votes)
            
            # 2. Compute Combined Score
            total_score = judge_pct + fan_pct
            
            # 3. Determine Elimination (Lowest Score)
            min_score_idx = np.where(total_score == np.min(total_score))[0]
            eliminated = np.random.choice(min_score_idx)
            elim_counts[eliminated] += 1
            
        return elim_counts / self.n_trials

    def update_parameters(self, global_indices, elim_probs, actual_elim_idx):
        """Updates X based on the difference between simulated and actual outcomes."""
        current_X_subset = self.X[global_indices]
        
        # Target Vector: 1 for actual loser, 0 for survivors
        target = np.zeros(len(current_X_subset))
        if len(actual_elim_idx) > 0:
            target[actual_elim_idx] = 1.0
            
        error = target - elim_probs
        
        update_factor = 1.0 - (self.lr * error)
        update_factor = np.clip(update_factor, 0.1, 2.0) 
        
        self.X[global_indices] = current_X_subset * update_factor
        self.X = self.normalize(self.X)

    def run(self):
        """Standard single run (Legacy support)."""
        print(f"Starting Reconstruction for Season {self.season_data['season'].iloc[0]}...")
        history = []
        for week in range(1, 11):
            data = self.get_week_data(week)
            if not data: break
            names, judges, elim_idx, global_idx = data
            
            for i in range(50): 
                current_priors = self.normalize(self.X[global_idx])
                probs = self.simulate_step(judges, current_priors)
                self.update_parameters(global_idx, probs, elim_idx)
            
            # Memory update
            current_priors = self.normalize(self.X[global_idx])
            judge_share = self.normalize(judges)
            new_priors = self.alpha * current_priors + (1 - self.alpha) * judge_share
            self.X[global_idx] = new_priors
            self.X = self.normalize(self.X)
            
            history.append({"week": week, "X": dict(zip(names, current_priors))})
        return history

# ==========================================
# Part 2: 扩展评估器类 (Extension Class)
# ==========================================
class Evaluator(InverseFanVoteEstimator):
    """
    Extends the base estimator to run multiple global simulations
    and calculate Consistency and Certainty metrics.
    """
    
    def run_with_metrics(self, n_global_runs=20):
        """
        Runs the estimator multiple times to calculate Certainty (Std Dev)
        and Consistency (Prob of Truth).
        :param n_global_runs: Number of full simulations to run for stability check.
        """
        print(f"Running stability analysis with {n_global_runs} global simulations...")
        
        # 存储所有运行的历史数据
        # Structure: week -> contestant -> list of estimated_scores
        all_runs_history = defaultdict(lambda: defaultdict(list))
        # 存储一致性分数
        # Structure: week -> list of probabilities assigned to actual loser
        consistency_scores = defaultdict(list)
        
        for run_id in range(n_global_runs):
            if run_id % 5 == 0:
                print(f"  - Simulation run {run_id}/{n_global_runs}")

            # 重置 X (重要!)
            self.X = np.ones(len(self.contestants)) / len(self.contestants)
            
            # 运行标准逻辑
            for week in range(1, 11):
                data = self.get_week_data(week)
                if not data: break
                names, judges, elim_idx, global_idx = data
                
                # --- Inner Optimization Loop ---
                for i in range(50): 
                    current_priors = self.normalize(self.X[global_idx])
                    probs = self.simulate_step(judges, current_priors)
                    self.update_parameters(global_idx, probs, elim_idx)
                
                # --- [关键点 1] 计算一致性 (Consistency) ---
                # 在参数更新完毕后，最后测一次：模型认为由于实际结果发生的概率是多少？
                final_probs = self.simulate_step(judges, self.normalize(self.X[global_idx]))
                
                if len(elim_idx) > 0:
                    # 实际被淘汰者的索引在 elim_idx[0]
                    actual_loser_prob = final_probs[elim_idx[0]]
                    consistency_scores[week].append(actual_loser_prob)
                else:
                    consistency_scores[week].append(1.0) # 无人淘汰，完美一致
                
                # --- 记录本轮的估计值用于计算确定性 ---
                current_priors = self.normalize(self.X[global_idx])
                for name, score in zip(names, current_priors):
                    all_runs_history[week][name].append(score)
                    
                # Time Evolution (Memory)
                judge_share = self.normalize(judges)
                new_priors = self.alpha * current_priors + (1 - self.alpha) * judge_share
                self.X[global_idx] = new_priors
                self.X = self.normalize(self.X)

        return self._compile_report(all_runs_history, consistency_scores)

    def _compile_report(self, history, consistency):
        """生成最终报告：均值、确定性(Std)、一致性(Prob)"""
        report = []
        for week in sorted(history.keys()):
            week_data = history[week]
            cons_list = consistency[week]
            
            # 1. 计算一致性 (取平均概率)
            avg_consistency = np.mean(cons_list)
            
            # 2. 计算确定性 (对每个选手算 Std)
            week_stats = {}
            for name, scores in week_data.items():
                mean_score = np.mean(scores)
                std_dev = np.std(scores) # 这就是确定性指标！越低越好
                
                week_stats[name] = {
                    'mean_vote': mean_score,
                    'certainty_score': std_dev, # 标准差
                    'cv': std_dev / mean_score if mean_score > 0 else 0 # 变异系数
                }
            
            report.append({
                'week': week,
                'consistency_index': avg_consistency, # 这一周结果的合理性
                'contestants': week_stats
            })
        return report

# ==========================================
# Part 3: 执行代码
# ==========================================
if __name__ == "__main__":
    # 请确保将路径修改为你本地实际的文件路径
    file_path = '/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/2026_MCM_Problem_C/2026_MCM_Problem_C_Data.csv'
    
    # 实例化 Evaluator 而不是 InverseFanVoteEstimator
    evaluator = Evaluator(file_path, season_id=5)
    
    # 运行带指标评估的模式 (跑20次全局模拟来计算方差)
    metrics = evaluator.run_with_metrics(n_global_runs=20)
    
    output_path = '/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/2026_MCM_Problem_C/evaluation_metrics.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存至: {output_path}")
    
    # 只打印前两周的结果以避免刷屏
    print(json.dumps(metrics[:2], indent=2))