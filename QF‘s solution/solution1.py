import numpy as np
import pandas as pd

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
        # X represents the underlying "True Popularity" vector
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
        # Filter contestants active in this week
        active_mask = self.season_data['elim_week_num'] >= week
        
        # Get judge scores columns
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
        
        # Convert absolute scores to percentages (Problem Rule)
        judge_pct = judge_scores / np.sum(judge_scores)
        
        for _ in range(self.n_trials):
            # 1. Perturb Fan Votes (Simulate randomness in voting process)
            # Noise model: X_sample ~ X_prior * Uniform(0.9, 1.1)
            noise = np.random.uniform(1 - 0.1, 1 + 0.1, n_active) 
            sample_votes = fan_priors * noise
            fan_pct = sample_votes / np.sum(sample_votes)
            
            # 2. Compute Combined Score
            total_score = judge_pct + fan_pct
            
            # 3. Determine Elimination (Lowest Score)
            min_score_idx = np.where(total_score == np.min(total_score))[0]
            # Randomly break ties
            eliminated = np.random.choice(min_score_idx)
            elim_counts[eliminated] += 1
            
        return elim_counts / self.n_trials  # fi: Probability of elimination

    def update_parameters(self, global_indices, elim_probs, actual_elim_idx):
        """
        Updates X based on the difference between simulated and actual outcomes.
        Algorithm: Heuristic Gradient Descent
        """
        current_X_subset = self.X[global_indices]
        
        # Target Vector: 1 for actual loser, 0 for survivors
        target = np.zeros(len(current_X_subset))
        if len(actual_elim_idx) > 0:
            target[actual_elim_idx] = 1.0
            
        # Error Signal: Delta = Target - Predicted_Prob
        # If Delta > 0 (Actual loser had low elim prob): We overestimated their fan base -> Decrease X
        # If Delta < 0 (Survivor had high elim prob): We underestimated their fan base -> Increase X
        error = target - elim_probs
        
        # Update Rule: X_new = X_old * (1 - learning_rate * error)
        # Note: If error is positive (should have died but didn't in sim), we reduce X.
        # Wait, if elim_prob is LOW, it means X is HIGH. To increase elim_prob, we must DECREASE X.
        # So X_new should be inversely proportional to Error?
        # Let's align signs:
        # Target=1, Prob=0.1 -> Error=0.9. We need to DECREASE X.
        # Formula: X * (1 - lr * 0.9) -> Decreases. Correct.
        # Target=0, Prob=0.9 -> Error=-0.9. We need to INCREASE X.
        # Formula: X * (1 - lr * -0.9) = X * (1 + 0.81) -> Increases. Correct.
        
        update_factor = 1.0 - (self.lr * error)
        update_factor = np.clip(update_factor, 0.1, 2.0) # Prevent explosion/negatives
        
        self.X[global_indices] = current_X_subset * update_factor
        
        # Normalize Global X (Simplex Constraint)
        self.X = self.normalize(self.X)

    def run(self):
        print(f"Starting Reconstruction for Season {self.season_data['season'].iloc[0]}...")
        history = []
        
        # Iterate through weeks
        for week in range(1, 11):
            data = self.get_week_data(week)
            if not data: break
            names, judges, elim_idx, global_idx = data
            
            # Inner Optimization Loop (Iterative fitting for current week)
            # "Repeat until X reaches a satisfactory threshold"
            for i in range(50): 
                # Extract current priors
                current_priors = self.normalize(self.X[global_idx])
                
                # Step 1: Simulate
                probs = self.simulate_step(judges, current_priors)
                
                # Step 2: Update
                self.update_parameters(global_idx, probs, elim_idx)
            
            # Step 3: Temporal Evolution (Memory)
            # Xt = a * X(t-1) + (1-a) * Performance
            # We assume judges score is a proxy for "earned" popularity
            current_priors = self.normalize(self.X[global_idx])
            judge_share = self.normalize(judges)
            
            new_priors = self.alpha * current_priors + (1 - self.alpha) * judge_share
            self.X[global_idx] = new_priors
            self.X = self.normalize(self.X) # Global Renormalization
            
            # Logging
            loser_name = names[elim_idx[0]] if len(elim_idx) > 0 else "None"
            top_fan = names[np.argmax(current_priors)]
            print(f"Week {week}: Eliminated {loser_name}. Top Fan Favorite: {top_fan}")
            
            history.append({
                "week": week,
                "X": dict(zip(names, current_priors))
            })
            
        return history

# --- Execution ---
# Assuming the file is in the current directory
estimator = InverseFanVoteEstimator('/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/2026_MCM_Problem_C/2026_MCM_Problem_C_Data.csv', season_id=5)
results = estimator.run()
# --- 将 results 输出为 CSV ---
import os

# 准备输出数据
output_rows = []
for week_data in results:
    week = week_data["week"]
    for name, fan_prob in week_data["X"].items():
        output_rows.append({
            "week": week,
            "celebrity_name": name,
            "estimated_fan_support": fan_prob
        })

# 转为 DataFrame
output_df = pd.DataFrame(output_rows)

# 确保输出目录存在（可选）
output_dir = "/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/2026_MCM_Problem_C/"
os.makedirs(output_dir, exist_ok=True)

# 保存为 CSV
output_path = os.path.join(output_dir, f"season5_fan_support_estimates.csv")
output_df.to_csv(output_path, index=False)

print(f"\n✅ Estimated fan support saved to: {output_path}")