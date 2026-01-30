import numpy as np
import pandas as pd
from scipy.stats import rankdata

class InverseFanVoteEstimator:
    """
    Implements the Inverse Parameter Estimation utilizing Monte Carlo Simulation.
    Logic: Pure Data-Driven Elimination Detection (Score Drop-off).
    """
    
    def __init__(self, data_path, season_id, scoring_method='percent', n_trials=1000, step_size=0.05, memory_alpha=0.7):
        self.df = pd.read_csv(data_path)
        self.season_data = self.df[self.df['season'] == season_id].copy().reset_index(drop=True)
        self.n_trials = n_trials
        self.lr = step_size
        self.alpha = memory_alpha
        self.scoring_method = scoring_method
        self.contestants = self.season_data['celebrity_name'].unique().tolist()
        
        # Initialize latent fan support X (Uniform Distribution)
        self.X = np.ones(len(self.contestants)) / len(self.contestants)
        
        # Determine the max possible week in the dataset for this season
        # Filter columns that look like 'weekX_judgeY_score'
        week_cols = [c for c in self.season_data.columns if 'week' in c and 'judge' in c]
        weeks = [int(c.split('week')[1].split('_')[0]) for c in week_cols]
        self.max_data_week = max(weeks) if weeks else 0
        print(f"ℹ️ Season {season_id}: Max data week detected = {self.max_data_week}")

    def normalize(self, v):
        return v / np.sum(v)

    def get_week_data_and_target(self, week):
        """
        Returns:
        1. active_names: Names of contestants performing THIS week.
        2. active_scores: Judge scores for THIS week.
        3. elim_indices_local: Indices (in active list) of contestants who vanish NEXT week.
        4. global_indices: Indices in the main self.X vector.
        """
        # 1. Identify who is active THIS week (Score > 0)
        score_cols_now = [c for c in self.season_data.columns if f'week{week}_judge' in c]
        scores_now_sum = self.season_data[score_cols_now].fillna(0).sum(axis=1)
        
        active_idx = scores_now_sum[scores_now_sum > 0].index
        if len(active_idx) == 0:
            return None # Season clearly over
            
        active_names = self.season_data.loc[active_idx, 'celebrity_name'].values
        active_scores = scores_now_sum.loc[active_idx].values
        global_indices = [self.contestants.index(n) for n in active_names]
        
        # 2. Look Ahead: Who is active NEXT week?
        # If this is the absolute last week of data, everyone "vanishes" but it's not an elimination.
        next_week = week + 1
        elim_indices_local = []
        
        if next_week <= self.max_data_week:
            score_cols_next = [c for c in self.season_data.columns if f'week{next_week}_judge' in c]
            
            # Check if NEXT week has ANY data at all (sometimes cols exist but are empty)
            scores_next_sum_all = self.season_data[score_cols_next].fillna(0).sum(axis=1).sum()
            
            if scores_next_sum_all > 0:
                # Next week is a valid round. Let's see who didn't make it.
                scores_next_sum = self.season_data.loc[active_idx, score_cols_next].fillna(0).sum(axis=1)
                
                # Those with score 0 in next week are the eliminated ones
                # We use numpy where to get indices relative to the 'active_names' array
                elim_mask = (scores_next_sum == 0).values
                elim_indices_local = np.where(elim_mask)[0]
            else:
                # Next week column exists but is empty -> Implies Season Finale reached.
                pass
        else:
            # No next week column -> Season Finale.
            pass
            
        return active_names, active_scores, elim_indices_local, global_indices

    def simulate_step(self, judge_scores, fan_priors):
        n_active = len(judge_scores)
        elim_counts = np.zeros(n_active)
        
        if self.scoring_method == 'percent':
            judge_component = judge_scores / np.sum(judge_scores)
        else:
            judge_component = rankdata(judge_scores, method='average')

        for _ in range(self.n_trials):
            noise = np.random.uniform(0.9, 1.1, n_active) 
            sample_votes = fan_priors * noise
            
            if self.scoring_method == 'percent':
                fan_pct = sample_votes / np.sum(sample_votes)
                total_score = judge_component + fan_pct
            else:
                fan_points = rankdata(sample_votes, method='average')
                total_score = judge_component + fan_points
            
            min_score = np.min(total_score)
            min_score_idx = np.where(total_score == min_score)[0]
            eliminated = np.random.choice(min_score_idx)
            elim_counts[eliminated] += 1
            
        return elim_counts / self.n_trials

    def update_parameters(self, global_indices, elim_probs, actual_elim_idx):
        if len(actual_elim_idx) == 0:
            return # No elimination happened (or finale), nothing to learn
            
        current_X_subset = self.X[global_indices]
        
        target = np.zeros(len(current_X_subset))
        target[actual_elim_idx] = 1.0 # These people disappeared next week
            
        error = target - elim_probs
        update_factor = 1.0 - (self.lr * error)
        update_factor = np.clip(update_factor, 0.1, 2.0)
        
        self.X[global_indices] = current_X_subset * update_factor
        self.X = self.normalize(self.X)

    def run(self):
        print(f"Starting Reconstruction for Season {self.season_data['season'].iloc[0]} (Data-Driven Mode)...")
        history = []
        
        # Iterate up to the max week found in data
        for week in range(1, self.max_data_week + 1):
            
            payload = self.get_week_data_and_target(week)
            if not payload: 
                print(f"Week {week}: No active contestants found. Stopping.")
                break
                
            names, judges, elim_idx, global_idx = payload
            
            # Identify who we are trying to eliminate
            if len(elim_idx) > 0:
                targets = names[elim_idx]
                target_str = ", ".join(targets)
                print(f"Week {week}: Detected Elimination of -> [{target_str}] (Score dropped to 0 next week)")
            else:
                target_str = "None (Season Finale or Non-Elimination)"
                print(f"Week {week}: No eliminations detected (Next week everyone is 0 or everyone survived).")
            
            # Optimization Loop (Only if there is someone to eliminate)
            if len(elim_idx) > 0:
                for i in range(50): 
                    current_priors = self.normalize(self.X[global_idx])
                    probs = self.simulate_step(judges, current_priors)
                    self.update_parameters(global_idx, probs, elim_idx)
            
            # Temporal Update (Always happens, even in finale, to reflect latest performance)
            current_priors = self.normalize(self.X[global_idx])
            judge_share = self.normalize(judges)
            new_priors = self.alpha * current_priors + (1 - self.alpha) * judge_share
            self.X[global_idx] = new_priors
            self.X = self.normalize(self.X)
            
            history.append({
                "week": week,
                "X": dict(zip(names, current_priors))
            })
            
            # If no one was eliminated, and we are near the end, we might want to stop early?
            # Or just let it run. The code handles "len(elim_idx) == 0" gracefully.
            
        return history

# --- 使用示例 ---
def estimate_season_datadriven(season, input_path, output_path, method='percent'):
    estimator = InverseFanVoteEstimator(input_path, season_id=season, scoring_method=method)
    results = estimator.run()
    
    output_rows = []
    for week_data in results:
        week = week_data["week"]
        for name, fan_prob in week_data["X"].items():
            output_rows.append({
                "week": week,
                "celebrity_name": name,
                "estimated_fan_support": fan_prob
            })
            
    pd.DataFrame(output_rows).to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == "__main__":
    # 强制只看分数：如果某人 Week 10 有分，Week 11 没分，那他就是在 Week 10 没的。
    # 如果所有人在 Week 11 也没分，那就说明 Week 10 是最后一舞。
    season = 1
    m = "rank"
    estimate_season_datadriven(season, '2026_MCM_Problem_C_Data.csv', f'/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/Q1_result/{season}_{m}_datadriven.csv', method=m)