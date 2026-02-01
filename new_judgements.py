import pandas as pd
import numpy as np

def calculate_new_scores():
    print("Loading data...")
    # Load constraints
    try:
        clean_df = pd.read_csv('cleaned_weekly_data.csv')
        fan_df = pd.read_csv('fan_vote_estimates_weekly.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Merge DataFrames
    # cleanliness check: names might need normalization if mismatch, but assuming same source
    merged = pd.merge(clean_df, fan_df, on=['season', 'week', 'name'], how='inner', suffixes=('', '_fan'))
    
    # Parameters
    LAMBDA = 100
    W_J = 0.8
    W_F = 0.2
    BETA = 0.05
    
    # Formula Calculation
    # S_total = w_j * P_judge + w_f * ln(1 + lambda * P_fan) / ln(1 + lambda) + bonus * beta
    
    # Ensure P_judge and P_fan are used correctly. 
    # clean_df has 'judge_pct'. fan_df has 'est_fan_pct'.
    
    def log_fan_score(p_fan):
        # Handle potential zeros or negatives (though pct shouldn't be negative)
        p_fan = max(0, p_fan)
        numerator = np.log(1 + LAMBDA * p_fan)
        denominator = np.log(1 + LAMBDA)
        return numerator / denominator

    merged['log_fan_term'] = merged['est_fan_pct'].apply(log_fan_score)
    
    # Bonus: 'is_most_improved' comes from cleaned_weekly_data.csv
    # Fill NaN with 0 just in case
    merged['is_most_improved'] = merged['is_most_improved'].fillna(0)
    
    merged['new_score'] = W_J * merged['judge_pct'] + W_F * merged['log_fan_term'] + merged['is_most_improved'] * BETA
    
    # Calculate New Ranks with Tie-Breaking
    # To avoid tied ranks (which complicate the "Bottom 2" logic), we use a multi-level sort:
    # 1. New Score (Descending)
    # 2. Judge Score Pct (Descending) - Tie-Breaker 1: Prioritize technical merit
    # 3. Fan Score Pct (Descending) - Tie-Breaker 2: Prioritize popularity
    # 4. Name (Ascending) - Final arbitrary tie-breaker for deterministic output
    
    merged = merged.sort_values(
        by=['season', 'week', 'new_score', 'judge_pct', 'est_fan_pct', 'name'],
        ascending=[True, True, False, False, False, True]
    )
    
    # Assign unique ranks (1, 2, 3...) within each group
    merged['new_rank'] = merged.groupby(['season', 'week']).cumcount() + 1
    
    # Identify Bottom 2 in New System
    # We need to know how many people are in that week
    merged['contestant_count'] = merged.groupby(['season', 'week'])['name'].transform('count')
    # Rank N and N-1 are bottom 2.
    # Note: rank is 1-based. So if count is 10, ranks 9 and 10 are bottom.
    # Logic: new_rank >= contestant_count - 1
    
    merged['in_new_bottom_2'] = merged['new_rank'] >= (merged['contestant_count'] - 1)

    # ==========================================
    # SUDDEN DEATH SIMULATION (Q4 Extension)
    # ==========================================
    print("Simulating Sudden Death Duels...")
    
    # Initialize columns
    merged['duel_win_prob'] = 0.0
    merged['won_duel'] = False
    
    # Iterate through each week to simulate the duel for the bottom 2
    for (season, week), group in merged.groupby(['season', 'week']):
        bottom_2 = group[group['in_new_bottom_2']]
        
        # We expect exactly 2 contestants. If not (e.g. only 1 left?), skip
        if len(bottom_2) != 2:
            continue
            
        # Get contestants
        c1 = bottom_2.iloc[0]
        c2 = bottom_2.iloc[1]
        
        # Parameters for simulation
        # Mean = Elo, Std = Rating Deviation
        # If RD is missing, default to 350 (standard initial RD)
        c1_elo = c1['elo_rating'] if pd.notnull(c1['elo_rating']) else 1500
        c1_rd = c1['rating_deviation'] if pd.notnull(c1['rating_deviation']) else 350
        
        c2_elo = c2['elo_rating'] if pd.notnull(c2['elo_rating']) else 1500
        c2_rd = c2['rating_deviation'] if pd.notnull(c2['rating_deviation']) else 350
        
        # Monte Carlo Simulation
        N_SIMS = 2000
        # Vectorized sampling
        c1_perf = np.random.normal(c1_elo, c1_rd, N_SIMS)
        c2_perf = np.random.normal(c2_elo, c2_rd, N_SIMS)
        
        # Count wins
        c1_wins = np.sum(c1_perf > c2_perf)
        c1_prob = c1_wins / N_SIMS
        c2_prob = 1.0 - c1_prob
        
        # Record probabilities
        merged.loc[bottom_2.index[0], 'duel_win_prob'] = c1_prob
        merged.loc[bottom_2.index[1], 'duel_win_prob'] = c2_prob
        
        # Determine Winner (Deterministically based on prob > 0.5)
        # Identify index of winner
        if c1_prob > c2_prob:
             merged.loc[bottom_2.index[0], 'won_duel'] = True
        elif c2_prob > c1_prob:
             merged.loc[bottom_2.index[1], 'won_duel'] = True
        else:
             # Tie in simulation (rare), tie-break by raw Elo
             if c1_elo > c2_elo:
                 merged.loc[bottom_2.index[0], 'won_duel'] = True
             else:
                 merged.loc[bottom_2.index[1], 'won_duel'] = True
                 
    # ==========================================
    
    # Compare with Reality
    # Reality: 'is_eliminated' == 1.
    # We want to see if the eliminated person is in the new bottom 2.
    
    # Prepare Output CSV
    # User asked for: Final Rank, Judge Avg, Judge Rank, Pred Fan Vote, Fan Rank, CI, Credibility, Consistency
    # We have these in 'merged' now.
    
    output_cols = [
        'season', 'week', 'name', 
        'placement',          # Final Place
        'avg_score',          # Judge Avg
        'judge_rank',         # Judge Rank
        'est_fan_pct',        # Pred Fan Vote
        # Fan Vote Rank? We need to calculate it.
        'judge_pct',
        'is_most_improved',
        'new_score',
        'new_rank',
        'ci_95_lower', 'ci_95_upper', # CI
        'certainty_score',    # Credibility?
        'consistency_score',   # Consistency
        'is_eliminated',
        'in_new_bottom_2',
        'won_duel',
        'duel_win_prob'
    ]
    
    # Calc Fan Rank
    merged['est_fan_rank'] = merged.groupby(['season', 'week'])['est_fan_pct'].rank(ascending=False, method='min')
    
    # Ensure columns exist
    final_cols = []
    for c in output_cols:
        if c in merged.columns:
            final_cols.append(c)
        else: 
            # fan rank needs to be added to list? It is calculated but I didn't add to output_cols list properly in definition
            pass
            
    final_cols.insert(7, 'est_fan_rank') # Insert after est_fan_pct
    
    # Select subset
    final_df = merged[[c for c in final_cols if c in merged.columns]]
    
    output_filename = 'Q4_New_System_Analysis.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"Analysis complete. Saved to {output_filename}")
    
    # Summary of Changes
    # Look at weeks where someone was eliminated
    elim_weeks = merged[merged['is_eliminated'] == 1]
    
    total_eliminations = len(elim_weeks)
    saved_by_rank = 0
    saved_by_duel = 0
    
    print("\n=== Analysis of Potential Altered Outcomes ===")
    for idx, row in elim_weeks.iterrows():
        # Scenario 1: Not in Bottom 2 at all (Saved by Ranking)
        if not row['in_new_bottom_2']:
            saved_by_rank += 1
            print(f"S{row['season']} W{row['week']}: {row['name']} - Rank {row['new_rank']:.0f}/{row['contestant_count']:.0f} - SAFE (Ranking)")
        
        # Scenario 2: In Bottom 2, but Wins Duel (Saved by Duel)
        elif row['in_new_bottom_2'] and row['won_duel']:
            saved_by_duel += 1
            print(f"S{row['season']} W{row['week']}: {row['name']} - Rank {row['new_rank']:.0f}/{row['contestant_count']:.0f} - SAVED (Duel Win {row['duel_win_prob']:.1%})")
            
    total_saved = saved_by_rank + saved_by_duel
    
    print(f"\nTotal Eliminations Analyzed: {total_eliminations}")
    print(f"Saved by New Ranking System (Safe): {saved_by_rank} ({saved_by_rank/total_eliminations:.2%})")
    print(f"Saved by Sudden Death Duel (Win):   {saved_by_duel} ({saved_by_duel/total_eliminations:.2%})")
    print(f"Total 'Unjust' Eliminations Prevented: {total_saved} ({total_saved/total_eliminations:.2%})")

if __name__ == "__main__":
    calculate_new_scores()
