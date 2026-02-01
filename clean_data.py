import pandas as pd
import numpy as np
import re
import os

def clean_dwts_data(file_path):
    print(f"Processing {file_path}...")
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # 2. Identify Score Columns
    score_cols = [c for c in df.columns if 'judge' in c.lower() and 'score' in c.lower()]
    
    # 3. Extract Info
    cleaned_rows = []
    
    for index, row in df.iterrows():
        name = row['celebrity_name']
        season = row['season']
        # Placement/Result
        result_str = str(row['results']) 
        placement = str(row['placement'])
        
        # Parse Elimination Week
        elim_week = 99
        is_withdrew = 'Withdrew' in result_str
        
        # Try to parse "Eliminated Week X"
        match = re.search(r'Week\s*(\d+)', result_str, re.IGNORECASE)
        if match:
            elim_week = int(match.group(1))
        
        # Handle "Withdrew" logic - find last active week
        last_active_week = 0
        if is_withdrew:
            for w in range(1, 15):
                # Check if this week's columns exist and have data
                week_scores = []
                for c in score_cols:
                    if f'week{w}_' in c.lower():
                        val = row[c]
                        if pd.notnull(val) and isinstance(val, (int, float)) and val > 0:
                            week_scores.append(val)
                if week_scores:
                    last_active_week = w
        
        # Iterate weeks
        prev_score = None # for tracking improvement
        
        for w in range(1, 15): # Support up to 14 weeks
             # Find columns for this week
            current_week_cols = [c for c in score_cols if f'week{w}_' in c.lower()]
            if not current_week_cols: 
                continue
            
            # Extract scores
            scores = []
            for c in current_week_cols:
                val = row[c]
                if pd.notnull(val):
                    try:
                        scores.append(float(val))
                    except:
                        pass
            
            if not scores:
                continue
            
            avg_score = sum(scores) / len(scores)
            
            # Filter near-zero averages
            if avg_score < 0.1:
                continue
            
            # Determine Status
            status = "In"
            if is_withdrew:
                if w == last_active_week:
                    status = "Withdrew" # Active this week, but leaves after
                elif w > last_active_week:
                    status = "Out"
            else:
                if w == elim_week:
                    status = "Eliminated"
                elif w > elim_week:
                    status = "Out"
            
            # Skip rows where contestant is already out (and no scores, but we checked scores > 0)
            if status == "Out":
                continue
                
            cleaned_rows.append({
                'season': season,
                'week': w,
                'name': name,
                'avg_score': avg_score,
                'status': status,
                'placement': placement,
                'is_eliminated': 1 if status == "Eliminated" else 0,
                'is_withdrew': 1 if status == "Withdrew" else 0
            })

    cleaned_df = pd.DataFrame(cleaned_rows)
    
    # 4. Calculate Derived Metrics (Per Season-Week)
    
    # Judge Percentage Share = (Score / Sum of all active scores in that week)
    cleaned_df['judge_total_weekly'] = cleaned_df.groupby(['season', 'week'])['avg_score'].transform('sum')
    cleaned_df['judge_pct'] = cleaned_df['avg_score'] / cleaned_df['judge_total_weekly']
    
    # Judge Rank (1 is best)
    # dense: 1, 2, 2, 3
    cleaned_df['judge_rank'] = cleaned_df.groupby(['season', 'week'])['avg_score'].rank(ascending=False, method='min')
    
    # Previous Week Score (Lag)
    cleaned_df = cleaned_df.sort_values(['season', 'name', 'week'])
    cleaned_df['prev_avg_score'] = cleaned_df.groupby(['season', 'name'])['avg_score'].shift(1)
    cleaned_df['score_improvement'] = cleaned_df['avg_score'] - cleaned_df['prev_avg_score']
    cleaned_df['score_improvement'] = cleaned_df['score_improvement'].fillna(0)
    
    # Determine the "Most Improved" bonus for the week
    # We find the MAX improvement in the week.
    def get_bonus_flag(x):
        max_imp = x['score_improvement'].max()
        # If max improvement is <= 0, we assume no bonus for fairness, or maybe everyone is 0
        if max_imp <= 0:
            return pd.Series([0]*len(x), index=x.index)
        # Handle ties: all get bonus
        return (x['score_improvement'] == max_imp).astype(int)

    # Need to group by Season AND Week
    # We use a trick to apply this group-wise
    # But since apply can be slow or tricky with indices, we can join back.
    
    # Calculate max improvement per week
    weekly_max = cleaned_df.groupby(['season', 'week'])['score_improvement'].max().reset_index().rename(columns={'score_improvement': 'max_imp'})
    cleaned_df = pd.merge(cleaned_df, weekly_max, on=['season', 'week'], how='left')
    
    cleaned_df['is_most_improved'] = 0
    # Condition: positive improvement AND equals max
    mask = (cleaned_df['score_improvement'] > 0) & (cleaned_df['score_improvement'] == cleaned_df['max_imp'])
    cleaned_df.loc[mask, 'is_most_improved'] = 1
    
    return cleaned_df

if __name__ == "__main__":
    df_final = clean_dwts_data('2026_MCM_Problem_C_Data.csv')
    df_final = df_final.sort_values(['season', 'week', 'avg_score'], ascending=[True, True, False])
    
    output_path = 'cleaned_weekly_data.csv'
    df_final.to_csv(output_path, index=False)
    print(f"Data cleaned and saved to {output_path}")
    print(df_final.head())
