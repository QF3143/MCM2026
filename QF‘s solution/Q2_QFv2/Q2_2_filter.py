import pandas as pd
import numpy as np

def calculate_weighted_controversy(file_path):
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # Helper to parse elimination week
    def get_elim_week(res):
        if pd.isna(res): return 99
        res = str(res).lower()
        if "eliminated week" in res:
            try: return int(res.split("week")[-1].strip())
            except: return -1
        if any(x in res for x in ["place", "winner", "runner"]): return 99
        return -1

    records = []
    
    for season in df['season'].unique():
        season_df = df[df['season'] == season].copy()
        season_df['elim_week'] = season_df['results'].apply(get_elim_week)
        
        # Iterate Weeks
        for w in range(1, 13):
            cols = [c for c in season_df.columns if f'week{w}_judge' in c]
            if not cols: continue
            
            active = season_df[season_df['elim_week'] >= w].copy()
            if len(active) < 2: continue
            
            # Scores & Z-Score
            active['week_score'] = active[cols].sum(axis=1)
            active = active[active['week_score'] > 0]
            if active.empty: continue
            
            mu, sigma = active['week_score'].mean(), active['week_score'].std()
            if sigma == 0: sigma = 1
            active['z_score'] = (active['week_score'] - mu) / sigma
            
            # Threshold (Mean of Eliminated)
            eliminated = active[active['elim_week'] == w]
            survived = active[active['elim_week'] > w]
            
            if eliminated.empty: continue
            threshold_z = eliminated['z_score'].mean()
            
            # Calculate Weighted Gap
            for _, row in survived.iterrows():
                if row['z_score'] < threshold_z:
                    gap = threshold_z - row['z_score']
                    
                    # [WEIGHTING LOGIC]
                    weight = w*w # Linearly increasing weight
                    weighted_gap = gap * weight
                    
                    records.append({
                        'celebrity_name': row['celebrity_name'],
                        'season': season,
                        'week': w,
                        'raw_gap': gap,
                        'weighted_gap': weighted_gap
                    })
                    
    # Aggregate
    res_df = pd.DataFrame(records)
    ranking = res_df.groupby(['celebrity_name', 'season']).agg(
        Weighted_Index=('weighted_gap', 'sum'),
        Raw_Index=('raw_gap', 'sum'),
        Count=('week', 'count')
    ).reset_index().sort_values('Weighted_Index', ascending=False)
    
    return ranking

# Execution
df_result = calculate_weighted_controversy('2026_MCM_Problem_C_Data.csv')
df_result.to_csv('controversal_name_list_weighted.csv', index=False)