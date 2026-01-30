import pandas as pd
import numpy as np
import re

df = pd.read_csv('2026_MCM_Problem_C_Data.csv')

# ========================================================
# 1. 数据清洗
# ========================================================

# 将 USA 直接替换为该选手所在的州
df['celebrity_homestate/homecountry/region'] = np.where(df['celebrity_homecountry/region'] == 'United States', df['celebrity_homestate'], df['celebrity_homecountry/region'])

# 统计选手的周评分
score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
df_long = df.melt(id_vars=['celebrity_name', 'season', 'placement'],
                  value_vars=score_cols,
                  var_name='judge_info',
                  value_name='score')

df_long[['week', 'judge_num']] = df_long['judge_info'].str.extract(r'week(\d+)_judge(\d+)')
df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')
df_long = df_long.dropna(subset=['score'])

weekly_avg = df_long.groupby(['celebrity_name', 'season', 'week'])['score'].mean().reset_index()
weekly_avg.rename(columns={'score': 'average_score'}, inplace=True)
weekly_avg = weekly_avg.sort_values(by=['season', 'week', 'average_score'], ascending=[True, True, False]).reset_index(drop=True)
print(weekly_avg.head(6))

weekly_avg.to_csv('cleaned_weekly_avg.csv', index=False, encoding='utf-8-sig')