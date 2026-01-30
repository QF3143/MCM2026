import pandas as pd
import re

def clean_dwts_data(file_path):
    # 1. 加载数据
    df = pd.read_csv(file_path)
    
    # 2. 确定所有的周次 (假设列名包含 week1, week2...)
    # 找出所有包含 'judge' 和 'score' 的列
    score_cols = [c for c in df.columns if 'judge' in c.lower() and 'score' in c.lower()]
    
    # 3. 提取赛季、选手名、周次、评分
    cleaned_rows = []
    
    for index, row in df.iterrows():
        name = row['celebrity_name']
        season = row['season']
        result_str = str(row['results']) # 例如 "Eliminated Week 3" 或 "Runner-up"
        
        # 解析该选手到底在哪一周被淘汰的
        elim_week = 99 # 默认没淘汰（进入决赛）
        match = re.search(r'Week\s*(\d+)', result_str)
        if 'Eliminated' in result_str and match:
            elim_week = int(match.group(1))
            
        # 遍历每一周提取分数
        for w in range(1, 12): # 假设最多11周
            # 找到当前周的所有评委得分列
            current_week_cols = [c for c in score_cols if f'week{w}' in c.lower()]
            if not current_week_cols: continue
            
            # 计算这一周的平均分
            scores = [row[c] for c in current_week_cols if pd.notnull(row[c])]
            
            if len(scores) > 0:
                avg_score = sum(scores) / len(scores)
                # 标记该选手在本周的状态
                status = "In"
                if w == elim_week:
                    status = "Eliminated"
                elif w > elim_week:
                    status = "Out"
                
                cleaned_rows.append({
                    'season': season,
                    'week': w,
                    'name': name,
                    'avg_score': avg_score,
                    'status': status
                })
    
    cleaned_df = pd.DataFrame(cleaned_rows)
    
    # 4. 关键步骤：计算每一周的分数占比 (Percentage Share)
    # 这是第一问算法的直接输入
    cleaned_df['judge_pct'] = cleaned_df.groupby(['season', 'week'])['avg_score'].transform(lambda x: x / x.sum())
    
    # 5. 计算每一周的排名 (Rank Share)
    # 这是第二问对比算法的输入
    cleaned_df['judge_rank'] = cleaned_df.groupby(['season', 'week'])['avg_score'].rank(ascending=False)
    
    return cleaned_df

# 使用方法
df_final = clean_dwts_data('2026_MCM_Problem_C_Data.csv')
df_final.to_csv('cleaned_weekly_data.csv', index=False)