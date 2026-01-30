# MCM2026

2026美赛共享库，包含 latex 模板，MCM 题面。我们选择的是 C 题。

## 数据清洗

在 `clean_data.py` 中实现清洗数据功能。对出身美国的名人，将其出生地标准化为州名。对其他国家或地区的名人，统一沿用原始数据。我们统计了每位名人每周的平均评委得分，按照 season-week-avg_score 的顺序排序，保存为 `cleaned_weekly_avg.csv` 文件。