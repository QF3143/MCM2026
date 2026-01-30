import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from Q1_Estimator import evaluate_and_save
from Q1_Basic_Model import estimate_season_datadriven

def automethod(season):
    rank_seasons = {1, 2, 21, 28, 29, 30, 32, 33, 34}
    return "rank" if season in rank_seasons else "percent"

if __name__ == "__main__":
    for season in range(1,35):
        
        m = automethod(season)
        INPUT_PATH = f'/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/Q1_result{season}_{m}_datadriven.csv'
        OUTPUT_PATH = f"/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/Q1_result{season}_{m}_estimator.json"
        
        estimate_season_datadriven(season, '2026_MCM_Problem_C_Data.csv', f'/Users/liuqiufan/Documents/SJTU_Local/MCM2026/QF‘s solution/Q1_result{season}_{m}_datadriven.csv', method=m)
        results = evaluate_and_save(
        input_csv_path=INPUT_PATH,
        output_json_path=OUTPUT_PATH,
        season_id= season ,
        n_global_runs=20
        )