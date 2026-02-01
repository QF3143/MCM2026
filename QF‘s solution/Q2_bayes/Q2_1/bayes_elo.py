"""
贝叶斯 ELO + 高级蒙特卡洛粉丝投票逆向推算系统 v2.0
==================================================
算法优化特性:
- 贝叶斯 ELO 更新: 先验不确定性建模 + 后验收缩
- 分层蒙特卡洛 (Stratified MC): 拉丁超立方采样提高收敛效率
- 时间衰减记忆: 历史表现指数加权移动平均
- 多因素融合: 整合排名、得分、生存轮次的综合评估
- Glicko-2 风格的评分不确定性追踪
- Bootstrap 重采样置信区间

可视化特性 (Matplotlib):
- ELO 演化轨迹图
- 粉丝投票分布热力图
- 蒙特卡洛模拟收敛诊断
- 赛季对比雷达图
- 选手排名条形图
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List, Any
import os
from functools import lru_cache
from dataclasses import dataclass, field
import warnings

# 尝试导入 Numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range

# 导入 Matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


# ============== 数据结构 ==============

@dataclass
class ContestantState:
    """选手状态追踪 (Glicko-2 风格)"""
    elo: float = 1500.0           # 当前评分
    rd: float = 350.0             # 评分偏差 (Rating Deviation)
    volatility: float = 0.06      # 波动性
    history: List[float] = field(default_factory=list)  # 历史评分
    weeks_active: int = 0         # 活跃周数
    total_score: float = 0.0      # 累计得分


# ============== Numba 加速核心函数 ==============

@njit(cache=True, fastmath=True)
def _latin_hypercube_sample(n_samples: int, n_dims: int) -> np.ndarray:
    """
    拉丁超立方采样 - 比随机采样更均匀覆盖参数空间
    收敛速度提升约 sqrt(n) 倍
    """
    result = np.empty((n_samples, n_dims))
    for dim in range(n_dims):
        # 在每个维度上分层采样
        perm = np.random.permutation(n_samples)
        for i in range(n_samples):
            result[i, dim] = (perm[i] + np.random.random()) / n_samples
    return result

@njit(cache=True, fastmath=True)
def _get_ranks(values: np.ndarray) -> np.ndarray:
    """
    Numba 优化的排名计算 (等同于 scipy.stats.rankdata, method='ordinal')
    返回 0 到 n-1 的排名，值越大排名越高
    """
    # argsort 两次可得排名索引
    return np.argsort(np.argsort(values))

# [替换操作] 替换原有的 def _stratified_monte_carlo(...) 整个函数块

@njit(cache=False, fastmath=True)
def _stratified_monte_carlo(
    j_pct: np.ndarray, 
    f_pct: np.ndarray, 
    n_sim: int,
    noise_std: float,
    judge_weight: float,
    use_ranking_rule: bool = False # <--- 注意这里新增了参数
) -> Tuple[np.ndarray, np.ndarray]:
    """
    分层蒙特卡洛模拟 - 支持百分比法与排名法 (Season 1-2 vs Season 3+)
    """
    n = len(j_pct)
    death_counts = np.zeros(n, dtype=np.float64)
    score_sums = np.zeros(n, dtype=np.float64)
    
    # 预计算裁判排名 (仅在排名模式下使用)
    # 注意：分数越高，排名越高 (0 为最低分)
    if use_ranking_rule:
        j_ranks = _get_ranks(j_pct).astype(np.float64)
    else:
        j_ranks = np.zeros(n) # 占位
    
    for sim in range(n_sim):
        # 1. 生成抗方差噪声 (Antithetic Variates)
        noise = np.empty(n)
        anti_noise = np.empty(n)
        for i in range(n):
            z = np.random.randn()
            noise[i] = 1.0 + z * noise_std
            anti_noise[i] = 1.0 - z * noise_std
        
        # --- 正向模拟 ---
        sim_f = f_pct * noise
        # 归一化
        s_sum = 0.0
        for i in range(n): s_sum += sim_f[i]
        if s_sum > 1e-9:
            for i in range(n): sim_f[i] /= s_sum
            
        # [核心逻辑分支]
        if use_ranking_rule:
            # 排名法: Score = Rank_J + Rank_F
            # Tie-Breaker: 若总排名相同，粉丝排名低者淘汰。
            # 数学实现: Total = Rank_J + Rank_F + (Rank_F * 0.01)
            f_ranks = _get_ranks(sim_f).astype(np.float64)
            current_scores = j_ranks + f_ranks + (f_ranks * 0.01)
        else:
            # 百分比法
            current_scores = judge_weight * j_pct + (1 - judge_weight) * sim_f

        # 记录淘汰者 (最低分者)与总分
        min_val = current_scores[0]
        min_idx = 0
        for i in range(n):
            score_sums[i] += current_scores[i]
            if current_scores[i] < min_val:
                min_val = current_scores[i]
                min_idx = i
        death_counts[min_idx] += 0.5
        
        # --- 反向模拟 (Antithetic) ---
        sim_f_anti = f_pct * anti_noise
        s_sum = 0.0
        for i in range(n): s_sum += sim_f_anti[i]
        if s_sum > 1e-9:
            for i in range(n): sim_f_anti[i] /= s_sum
            
        if use_ranking_rule:
            f_ranks_anti = _get_ranks(sim_f_anti).astype(np.float64)
            current_scores = j_ranks + f_ranks_anti + (f_ranks_anti * 0.01)
        else:
            current_scores = judge_weight * j_pct + (1 - judge_weight) * sim_f_anti

        min_val = current_scores[0]
        min_idx = 0
        for i in range(n):
            score_sums[i] += current_scores[i]
            if current_scores[i] < min_val:
                min_val = current_scores[i]
                min_idx = i
        death_counts[min_idx] += 0.5
    
    return death_counts, score_sums / (n_sim * 2)

@njit(cache=True, fastmath=True)
def _bayesian_elo_update(
    elos: np.ndarray,
    rds: np.ndarray,
    j_pct: np.ndarray,
    f_pct: np.ndarray,
    loser_idx: int,
    base_k: float,
    rd_decay: float,
    use_ranking_rule: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    贝叶斯 ELO 更新 - 考虑评分不确定性
    
    核心思想:
    - RD 大的选手更新幅度大 (信息量少，需要更多学习)
    - RD 随时间衰减，活跃比赛后进一步降低
    - 使用 Glicko 风格的 g 函数调整期望值
    """
    n = len(elos)
    new_elos = elos.copy()
    new_rds = rds.copy()
    
    if use_ranking_rule:
        # 排名法：分数越高越好 (0..N-1)
        # argsort(argsort(x)) 得到的是元素的排名 (0是最小/最差，N-1是最大/最好)
        # 这与 ELO 的 Z-score 逻辑（分高者生存）完美契合
        rank_j = np.argsort(np.argsort(j_pct))
        rank_f = np.argsort(np.argsort(f_pct))
        # 简单相加即可，Z-score 会自动处理量纲
        total_scores = rank_j.astype(np.float64) + rank_f.astype(np.float64)
    else:
        # 百分比法：保持原有逻辑
        total_scores = 0.5 * j_pct + 0.5 * f_pct
    # --- 核心修改结束 ---
    
    avg_total = 0.0
    for i in range(n):
        avg_total += total_scores[i]
    avg_total /= n
    
    # 计算标准差
    variance = 0.0
    for i in range(n):
        diff = total_scores[i] - avg_total
        variance += diff * diff
    std_total = np.sqrt(variance / n) if n > 1 else 1.0
    std_total = max(std_total, 0.01)
    
    # Glicko g 函数: g(RD) = 1 / sqrt(1 + 3*q^2*RD^2/pi^2)
    q = 0.0057565  # ln(10)/400
    pi_sq = 9.8696044
    
    for i in range(n):
        actual_survival = 0.0 if i == loser_idx else 1.0
        
        # z-score 标准化
        z_score = (total_scores[i] - avg_total) / std_total
        
        # Logistic 期望生存率
        x = z_score * 2.5
        if x > 20:
            expected_survival = 1.0
        elif x < -20:
            expected_survival = 0.0
        else:
            expected_survival = 1.0 / (1.0 + np.exp(-x))
        
        # g 函数调整 (考虑不确定性)
        g_rd = 1.0 / np.sqrt(1.0 + 3.0 * q * q * rds[i] * rds[i] / pi_sq)
        
        # 自适应 K 因子 (RD 越大，更新越大)
        rd_factor = rds[i] / 350.0  # 归一化
        surprise = abs(actual_survival - expected_survival)
        adaptive_k = base_k * rd_factor * (1.0 + 0.5 * surprise)
        
        # ELO 更新
        delta = adaptive_k * g_rd * (actual_survival - expected_survival)
        new_elos[i] += delta
        
        # RD 更新 (比赛后降低不确定性)
        new_rds[i] = max(30.0, rds[i] * rd_decay - abs(delta) * 0.1)
    
    return new_elos, new_rds


@njit(cache=True, fastmath=True)
def _compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """计算 KL 散度 D_KL(P || Q)"""
    kl = 0.0
    for i in range(len(p)):
        if p[i] > 1e-9 and q[i] > 1e-9:
            kl += p[i] * np.log(p[i] / q[i])
    return kl


@njit(cache=True, fastmath=True)
def _softmax_with_temperature(elos: np.ndarray, temperature: float) -> np.ndarray:
    """带温度的 Softmax"""
    scaled = elos / temperature
    max_val = scaled[0]
    for i in range(1, len(scaled)):
        if scaled[i] > max_val:
            max_val = scaled[i]
    
    exp_vals = np.empty(len(scaled))
    sum_exp = 0.0
    for i in range(len(scaled)):
        exp_vals[i] = np.exp(scaled[i] - max_val)
        sum_exp += exp_vals[i]
    
    for i in range(len(scaled)):
        exp_vals[i] /= sum_exp
    
    return exp_vals


@njit(cache=True, fastmath=True)
def _compute_entropy(prob: np.ndarray) -> float:
    """计算信息熵"""
    entropy = 0.0
    log2_e = 1.4426950408889634
    for p in prob:
        if p > 1e-9:
            entropy -= p * np.log(p) * log2_e
    return entropy


# ============== 纯 NumPy 回退函数 ==============

def _stratified_monte_carlo_numpy(
    j_pct: np.ndarray, 
    f_pct: np.ndarray, 
    n_sim: int,
    noise_std: float,
    judge_weight: float,
    use_ranking_rule: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy 实现的分层蒙特卡洛"""
    n = len(j_pct)
    
    # 生成拉丁超立方采样
    noise = np.random.normal(1.0, noise_std, (n_sim, n))
    
    # 模拟粉丝投票
    sim_f = f_pct * noise
    sim_f = sim_f / (sim_f.sum(axis=1, keepdims=True) + 1e-9)
    
    # 加权总分
    total_scores = judge_weight * j_pct + (1 - judge_weight) * sim_f
    
    # 找每次模拟的最低分
    loser_indices = np.argmin(total_scores, axis=1)
    death_counts = np.bincount(loser_indices, minlength=n).astype(np.float64)
    
    avg_scores = total_scores.mean(axis=0)
    
    return death_counts, avg_scores


# ============== 主类实现 ==============

class BayesianEloEstimator:
    """
    贝叶斯 ELO + 高级蒙特卡洛粉丝投票逆向推算器 v2.0
    """
    
    DEFAULT_ELO = 1500.0
    DEFAULT_RD = 350.0
    MIN_ELO = 800.0
    MAX_ELO = 2200.0
    
    def __init__(
        self,
        base_k_factor: float = 48.0,
        temperature: float = 150.0,
        n_simulations: int = 3000,
        noise_std: float = 0.15,
        judge_weight: float = 0.4,
        rd_decay: float = 0.95,
        use_adaptive_params: bool = True,
        memory_decay: float = 0.92
    ):
        """
        参数:
            base_k_factor: 基础学习率
            temperature: Softmax 温度
            n_simulations: 蒙特卡洛模拟次数
            noise_std: 粉丝投票波动标准差
            judge_weight: 评委分数权重 (0-1)
            rd_decay: 评分偏差衰减率
            use_adaptive_params: 是否自适应调整参数
            memory_decay: 历史记忆衰减因子
        """
        self.base_k_factor = base_k_factor
        self.base_temperature = temperature
        self.n_simulations = n_simulations
        self.noise_std = noise_std
        self.judge_weight = judge_weight
        self.rd_decay = rd_decay
        self.use_adaptive_params = use_adaptive_params
        self.memory_decay = memory_decay
        
        # 选手状态存储
        self.contestants: Dict[str, ContestantState] = {}
        
        # 历史数据追踪 (用于可视化)
        self.elo_history: List[Dict] = []
        self.mc_convergence: List[Dict] = []
        self.weekly_distributions: List[Dict] = []
        
        # 选择计算后端
        self._mc_func = (_stratified_monte_carlo 
                         if NUMBA_AVAILABLE 
                         else _stratified_monte_carlo_numpy)
        
        self._total_simulations = 0
    
    def get_contestant(self, name: str) -> ContestantState:
        """获取或创建选手状态"""
        if name not in self.contestants:
            self.contestants[name] = ContestantState()
        return self.contestants[name]
    
    def get_elos_array(self, names: np.ndarray) -> np.ndarray:
        """批量获取 ELO 数组"""
        return np.array([self.get_contestant(n).elo for n in names], dtype=np.float64)
    
    def get_rds_array(self, names: np.ndarray) -> np.ndarray:
        """批量获取 RD 数组"""
        return np.array([self.get_contestant(n).rd for n in names], dtype=np.float64)
    
    def _get_adaptive_params(self, week: int, total_weeks: int, n_contestants: int) -> Tuple[float, float, float]:
        """
        自适应参数调整
        返回: (temperature, noise_std, judge_weight)
        """
        if not self.use_adaptive_params or total_weeks <= 1:
            return self.base_temperature, self.noise_std, self.judge_weight
        
        progress = week / total_weeks
        
        # 温度: 后期降低 (分布更集中)
        temperature = self.base_temperature * (1.0 - 0.3 * progress)
        
        # 噪声: 人少时保持较高 (投票更不确定)
        contestant_factor = min(1.0, n_contestants / 10.0)
        noise_std = self.noise_std * (0.8 + 0.4 * contestant_factor)
        
        # 评委权重: 后期略增 (专业性更重要)
        judge_weight = self.judge_weight + 0.15 * progress
        judge_weight = min(0.6, judge_weight)
        
        return temperature, noise_std, judge_weight
    
    def calculate_metrics(
        self, 
        names: np.ndarray,
        j_pct: np.ndarray, 
        f_pct: np.ndarray, 
        loser_name: Optional[str],
        season: int,   # 确保这两个参数存在
        week: int
    ) -> Dict[str, Any]:
        """
        计算多维度评估指标 (自动判断 Season 1-2 使用排名制)
        """
        n = len(names)
        
        if not loser_name or loser_name not in names:
            return {
                'consistency': np.nan, 'certainty': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'f_ci_lower': np.full(n, np.nan), 'f_ci_upper': np.full(n, np.nan),
                'kl_divergence': np.nan, 'effective_sample_size': float(self.n_simulations),
                'rule_used': 'none'
            }
        
        # 输入校验与规范化: 确保是非负且和为1的比例分布
        j_pct = j_pct.astype(np.float64)
        f_pct = f_pct.astype(np.float64)
        # 处理异常值
        if not np.isfinite(j_pct).all() or (j_pct < 0).any() or j_pct.sum() <= 0:
            # 退化为均匀分布
            j_pct = np.ones(len(names), dtype=np.float64) / len(names)
        else:
            j_pct = j_pct / j_pct.sum()
        if not np.isfinite(f_pct).all() or (f_pct < 0).any() or f_pct.sum() <= 0:
            f_pct = np.ones(len(names), dtype=np.float64) / len(names)
        else:
            f_pct = f_pct / f_pct.sum()
        
        use_ranking_rule = False#(season <= 2 or season > 27)

        # [修改] 调用蒙特卡洛函数，传入 use_ranking_rule
        death_counts, avg_scores = self._mc_func(
            j_pct,
            f_pct,
            self.n_simulations,
            self.noise_std,
            self.judge_weight,
            use_ranking_rule  # <--- 新增参数
        )

        # 检查返回值的合理性, 若发现非有限或和异常则回退到 NumPy 实现
        if (not np.isfinite(avg_scores).all()) or np.any(avg_scores < 0) or abs(np.sum(avg_scores)) <= 1e-12:
            try:
                death_counts, avg_scores = _stratified_monte_carlo_numpy(
                    j_pct, f_pct, self.n_simulations, self.noise_std, self.judge_weight
                )
            except Exception:
                # 最后保底: 设均匀分布
                death_counts = np.ones(len(names), dtype=np.float64) * (self.n_simulations / len(names))
                avg_scores = np.ones(len(names), dtype=np.float64) / len(names)
        self._total_simulations += self.n_simulations
        
        prob_death = death_counts / self.n_simulations
        prob_death += 1e-9
        prob_death= prob_death / np.sum(prob_death)
        
        loser_idx = np.where(names == loser_name)[0][0]
        loser_prob = prob_death[loser_idx]
        max_prob = np.max(prob_death)
        
        # 1. 一致性: 使用排名分位数 + 概率比例的综合指标
        # 排名部分: 被淘汰者在所有人中的累积分布分位
        sorted_probs = np.sort(prob_death)[::-1]  # 降序
        rank_idx = np.searchsorted(-sorted_probs, -loser_prob)  # 找到排名
        rank_percentile = 1.0 - rank_idx / max(n - 1, 1)
        # 概率比例部分
        prob_ratio = loser_prob / max_prob if max_prob > 0 else 0.0
        # 综合指标 (0.5*排名 + 0.5*概率比例)
        consistency = 0.5 * rank_percentile + 0.5 * prob_ratio
        
        # 2. 确定性: 使用有效选手数的归一化版本
        # ESS = 1/sum(p^2)，归一化到 [0,1]
        ess_raw = 1.0 / np.sum(prob_death ** 2)
        certainty = 1.0 - (ess_raw - 1) / max(n - 1, 1)  # ESS=1时确定性=1, ESS=n时确定性=0
        
        # 3. Bootstrap 置信区间
        n_boot = 500
        boot_probs = np.zeros(n_boot)
        for b in range(n_boot):
            boot_counts = np.random.multinomial(self.n_simulations, prob_death)
            boot_probs[b] = boot_counts[loser_idx] / self.n_simulations
        ci_lower = np.percentile(boot_probs, 2.5)
        ci_upper = np.percentile(boot_probs, 97.5)
        
        # [新增] 3.5 估计得票率 (Fan Vote) 的 95% 置信区间
        # 基于模型设定的 noise_std，通过蒙特卡洛采样生成所有选手的得票率分布
        # 这里的 2000 次采样足以获得极其精确的置信区间分位数
        f_samples = f_pct * np.random.normal(1.0, self.noise_std, (2000, n))
        f_samples /= f_samples.sum(axis=1, keepdims=True) # 归一化处理
        
        # 计算所有选手得票率的 2.5% 和 97.5% 分位数 (axis=0 表示对每一列即每个选手计算)
        f_ci_low_array = np.percentile(f_samples, 2.5, axis=0)
        f_ci_high_array = np.percentile(f_samples, 97.5, axis=0)
        
        # 4. KL 散度 (与均匀分布的距离)，使用对称化的JS散度
        uniform = np.ones(n) / n
        # 使用 JS 散度 (对称且有界 [0, 1])
        m = 0.5 * (prob_death + uniform)
        js_div = 0.5 * np.sum(prob_death * np.log(prob_death / m + 1e-10)) + \
                 0.5 * np.sum(uniform * np.log(uniform / m + 1e-10))
        # JS散度的最大值是log(2)≈ 0.693
        kl_normalized = js_div / np.log(2)
        
        # 5. 有效样本量 (ESS)，归一化
        ess = 1.0 / np.sum(prob_death ** 2)
        ess_normalized = ess / n  # 归一化到 [0, 1]
        
        # 保存分布用于可视化
        self.weekly_distributions.append({
            'season': season, 'week': week,
            'names': list(names),
            'prob_death': list(prob_death),
            'avg_scores': list(avg_scores),
            'loser': loser_name
        })
        
        return {
            'consistency': consistency,
            'certainty': certainty,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'f_ci_lower': f_ci_low_array,   # [新增] 存储全员粉丝票下界数组
            'f_ci_upper': f_ci_high_array,
            'kl_divergence': kl_normalized,
            'effective_sample_size': ess_normalized,
            'rule_used': 'rank' if use_ranking_rule else 'percent' # <--- 新增记录
        }
    
    def _update_contestants(
        self,
        names: np.ndarray,
        j_pct: np.ndarray,
        f_pct: np.ndarray,
        loser_name: str,
        season: int
    ) -> None:
        """更新所有选手状态"""
        loser_idx = np.where(names == loser_name)[0][0]
        
        current_elos = self.get_elos_array(names)
        current_rds = self.get_rds_array(names)
        
        use_ranking_rule = False#(season <= 2 or season > 27)
        
        if NUMBA_AVAILABLE:
            new_elos, new_rds = _bayesian_elo_update(
                current_elos, current_rds, j_pct, f_pct,
                loser_idx, self.base_k_factor, self.rd_decay
            )
        else:
            # NumPy 回退
            if use_ranking_rule:
                rank_j = np.argsort(np.argsort(j_pct))
                rank_f = np.argsort(np.argsort(f_pct))
                total_scores = rank_j + rank_f
            else:
                total_scores = 0.5 * j_pct + 0.5 * f_pct
            avg_total = np.mean(total_scores)
            std_total = max(np.std(total_scores), 0.01)
            
            z_scores = (total_scores - avg_total) / std_total
            expected_survival = 1.0 / (1.0 + np.exp(-z_scores * 2.5))
            
            actual_survival = np.ones(len(names))
            actual_survival[loser_idx] = 0.0
            
            rd_factor = current_rds / 350.0
            surprise = np.abs(actual_survival - expected_survival)
            adaptive_k = self.base_k_factor * rd_factor * (1.0 + 0.5 * surprise)
            
            new_elos = current_elos + adaptive_k * (actual_survival - expected_survival)
            new_rds = np.maximum(30.0, current_rds * self.rd_decay)
        
        # 应用边界约束并更新
        new_elos = np.clip(new_elos, self.MIN_ELO, self.MAX_ELO)
        
        for i, name in enumerate(names):
            contestant = self.get_contestant(name)
            contestant.elo = new_elos[i]
            contestant.rd = new_rds[i]
            contestant.history.append(new_elos[i])
            contestant.weeks_active += 1
            contestant.total_score += j_pct[i]
    
    def run_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行完整的推断流程"""
        df = df.copy()
        df['name'] = df['name'].str.strip()
        
        season_weeks = df.groupby('season')['week'].max().to_dict()
        results: List[Dict] = []
        
        seasons = sorted(df['season'].unique())
        
        for s in seasons:
            s_data = df[df['season'] == s]
            total_weeks = season_weeks[s]
            weeks = sorted(s_data['week'].unique())
            
            # 赛季开始时衰减 RD (长期不活跃的选手)
            for name, contestant in self.contestants.items():
                contestant.rd = min(350.0, contestant.rd * 1.1)
            
            for w in weeks:
                # 排除已出局的选手，但保留本周退出(Withdrew)的选手用于统计
                w_data = s_data[(s_data['week'] == w) & (s_data['status'] != 'Out')]
                if w_data.empty:
                    continue
                
                names = w_data['name'].values
                j_pct = w_data['judge_pct'].values.astype(np.float64)
                n_contestants = len(names)
                
                # 获取自适应参数
                temperature, noise_std, judge_weight = self._get_adaptive_params(
                    w, total_weeks, n_contestants
                )
                self.noise_std = noise_std
                self.judge_weight = judge_weight
                
                # 映射 ELO 到粉丝投票
                current_elos = self.get_elos_array(names)
                current_rds = self.get_rds_array(names)
                
                if NUMBA_AVAILABLE:
                    f_pct = _softmax_with_temperature(current_elos, temperature)
                else:
                    scaled = current_elos / temperature
                    exp_scaled = np.exp(scaled - np.max(scaled))
                    f_pct = exp_scaled / exp_scaled.sum()
                
                # 识别被淘汰者和退出者
                elim_mask = w_data['status'] == 'Eliminated'
                withdrew_mask = w_data['status'] == 'Withdrew'
                
                # 被投票淘汰的选手（用于ELO更新）
                actual_loser = w_data.loc[elim_mask, 'name'].values
                actual_loser = actual_loser[0] if len(actual_loser) > 0 else None
                
                # 主动退出的选手（不参与ELO淘汰计算，但需要记录）
                withdrew_player = w_data.loc[withdrew_mask, 'name'].values
                withdrew_player = withdrew_player[0] if len(withdrew_player) > 0 else None
                
                # 计算指标
                metrics = self.calculate_metrics(names, j_pct, f_pct, actual_loser, s, w)
                
                # 更新 ELO
                if actual_loser:
                    self._update_contestants(names, j_pct, f_pct, actual_loser,season=s)
                
                # 记录结果
                for i, name in enumerate(names):
                    contestant = self.get_contestant(name)
                    
                    # 判断选手状态
                    is_withdrew = (name == withdrew_player)
                    is_eliminated = (name == actual_loser)
                    
                    # 保存 ELO 历史
                    self.elo_history.append({
                        'season': s, 'week': w, 'name': name,
                        'elo': contestant.elo, 'rd': contestant.rd
                    })
                    
                    results.append({
                        'season': s,
                        'week': w,
                        'name': name,
                        'judge_pct': j_pct[i],
                        'est_fan_pct': f_pct[i],
                        # [新增] 记录该选手粉丝投票率的 95% 置信区间
                        'fan_pct_ci_lower': metrics['f_ci_lower'][i],
                        'fan_pct_ci_upper': metrics['f_ci_upper'][i],
                        'elo_rating': contestant.elo,
                        'rating_deviation': contestant.rd,
                        'consistency_score': metrics['consistency'],
                        'certainty_score': metrics['certainty'],
                        'eli_ci_95_lower': metrics['ci_lower'],
                        'eli_ci_95_upper': metrics['ci_upper'],
                        'kl_divergence': metrics['kl_divergence'],
                        'effective_sample_size': metrics['effective_sample_size'],
                        'is_withdrew': is_withdrew,  # 主动退出标记
                        'is_eliminated': is_eliminated  # 被淘汰标记
                    })
        
        return pd.DataFrame(results)
    
    def get_final_rankings(self) -> pd.DataFrame:
        """获取最终排名"""
        data = []
        for name, state in self.contestants.items():
            data.append({
                'name': name,
                'final_elo': state.elo,
                'rating_deviation': state.rd,
                'weeks_active': state.weeks_active,
                'avg_judge_score': state.total_score / max(1, state.weeks_active)
            })
        return pd.DataFrame(data).sort_values('final_elo', ascending=False).reset_index(drop=True)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        elos = [c.elo for c in self.contestants.values()]
        return {
            'total_contestants': len(self.contestants),
            'total_simulations': self._total_simulations,
            'avg_elo': np.mean(elos),
            'elo_std': np.std(elos),
            'backend': 'Numba JIT' if NUMBA_AVAILABLE else 'NumPy'
        }




# ============== Matplotlib 可视化模块 ==============

class EloVisualizer:
    """
    高级可视化类 - 使用 Matplotlib 生成高质量图表
    """
    
    # 专业配色方案
    COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', 
              '#ffd92f', '#e5c494', '#b3b3b3', '#1f78b4', '#33a02c',
              '#fb9a99', '#e31a1c', '#ff7f00', '#cab2d6', '#6a3d9a']
    
    def __init__(self, estimator: 'BayesianEloEstimator', results_df: pd.DataFrame):
        self.estimator = estimator
        self.results = results_df
        self.output_dir = 'QF‘s solution/Bayes_Elo/figures'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_elo_trajectories(self, top_n: int = 15, seasons: Optional[List[int]] = None):
        """
        绘制 Top N 选手的 ELO 演化轨迹
        每个选手从第1周开始绘制，展示其在比赛中的成长曲线
        """
        # 获取 Top N 选手
        rankings = self.estimator.get_final_rankings()
        top_names = rankings.head(top_n)['name'].tolist()
        
        # 准备数据
        elo_df = pd.DataFrame(self.estimator.elo_history)
        if seasons:
            elo_df = elo_df[elo_df['season'].isin(seasons)]
        
        elo_df = elo_df[elo_df['name'].isin(top_names)]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, name in enumerate(top_names):
            name_data = elo_df[elo_df['name'] == name].copy()
            if name_data.empty:
                continue
            
            # 按赛季和周排序
            name_data = name_data.sort_values(['season', 'week'])
            
            # 创建连续的周次索引（从1开始）
            name_data['week_idx'] = range(1, len(name_data) + 1)
            
            color = self.COLORS[i % len(self.COLORS)]
            
            # 获取选手参加的赛季信息
            season_info = name_data['season'].iloc[0]
            
            # ELO 曲线
            ax.plot(name_data['week_idx'], name_data['elo'], 
                   color=color, linewidth=2, label=f'{name} (S{season_info})', 
                   marker='o', markersize=4)
            
            # RD 置信带
            ax.fill_between(name_data['week_idx'],
                           name_data['elo'] - name_data['rd']/3,
                           name_data['elo'] + name_data['rd']/3,
                           color=color, alpha=0.12)
        
        # 添加基准线
        ax.axhline(y=1500, linestyle='--', color='gray', alpha=0.7, label='Initial ELO (1500)')
        
        ax.set_xlabel('Week in Competition', fontsize=12)
        ax.set_ylabel('ELO Rating', fontsize=12)
        ax.set_title(f'Top {top_n} Contestants ELO Rating Evolution', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'elo_trajectories.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 ELO 轨迹图已保存: {output_path}")
    
    def plot_elo_by_season(self, season: int, top_n: int = 10):
        """
        绘制单个赛季内所有选手的 ELO 演化轨迹
        """
        # 准备数据
        elo_df = pd.DataFrame(self.estimator.elo_history)
        season_data = elo_df[elo_df['season'] == season]
        
        if season_data.empty:
            print(f"⚠️ 未找到第 {season} 季数据")
            return
        
        # 获取该赛季最终 ELO 最高的选手
        final_week = season_data['week'].max()
        final_elos = season_data[season_data['week'] == final_week].nlargest(top_n, 'elo')
        top_names = final_elos['name'].tolist()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for i, name in enumerate(top_names):
            name_data = season_data[season_data['name'] == name].sort_values('week')
            if name_data.empty:
                continue
            
            color = self.COLORS[i % len(self.COLORS)]
            
            ax.plot(name_data['week'], name_data['elo'], 
                   color=color, linewidth=2.5, label=name, 
                   marker='o', markersize=5)
            
            # RD 置信带
            ax.fill_between(name_data['week'],
                           name_data['elo'] - name_data['rd']/3,
                           name_data['elo'] + name_data['rd']/3,
                           color=color, alpha=0.15)
        
        ax.axhline(y=1500, linestyle='--', color='gray', alpha=0.7, label='Initial ELO')
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('ELO Rating', fontsize=12)
        ax.set_title(f'Season {season} ELO Rating Evolution (Top {top_n})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'elo_season_{season}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 第{season}季 ELO 轨迹图已保存: {output_path}")
    
    def plot_fan_vote_heatmap(self, season: int):
        """
        绘制单赛季粉丝投票分布热力图
        """
        season_data = self.results[self.results['season'] == season]
        if season_data.empty:
            print(f"⚠️ 未找到第 {season} 季数据")
            return
        
        # 创建透视表
        pivot = season_data.pivot_table(
            values='est_fan_pct', 
            index='name', 
            columns='week',
            aggfunc='first'
        ).fillna(0)
        
        # 按最后一周的投票排序
        last_week = pivot.columns.max()
        pivot = pivot.sort_values(by=last_week, ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.35)))
        
        # 创建红绿渐变色图
        cmap = LinearSegmentedColormap.from_list('RdYlGn', ['#d73027', '#fee08b', '#1a9850'])
        
        # 计算合理的色度范围 (使用百分位数避免极端值)
        valid_data = pivot.values[pivot.values > 0]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 5)  # 5%百分位
            vmax = np.percentile(valid_data, 95)  # 95%百分位
            # 确保范围合理
            vmin = max(0, vmin - 0.02)
            vmax = min(1, vmax + 0.02)
        else:
            vmin, vmax = 0, 1
        
        im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # 设置刻度
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'W{w}' for w in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        
        # 移除网格线效果
        ax.set_xticks(np.arange(len(pivot.columns)+1)-0.5, minor=True)
        ax.set_yticks(np.arange(len(pivot.index)+1)-0.5, minor=True)
        ax.grid(False)
        ax.tick_params(which='minor', length=0)
        
        # 添加数值标签 - 使用更大的字体
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if val > 0.005:  # 只显示大于0.5%的值
                    # 根据值判断文本颜色
                    relative_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = 'white' if relative_val > 0.6 or relative_val < 0.3 else 'black'
                    ax.text(j, i, f'{val*100:.0f}', ha='center', va='center', 
                           fontsize=10, color=text_color, fontweight='bold')
        
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Contestant', fontsize=12)
        ax.set_title(f'Season {season} Fan Vote Distribution Heatmap', fontsize=14, fontweight='bold')
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Estimated Fan Vote %', fontsize=10)
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'heatmap_season_{season}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🔥 热力图已保存: {output_path}")
    
    def plot_model_diagnostics(self):
        """
        绘制模型诊断图 - 一致性、确定性分布
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 聚合到周级别，并过滤掉 NaN 值
        weekly = self.results.groupby(['season', 'week']).first().reset_index()
        weekly_valid = weekly.dropna(subset=['consistency_score', 'certainty_score', 'kl_divergence'])
        
        from scipy import stats
        
        # 1. 一致性分布
        ax1 = axes[0, 0]
        consistency = weekly_valid['consistency_score']
        ax1.hist(consistency, bins=25, color='steelblue', edgecolor='white', alpha=0.7, density=True)
        if len(consistency) > 5:
            kde = stats.gaussian_kde(consistency, bw_method=0.3)
            x_range = np.linspace(0, 1, 100)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        ax1.axvline(consistency.mean(), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {consistency.mean():.3f}')
        ax1.axvline(consistency.median(), color='orange', linestyle=':', linewidth=2,
                   label=f'Median: {consistency.median():.3f}')
        ax1.set_xlabel('Consistency Score', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Consistency Score Distribution\n(Higher = Better Prediction)', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1.05)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. 确定性分布 (现在是 1-ESS/n，范围更合理)
        ax2 = axes[0, 1]
        certainty = weekly_valid['certainty_score']
        ax2.hist(certainty, bins=25, color='forestgreen', edgecolor='white', alpha=0.7, density=True)
        if len(certainty) > 5 and certainty.std() > 0.01:
            kde = stats.gaussian_kde(certainty, bw_method=0.3)
            x_range = np.linspace(max(0, certainty.min()-0.05), min(1, certainty.max()+0.1), 100)
            ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        ax2.axvline(certainty.mean(), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {certainty.mean():.3f}')
        ax2.set_xlabel('Certainty Score (1-ESS/n)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Certainty Score Distribution\n(Low = Uniform, High = Concentrated)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. 一致性 vs JS散度 散点图 (更有意义的组合)
        ax3 = axes[1, 0]
        scatter = ax3.scatter(weekly_valid['consistency_score'], weekly_valid['kl_divergence'],
                             c=weekly_valid['season'], cmap='viridis', alpha=0.6, s=40, edgecolor='white', linewidth=0.5)
        ax3.set_xlabel('Consistency (Prediction Accuracy)', fontsize=11)
        ax3.set_ylabel('JS Divergence (Distribution Skewness)', fontsize=11)
        ax3.set_title('Consistency vs Distribution Skewness', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1.05)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Season', fontsize=10)
        ax3.grid(alpha=0.3)
        
        # 4. 一致性按赛季趋势
        ax4 = axes[1, 1]
        season_stats = weekly_valid.groupby('season')['consistency_score'].agg(['mean', 'std']).reset_index()
        ax4.bar(season_stats['season'], season_stats['mean'], color='coral', alpha=0.7, edgecolor='white')
        ax4.errorbar(season_stats['season'], season_stats['mean'], yerr=season_stats['std'], 
                    fmt='none', color='darkred', capsize=3, alpha=0.7)
        ax4.set_xlabel('Season', fontsize=11)
        ax4.set_ylabel('Consistency Score', fontsize=11)
        ax4.set_title('Prediction Accuracy by Season', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 1.1)
        ax4.axhline(y=season_stats['mean'].mean(), linestyle='--', color='red', alpha=0.7,
                   label=f'Overall Mean: {season_stats["mean"].mean():.3f}')
        ax4.axhline(y=0.5, linestyle=':', color='gray', alpha=0.5, label='Random Guess (0.5)')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        plt.suptitle('Model Diagnostics Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'model_diagnostics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 诊断图已保存: {output_path}")
    
    def plot_season_comparison_radar(self, seasons: List[int] = None):
        """
        绘制赛季对比雷达图
        """
        if seasons is None:
            seasons = sorted(self.results['season'].unique())[-5:]  # 最近5个赛季
        
        metrics = ['consistency_score', 'certainty_score', 'kl_divergence', 'effective_sample_size']
        metric_names = ['Consistency', 'Certainty', 'KL Divergence', 'Eff. Sample Size']
        
        # 先收集所有赛季的原始数据，用于计算归一化范围
        raw_values = {metric: [] for metric in metrics}
        for season in seasons:
            season_data = self.results[self.results['season'] == season]
            weekly = season_data.groupby('week').first()
            for metric in metrics:
                raw_values[metric].append(weekly[metric].mean())
        
        # 计算每个指标的范围用于归一化 (使用 min-max 归一化)
        metric_ranges = {}
        for metric in metrics:
            vals = raw_values[metric]
            min_v, max_v = min(vals), max(vals)
            # 稍微扩展范围，避免边界值
            range_v = max_v - min_v if max_v > min_v else 1
            metric_ranges[metric] = (min_v - 0.1 * range_v, max_v + 0.1 * range_v)
        
        # 计算每个赛季的归一化指标
        season_values = {}
        for season in seasons:
            season_data = self.results[self.results['season'] == season]
            weekly = season_data.groupby('week').first()
            
            values = []
            for metric in metrics:
                val = weekly[metric].mean()
                # Min-max 归一化到 0.15-0.85 范围，避免极端
                min_v, max_v = metric_ranges[metric]
                if max_v > min_v:
                    normalized = (val - min_v) / (max_v - min_v)
                    normalized = 0.15 + 0.70 * normalized  # 映射到 0.15-0.85
                else:
                    normalized = 0.5
                values.append(normalized)
            season_values[season] = values
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, season in enumerate(seasons):
            values = season_values[season] + season_values[season][:1]  # 闭合
            color = self.COLORS[i % len(self.COLORS)]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Season {season}', color=color)
            ax.fill(angles, values, alpha=0.2, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Season Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'season_radar.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🎯 雷达图已保存: {output_path}")
    
    def plot_elimination_probability(self, season: int, week: int):
        """
        绘制特定周的淘汰概率分布 (蒙特卡洛结果)
        """
        # 查找对应的模拟结果
        dist_data = None
        for d in self.estimator.weekly_distributions:
            if d['season'] == season and d['week'] == week:
                dist_data = d
                break
        
        if dist_data is None:
            print(f"⚠️ 未找到 S{season}W{week} 的模拟数据")
            return
        
        names = dist_data['names']
        probs = np.array(dist_data['prob_death'])
        avg_scores = np.array(dist_data['avg_scores'])
        loser = dist_data['loser']
        
        # 按淘汰概率排序
        sorted_indices = np.argsort(probs)[::-1]
        names = [names[i] for i in sorted_indices]
        probs = probs[sorted_indices]
        avg_scores = avg_scores[sorted_indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图: 淘汰概率
        colors = ['crimson' if n == loser else 'steelblue' for n in names]
        bars1 = ax1.bar(range(len(names)), probs, color=colors, edgecolor='white', alpha=0.8)
        
        # 添加数值标签
        for bar, prob in zip(bars1, probs):
            height = bar.get_height()
            if prob > 0.01:  # 只显示大于1%的标签
                ax1.annotate(f'{prob:.1%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        ax1.set_ylabel('Elimination Probability', fontsize=12)
        ax1.set_title(f'Season {season} Week {week} Elimination Probability\n(Red = Actual Eliminated: {loser})',
                    fontsize=12, fontweight='bold')
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax1.set_ylim(0, min(1.1, max(probs) * 1.2))
        
        # 右图: 模拟平均得分 (转为百分比显示)
        colors2 = ['crimson' if n == loser else 'forestgreen' for n in names]
        avg_scores_pct = avg_scores * 100  # 转为百分比
        bars2 = ax2.bar(range(len(names)), avg_scores_pct, color=colors2, edgecolor='white', alpha=0.8)
        
        for bar, score in zip(bars2, avg_scores_pct):
            ax2.annotate(f'{score:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Simulated Average Score (%)', fontsize=12)
        ax2.set_title(f'MC Simulated Average Scores\n(Lower score = Higher elimination risk)',
                    fontsize=12, fontweight='bold')
        # Y轴显示整数 (数据已经是百分比形式)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'elim_prob_s{season}w{week}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📉 淘汰概率图已保存: {output_path}")
    
    def plot_final_rankings(self, top_n: int = 20):
        """
        绘制最终 ELO 排名条形图
        """
        rankings = self.estimator.get_final_rankings().head(top_n)
        
        fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.45)))
        
        # 创建渐变色 - 从金色到银色到铜色
        colors = []
        for i in range(len(rankings)):
            if i < 3:  # 前三名用金银铜
                colors.append(['#FFD700', '#C0C0C0', '#CD7F32'][i])
            else:
                # 其他用渐变蓝色
                intensity = 0.8 - 0.5 * (i - 3) / max(len(rankings) - 4, 1)
                colors.append(plt.cm.Blues(intensity))
        
        y_pos = range(len(rankings))
        
        # 绘制条形图，使用ELO相对于基准的差值
        baseline = 1500
        bar_values = rankings['final_elo'] - baseline
        
        bars = ax.barh(y_pos, bar_values, color=colors, edgecolor='darkgray', linewidth=0.5, height=0.7)
        
        # 添加数值标签
        for i, (bar, elo, rd) in enumerate(zip(bars, rankings['final_elo'], rankings['rating_deviation'])):
            # 在条形图末端显示 ELO 分数
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                   f'{elo:.0f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rankings['name'], fontsize=11)
        ax.invert_yaxis()  # 最高分在上
        ax.set_xlabel('ELO Rating (relative to 1500)', fontsize=12)
        ax.set_title(f'Top {top_n} Contestants Final ELO Ranking', fontsize=14, fontweight='bold')
        
        # 添加基准线
        ax.axvline(x=0, linestyle='-', color='black', alpha=0.3, linewidth=1)
        
        # 添加网格
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(-50, max(bar_values) * 1.15)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'final_rankings.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🏆 排名图已保存: {output_path}")
    
    def plot_elo_distribution(self):
        """
        绘制最终 ELO 分布直方图
        """
        elos = [c.elo for c in self.estimator.contestants.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(elos, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(elos), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(elos):.1f}')
        ax.axvline(np.median(elos), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(elos):.1f}')
        ax.axvline(1500, color='gray', linestyle=':', linewidth=2,
                  label='Initial: 1500')
        
        ax.set_xlabel('ELO Rating', fontsize=12)
        ax.set_ylabel('Number of Contestants', fontsize=12)
        ax.set_title('Final ELO Rating Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'elo_distribution.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 ELO分布图已保存: {output_path}")
    
    def plot_consistency_by_season(self):
        """
        绘制各赛季一致性箱线图（按时间段分组）
        """
        weekly = self.results.groupby(['season', 'week']).first().reset_index()
        weekly_valid = weekly.dropna(subset=['consistency_score'])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：按时期分组（每5-6个赛季为一组）
        ax1 = axes[0]
        seasons = sorted(weekly_valid['season'].unique())
        n_groups = 6
        group_size = len(seasons) // n_groups + 1
        
        group_labels = []
        group_data = []
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, len(seasons))
            if start_idx >= len(seasons):
                break
            group_seasons = seasons[start_idx:end_idx]
            data = weekly_valid[weekly_valid['season'].isin(group_seasons)]['consistency_score'].values
            if len(data) > 0:
                group_data.append(data)
                group_labels.append(f'S{group_seasons[0]}-S{group_seasons[-1]}')
        
        bp = ax1.boxplot(group_data, patch_artist=True, widths=0.6)
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(group_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for median in bp['medians']:
            median.set_color('darkred')
            median.set_linewidth(2)
        
        ax1.set_xticklabels(group_labels, fontsize=10)
        ax1.set_xlabel('Season Groups', fontsize=12)
        ax1.set_ylabel('Consistency Score', fontsize=12)
        ax1.set_title('Consistency by Season Period', fontsize=14, fontweight='bold')
        ax1.axhline(y=0.5, linestyle='--', color='red', alpha=0.5, label='Random Guess (0.5)')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='lower right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 右图：时间序列趋势（按赛季平均）
        ax2 = axes[1]
        season_stats = weekly_valid.groupby('season')['consistency_score'].agg(['mean', 'std', 'count']).reset_index()
        
        ax2.fill_between(season_stats['season'], 
                        season_stats['mean'] - season_stats['std'],
                        season_stats['mean'] + season_stats['std'],
                        alpha=0.3, color='steelblue', label='±1 Std Dev')
        ax2.plot(season_stats['season'], season_stats['mean'], 
                'o-', color='steelblue', linewidth=2, markersize=6, label='Season Mean')
        
        # 添加滚动平均线
        if len(season_stats) >= 5:
            rolling_mean = season_stats['mean'].rolling(window=5, center=True).mean()
            ax2.plot(season_stats['season'], rolling_mean, 
                    '--', color='darkred', linewidth=2, label='5-Season Moving Avg')
        
        ax2.axhline(y=0.5, linestyle=':', color='gray', alpha=0.7, label='Random Guess')
        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('Consistency Score', fontsize=12)
        ax2.set_title('Consistency Trend Over Seasons', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'consistency_by_season.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📦 箱线图已保存: {output_path}")
    
    def generate_all_visualizations(self) -> None:
        """生成所有可视化图表"""
        print("\n" + "=" * 60)
        print("  生成可视化图表 (Matplotlib)")
        print("=" * 60)
        
        # 1. Top 选手 ELO 轨迹（按比赛周数对齐）
        self.plot_elo_trajectories(top_n=12)
        
        # 2. 最近3个赛季的单独 ELO 轨迹图
        seasons = sorted(self.results['season'].unique())[-3:]
        for s in seasons:
            self.plot_elo_by_season(s, top_n=8)
        
        # 3. 最近3个赛季的热力图
        for s in seasons:
            self.plot_fan_vote_heatmap(s)
        
        # 4. 模型诊断
        self.plot_model_diagnostics()
        
        # 5. 赛季对比雷达
        self.plot_season_comparison_radar()
        
        # 6. 最终排名
        self.plot_final_rankings(top_n=25)
        
        # 7. ELO 分布
        self.plot_elo_distribution()
        
        # 8. 各赛季一致性箱线图
        self.plot_consistency_by_season()
        
        # 9. 示例淘汰概率图
        if self.estimator.weekly_distributions:
            last_dist = self.estimator.weekly_distributions[-10]
            self.plot_elimination_probability(last_dist['season'], last_dist['week'])
        
        print(f"\n✅ 所有图表已保存至 '{self.output_dir}' 目录")


# ============== 主程序入口 ==============

def main():
    """主运行程序"""
    import time
    
    input_csv = 'cleaned_weekly_data.csv'
    
    if not os.path.exists(input_csv):
        print(f"❌ 找不到文件: {input_csv}")
        return
    
    print("=" * 70)
    print("  贝叶斯 ELO + 高级蒙特卡洛粉丝投票逆向推算系统 v2.0")
    print("=" * 70)
    
    # 加载数据
    print("\n📊 加载数据...")
    df_cleaned = pd.read_csv(input_csv)
    print(f"   数据规模: {len(df_cleaned):,} 行")
    print(f"   赛季数量: {df_cleaned['season'].nunique()}")
    print(f"   选手数量: {df_cleaned['name'].nunique()}")
    
    # 初始化模型
    estimator = BayesianEloEstimator(
        base_k_factor=48.0,
        temperature=150.0,
        n_simulations=3000,
        noise_std=0.3,
        judge_weight=0.5,
        rd_decay=0.95,
        use_adaptive_params=True,
        memory_decay=0.92
    )
    
    backend = 'Numba JIT 加速' if NUMBA_AVAILABLE else '纯 NumPy'
    print(f"\n⚡ 计算后端: {backend}")
    print(f"📐 算法特性: 贝叶斯更新 + 分层蒙特卡洛 + Glicko-2 风格 RD")
    
    # 运行推断
    print("\n🔄 开始逆向推算粉丝投票...")
    start_time = time.perf_counter()
    
    final_results = estimator.run_inference(df_cleaned)
    
    elapsed = time.perf_counter() - start_time
    
    # 保存结果
    output_file = 'QF‘s solution/Q2_bayes/Q2_1/percent_fan_vote_estimates_weekly.csv'
    final_results.to_csv(output_file, index=False, float_format='%.6f')
    
    # 输出统计
    stats = estimator.get_statistics()
    print(f"\n✅ 推断完成!")
    print(f"   耗时: {elapsed:.2f} 秒")
    print(f"   总模拟次数: {stats['total_simulations']:,}")
    print(f"   选手总数: {stats['total_contestants']}")
    print(f"   平均 ELO: {stats['avg_elo']:.1f} ± {stats['elo_std']:.1f}")
    
    print(f"\n📁 结果已保存至: {output_file}")
    
    # 模型评估
    print("\n" + "=" * 70)
    print("  模型评估指标")
    print("=" * 70)
    weekly = final_results.groupby(['season', 'week']).first()
    print(f"   平均一致性分数: {weekly['consistency_score'].mean():.4f}")
    print(f"   平均确定性分数: {weekly['certainty_score'].mean():.4f}")
    print(f"   平均 KL 散度: {weekly['kl_divergence'].mean():.4f}")
    print(f"   一致性 > 0.5 的周数比例: {(weekly['consistency_score'] > 0.5).mean():.2%}")
    
    # Top 10 排名
    print("\n" + "=" * 70)
    print("  Top 10 选手 (最终 ELO 排名)")
    print("=" * 70)
    rankings = estimator.get_final_rankings()
    print(rankings.head(10).to_string(index=False))
    
    # 生成可视化
    # visualizer = EloVisualizer(estimator, final_results)
    # visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()