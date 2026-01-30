# MCM2026

2026美赛共享库，包含 latex 模板，MCM 题面。我们选择的是 C 题。

## 数据清洗

在 `clean_data.py` 中实现清洗数据功能。对出身美国的名人，将其出生地标准化为州名。对其他国家或地区的名人，统一沿用原始数据。我们统计了每位名人每周的平均评委得分，按照 season-week-avg_score 的顺序排序，保存为 `cleaned_weekly_avg.csv` 文件。

## 核心算法
第一问算法:[蒙特卡洛模拟 + 启发式梯度更新]：
对每一周，重复50次以下步骤：
    初始假设观众对每个选手的投票率相等：
    比如X = [0.25,0.25,0.25,0.25],SUM(Xi)=1
    基于当前参数X ，生成 M组投票样本M = 1000，设置步长n，设 fi为选手 i在模拟中被淘汰的概率
    淘汰总得分最低者（平局时随机选择）； 
    统计每位活跃选手在 1000 次模拟中被淘汰的频率，记为 f_i（即模拟淘汰概率）； 
    构造目标向量 t_i：实际被淘汰者 t_i = 1，其余 t_i = 0； 
    计算误差 e_i = t_i - f_i； 
    更新每位活跃选手的支持度：X_i ← X_i × (1 - η × e_i)，其中 η 为步长（如 0.05）； 
    对全局 X 向量重新归一化，确保 sum(X_i) = 1
下一周，引入记忆系数a=0.7以评估观众的记忆，令Xt = aX(t-1)+(1-a)Xbase, 重新归一化，使得SUM(Xi)=1，重复上述步骤。

## 性能评估方法
一致性：
设在第 t 周，实际被淘汰的选手为 e obs(t)
​模型推断出的潜在观众支持度向量为 Xt，裁判评分为 Jt
则模型预测每个选手 i 被淘汰的概率分布为Pt为关于Jit和Xit的一个函数
那么定义一致性指标CIt = sum(I*Pit)
I = 1 ，如果选手i是实际淘汰者
I = 0 ，如果选手i不是实际被淘汰者

我们做20次独立的蒙特卡罗模拟，即计算20个主模型，取CI的平均值作为输出。如果这个均值很低且方差大，说明比赛规则在这一周充满了混沌，结果几乎是随机的。

确定性：
在20次模拟中，如果X波动较小（方差小），则认为该解空间置信度较高，具有较高的确定性

## 核心算法V2
依然基于[蒙特卡洛模拟]，但是初始假设[改为截断的齐普夫分布],规避了均匀分布的初始假设可能会导致结果偏向于“大家的票数都差不多”的问题

对每一比赛周：
首先随机选定一组满足[截断的齐普夫分布]的base_share,例如[0.61, 0.31, 0.08]
对每一周，重复10000次以下步骤：
    将每一base_share随机分配给不同选手得到fan_share：比如【A：0.61；B：0.08；C：0.31】
    模拟一次投票
对10000次模拟投票结果：
    假设有n次投票结果符合实际结果，那么取这n次的fan_share取平均
    根据这n次模拟的结果，可以计算N次模拟的方差，用以评估模型的[稳定性]
    对n次模拟的结果排序，取每个选手的2.5% 和 97.5% 分位数，构成[95%置信区间]
    得到最终输出的投票率向量X
对输出的投票率向量X：
    再次运行蒙特卡洛模拟1000次，看看有n次与现实相符，即可输出[一致性]

目前的方法已经实现：
    忽略退赛者；其最后一期裁判分数计入投票率，但是不影响下一期
    对决赛周，排名序列（1，2，3）名将纳入考虑
    
局限：
    不淘汰人的周没法产生有效信息；对这样的周，无法得到合理预测；

## 可视化
[Normalized_Survival_Frontier.png]:
1. 为什么要“归一化”？
背景差异：
在第1周（12人参赛），平均得票率是 1/12≈8.3%。如果你拿了 10% 的票，你已经表现很好了。
在第9周（4人参赛），平均得票率是 1/4=25%。如果你拿了 10% 的票，你就是倒数第一，必死无疑。
归一化公式：
Normalized Score=Actual Share × N
其中 N 是当周参赛人数。
数学含义：
值 = 1.0：表现处于平均水平。
值 > 1.0：表现优于平均。
值 < 1.0：表现低于平均。
通过这个变换，我们可以把34个赛季所有周次的数据点放在同一张图上，寻找那条普适的“生存法则”。
2. 图表解读 (论文看点)
X轴 (Normalized Judge Score)：裁判给了你多少分（相对于平均水平）。
Y轴 (Normalized Fan Vote)：模型估算出你拿了多少观众票（相对于平均水平）。
死亡区 (Death Zone)：左下角的区域（X<1 且 Y<1）。你会发现红色的淘汰点高度集中在这里。这意味着：如果你裁判分低，观众票也低，你必死无疑。
生存边界 (The Frontier)：红点和蓝点之间会形成一个模糊的分界面。这个分界面大致遵循 X+Y=C 的规律。
高能异常点 (High-Fan Anomalies)：重点关注那些位于左上角的蓝点。
它们代表：裁判分很低 (X<0.8)，但居然存活了（蓝色）。
原因：你看它们的 Y 轴坐标，通常高达 1.5 甚至 2.0。这直观地证明了：只要粉丝够疯狂（票数是平均值的2倍），裁判给0分你也走不了。
[Trajectory_Alfonso_Ribeiro.png]
展示19季冠军的人气曲线；置信区间在决赛场收窄是因为决赛周的模型逻辑从“存活判定”切换到了“精确排名判定”的全序约束，极大地压缩了参数空间，剔除了 99% 以上在常规周可行的随机扰动。这种从“不等式约束”向“等式约束”的质变，是导致 CI 宽度骤降的根本原因。
"The persistent width of the confidence intervals during preliminary weeks reflects the inherent observational sparsity of the DWTS voting system. Our Monte Carlo analysis correctly identifies that the 'survival constraint' (avoiding elimination) is insufficient to uniquely localize fan preferences. The subsequent sharp convergence of the CI in the final week (W=0.0275) validates that our model successfully transitions from a low-information regime to a high-information regime, capturing the definitive popularity of the winner only when strict ordinal data becomes available."

