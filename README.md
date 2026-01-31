# MCM2026

2026美赛共享库，包含 latex 模板，MCM 题面。我们选择的是 C 题。

## 数据清洗

在 `clean_data.py` 中实现清洗数据功能。对出身美国的名人，将其出生地标准化为州名。对其他国家或地区的名人，统一沿用原始数据。我们统计了每位名人每周的平均评委得分，按照 season-week-avg_score 的顺序排序，保存为 `cleaned_weekly_avg.csv` 文件。

## 核心算法

### 第一问算法 1：**蒙特卡洛模拟 + 启发式梯度更新**

 对每一周，重复50次以下步骤：  
    &emsp;&emsp;均匀初始化观众投票率 $X$ 使得 $\sum X_i=1$ 且 $X_i=1/N$，$N$ 为存活的选手数；  
    &emsp;&emsp;基于当前参数X ，生成 M 组投票样本 M = 1000，设置步长n，设 $f_i$为选手 i 在模拟中被淘汰的概率  
    &emsp;&emsp;淘汰总得分最低者（平局时随机选择）；   
    &emsp;&emsp;统计每位活跃选手在 1000 次模拟中被淘汰的频率，记为 $f_i$（即模拟淘汰概率）；   
    &emsp;&emsp;构造目标向量 $t_i$：实际被淘汰者 $t_i = 1$，其余 $t_i = 0$；   
    &emsp;&emsp;计算误差 $e_i = t_i - f_i$；   
    &emsp;&emsp;更新每位活跃选手的支持度：$X_i ← X_i × (1 - η × e_i)$，其中 η 为步长（如 0.05）；   
    &emsp;&emsp;对全局 X 向量重新归一化，确保 $sum(X_i) = 1$  
下一周，引入记忆系数a=0.7以评估观众的记忆，令$Xt = aX(t-1)+(1-a)Xbase$, 重新归一化，使得$SUM(Xi)=1$，重复上述步骤。

### 第一问算法 2：**贝叶斯推断 + Stratified MC**

**初始化**：
- 为所有选手初始化 elo 评分 $Elo_i = 1500$，RD 不确定度 $RD_i = 350$。

**每周更新**：  
1. 使用 $Softmax$ 函数将 elo 评分转化为观众投票可能性分布 $X$：  
   $$X_i = \frac{e^{Elo_i / T}}{\sum_{j} e^{Elo_j / T}}$$  
   其中 $T$ 为温度参数，控制分布的平滑度，随时间衰减。

2. 施加分层随机噪声，使用蒙特卡洛方法模拟 3000 次投票过程，记录每位选手的淘汰频率 $f_i$。
3. 比较模型预测的淘汰者和实际淘汰者，计算误差 $e_i = t_i - f_i$，其中 $t_i$ 为目标向量。
4. 更新选手的 elo 评分和 RD：  
   - 若选手被淘汰，降低其 elo 评分，增加 RD。  
   - 若选手未被淘汰，提升其 elo 评分，减少 RD。  
   - 更新公式参考 Glicko-2 系统。

**性能分析**
1. RD 能够描述模型对局势掌控程度，越高的 RD 代表模型对选手状态越没有把握。
2. KL 散度描述模型预测分布和实际分布差异，越低的 KL 散度代表越好的预测能力。

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

## Q1V2可视化
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
3. [模型评价图]
【不确定性分布直方图】 (Distribution of Model Uncertainty)
图象特征：如果直方图呈现“高瘦”的形态（High Peak），且长尾（Long Tail）较少，大部分数据集中在左侧低值区。说明了模型对绝大多数选手的粉丝份额估计是非常确定的（Standard Deviation很小）。模型不是在“瞎猜”，而是有很强的信号捕捉能力。异常值受控：虽然存在极少数难以预测的个例（长尾部分），但整体系统的预测误差是收敛且可控的。
【估计值与置信区间宽度关系图】 (Estimate vs. CI Width)图象特征：横轴是粉丝份额，纵轴是置信区间宽度。如果你看到点沿一条斜线分布，或者均匀分布，没有奇怪的离群点。结构一致性：证明了模型对待所有选手是“公平”的。不确定性（CI Width）通常随估计值（Est Share）的增加而线性增加，这是统计学上的自然现象（方差与均值相关）。无系统性偏差：如果图上出现某些点偏离主群体很远，说明模型对某些特定类型的选手“失灵”了。如果点聚类紧密，说明模型逻辑高度一致。
【跨赛季波动性箱线图】 (Robustness Across Seasons)
图象特征：横轴是赛季，纵轴是标准差。如果各个箱子的高度（中位数）基本持平，没有某个赛季突然极其离谱。说明了什么（Robustness / Generalizability）：
适应规则变更：题目背景中提到DWTS的规则在第1、2赛季用Rank，后面用Percentage。如果这张图显示早期赛季和后期赛季的模型波动性一致，直接证明了你的模型具有极强的泛化能力，未受规则剧变的影响。时间鲁棒性：证明模型不会因为时间推移、选手质量变化而失效。
【模拟有效性与不确定性关系图】 (Valid Simulations vs. Uncertainty)
图象特征：横轴是有效模拟次数，纵轴是标准差。如果随着模拟次数增加，标准差趋于稳定（不再剧烈跳动）。说明了什么（Algorithmic Convergence）：算法收敛：这是对你使用的蒙特卡洛（Monte Carlo）或贝叶斯采样方法的直接验证。说明你的采样次数（Simulations）已经足够多，结果已经收敛到真值附近，不再受随机数种子的干扰。

## Q2
基于Q1V2计算：分别用两种方法估算了每个赛季的预测投票率，可视化方法展示如下：
【散点图】如果两个模型完全一致，所有点都应该沿着对角线分布
现在发现：低分区：点分布在红线上方。这意味着对于弱势选手（粉丝少），排名制给出的权重往往高于其实际得票率。（Rank制在“补贴”弱者）。高分区：点分布在红线下方。这意味着对于超级明星（粉丝多），排名制给出的权重往往低于其实际得票率。（Rank制在“打压”强者）。证明了排名制具有**“均值回归（Regression to the Mean）”**的系统性偏差：
观察上色后的散点图，实际上采用了百分比法的人，如果将方法改成排名法，那么他的估算得票率将更高；说明如果所有选手的评委分数非常接近（例如都在 25−28 分之间），他们在百分比制下的得分差距极小，在这种情况下，选手只需要少量的粉丝投票优势（较低的得票率P）就能弥补评委分的不足。而一旦切换到排名制，这种“接近”的状态消失了，取而代之的是冰次的位次差距，迫使模型估算出更高的粉丝得票率来维持其生存结果。
[方差压缩]排名制将极端的得票差异（如 90% vs 10%）强制压缩为相邻的整数（Rank 1 vs Rank 2）。这导致了数据的方差显著降低。
[罗宾汉效应]Rank Rule 实际上在进行“劫富济贫”。它拿走了高人气选手的“溢出选票”（多出来的票对排名没用），并隐性地提升了低人气选手的权重（只要不是最后一名，排名分就不会太难看）。
[信息丢失]散点的离散程度（垂直方向的抖动）表明，排名制丢失了“差距到底有多大”这一关键信息。
Figure 1 presents a scatter plot comparison between the estimated fan shares derived from the Percentage Rule (x-axis) and the Rank Rule (y-axis). The red dashed line represents the locus of perfect agreement (y=x). While the two methods exhibit a positive correlation (R^2>0.9), a distinct sigmoidal deviation pattern is observable, revealing the inherent "equalizing bias" of the Rank Rule.
1. Subsidization of the Tail (Low-end Bias): In the lower quadrant (fan share <0.15), the data points predominantly lie above the reference line. This indicates that the Rank Rule systematically overestimates the support for less popular contestants. By converting continuous vote counts into discrete ordinal ranks, the algorithm artificially inflates the value of marginal survival, effectively providing a "safety net" for weaker contestants.
2. Suppression of the Head (High-end Compression): Conversely, in the upper quadrant, the Rank Rule estimates consistently fall below the reference line. This phenomenon, known as variance compression, demonstrates that the Rank mechanism fails to capture the magnitude of "superstar" popularity. A contestant with 50% of the total votes receives the same rank score (Rank 1) as one with 25% in a tighter race, thereby discarding significant voter preference information.
Conclusion: Mathematically, the Rank Rule acts as a low-pass filter, smoothing out extreme variations in public opinion. While this may increase competition suspense, it statistically distorts the true "Voice of the People," making it harder for dominant talent to secure a mathematically justified lead.
【柱状图】【概率密度分布图】感觉这两张图没什么有用的信息……主要都显示了两张图粉丝投票总体分布上是没有差异的

## Q2_2)
数据标准化：为了消除不同赛季裁判人数（3人或4人）的影响，我们将每周的裁判原始分转换为标准分（Z-Score）。
设定淘汰阈值：在每一周，找到被淘汰选手中的平均分。这代表了当周“被淘汰的平均水平”。
计算幸存偏差：检查所有幸存（未被淘汰）的选手。如果某个幸存选手的得分低于这个阈值，说明他/她本该被淘汰（按裁判标准），但被粉丝救了回来。
争议分累加：计算差值（阈值 - 幸存者得分），并将其累加。差值越大，说明“德不配位”的程度越严重。
总排名：按累加的争议分对所有历史选手进行排序，得分最高者即为“最具争议选手”（也就是依靠粉丝投票逆风翻盘最严重的选手）。
## Q3
第二阶段：特征工程 (Feature Engineering)
【舞伴能力指数】：直接用 One-Hot 编码舞伴会导致特征稀疏（维度太高）。使用Target Encoding (目标编码)。计算该舞伴在其他赛季的历史平均分。公式Efficacyp=Average Score of Partner p in all seasons except current
【行业归类】 将细碎的 Industry（如 "Soap Opera Actor", "Film Actor"）归类为大类（"Actor"）。创建 Is_Athlete (是否运动员), Is_Performer (是否表演类) 等布尔特征。
【年龄的非线性处理】：假设：年龄的影响不是线性的（太老跳不动，太年轻没阅历）。添加年龄的平方项，捕捉 "倒U型" 或 "U型" 关系

第三阶段：双模型训练 (Dual-Model Training)
XGBoost模型训练
Model A (Judge):
X (特征): Age, Partner_Efficacy, Industry_Features, Gender, Season, Week...
Y (目标): Total_Judge_Score
Model B (Fan):
X (特征): 同上
Y (目标): Estimated_Fan_Votes (需标准化，如取对数 log1p，因为票数跨度大)
第四阶段：SHAP 归因分析与对比:
1. 核心发现 (Key Findings)
根据模型输出的特征重要性对比（已生成 feature_importance_comparison.csv）：舞伴是决定性因素，但裁判更看重 (The "Pro" Effect)差异：它解释了 54.3% 的裁判打分变异，但只解释了 44.5% 的粉丝投票变异。裁判评分高度依赖于舞伴的技术指导（Technical Merit），而粉丝虽然也看重表现，但受舞伴的影响相对较弱（约低10个百分点），这留出了更多空间给“明星个人魅力”。
年龄的双重效应 (The Age Paradox)
年龄（及年龄平方）在两个模型中的重要性都排在第2、3位（合计约38%-41%）。差异：年龄对粉丝的影响权重（合计 41.3%）竟然略高于对裁判的影响（合计 38.0%）。通常认为裁判会惩罚高龄选手的僵硬，但数据显示粉丝对年龄非常敏感。结合SHAP分析，这可能意味着粉丝不仅会投票给年轻偶像，也会对高龄励志选手表现出强烈的情感偏好（Sympathy Vote），或者反过来，极其排斥某些年龄段。行业影响微弱,行业特征（Industry）的整体贡献度较低，说明“你是谁（演员/运动员）”不如“你表现得怎么样（舞伴/年龄状态）”重要。
Divergent Age Dynamics. The blue dashed line illustrates a steep technical penalty for older contestants from judges. In contrast, the orange solid line reveals that fan support remains relatively resilient for older demographics, decoupling popularity from physical performance decline.

## Q4
> Propose another system using fan votes and judge scores each week that you believe is more
“fair” (or “better” in some other way such as making the show more exciting for the fans).
Provide support for why your approach should be adopted by the show producers.
>
> 使用每周的粉丝投票和评委评分数据来设计另一个你认为更“公平”的评分系统（或者在某些方面更“好”，比如让节目对粉丝而言更精彩）。提供支持你观点的理由，说明为什么你的方法应该被节目制作人采纳。

[] 尚未实现代码

解决这个问题要关注多方面因素：

1. 公平性：相信评委专业性，始终确保评委评分占大头，粉丝投票作为节目效果不应过分影响结果；
2. 趣味性：粉丝投票能影响结果，留下人气选手/引发争议增加节目热度；

目前的计分方案：

- 采用带权重的双对数形式：
  $$ total\_score=judge\_pct+0.2\cdot\frac{\ln(1+fan\_pct)}{\ln(1+\max(fan\_pct))} $$

理由：

1. **公平**。评委评分占比极大，保证专业性，粉丝投票经过对数压缩，避免人气过分影响结果；
2. **趣味**。粉丝投票有影响力，能够挽救人气选手维持热度；

其他额外方案：

- **突然死亡**：每周统计总分后选出垫底的两个选手，进行淘汰对决，由评委和粉丝重新投票计算总分。
- **进步奖励**：每周计算每个选手的评委评分与上周的差值，进步最大的选手本次粉丝投票得分权重变为0.3；