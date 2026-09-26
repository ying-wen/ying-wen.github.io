# 温颖的 AI 艺术研究笔记

规范页面: https://yingwen.io/zh/art/

## 研究思考：第一性原理 Slogans

一组 20 幅极简抽象图，围绕 AI 研究中的问题意识、世界 grounding、经验学习、验证机制、抽象形成与可累积科学展开。

### 问题先行

深灰色块是现实瓶颈，是那个真正阻碍进展的部分。蓝色方法碎片悬在其上，有用但尚未落地；绿色路径从问题本身出发，最后才抵达琥珀色验证点。这里的解读是：好的研究不是先问用什么流行方法，而是先找到那个一旦解决就会改变长期能力的约束。

- Image: https://yingwen.io/art/yingwen-research-slogans/01-problem-before-method.png
- 原始表述: 从问题出发，不从方法出发。

### 世界先于模型

宽阔的灰绿色平面代表世界：环境、信号、行动界面和后果。蓝色模型被画得很小，因为它只是闭环中的中介，而不是世界本身的替代物。图像反对模型中心主义，追问系统是否真正接入了反馈、干预和外部结构。

- Image: https://yingwen.io/art/yingwen-research-slogans/02-world-before-model.png
- 原始表述: 从世界出发，不从模型出发。

### 主体视锥

黑色开口是处在情境中的主体，蓝色视锥是从它的位置出发可被感知、可被行动化的世界。它不是旁观者的完整地图，而是在有限信息、成本和延迟反馈下的可用世界。视锥外的红点表示那些也许真实、但尚不能指导行动的事实，因此 agent 视角的研究要问系统知道什么、能做什么、如何更新、如何恢复。

- Image: https://yingwen.io/art/yingwen-research-slogans/03-agent-point-of-view.png
- 原始表述: 采取 agent 的视角。

### 经验小环

浅色堆叠是静态数据：有价值、覆盖广、来自过去，但本身是惰性的。观察、行动、反馈和轨迹组成的小环被放在视觉中心，因为它能生成新信息并改变学习者。这里不是反数据，而是强调当数据变成经验、经验变成更新时，智能才开始复利。

- Image: https://yingwen.io/art/yingwen-research-slogans/04-experience-before-static-data.png
- 原始表述: 经验比静态数据更接近智能的来源。

### 验证门

灰色碎片靠近紫色验证门，只有一个通过后变成稳定的蓝色对象。红色碎片被留在门外，作为残留而不是被吸收进记忆。它表达的是学习卫生：反馈可以弱、可以延迟、可以来自社会、实验或形式证明，但在被系统当成知识之前，必须先说明它的验证强度。

- Image: https://yingwen.io/art/yingwen-research-slogans/05-learn-only-what-can-verify.png
- 原始表述: 不要让系统学习它无法验证的东西。

### 能力曲线

浅色记分板被故意淡化：它记录了数字，但不能自动证明能力。绿色曲线由不同情境中的琥珀色测量点支撑，暗示迁移、保持、恢复和成本下降。解读是：benchmark 是仪器，不是目标；当分数取代了它本来要测量的能力概念时，就会误导研究。

- Image: https://yingwen.io/art/yingwen-research-slogans/06-capability-not-score.png
- 原始表述: 衡量能力，不要崇拜分数。

### 抽象升起

灰色样本云停留在底部，蓝色结构从中升起。绿色行动线和琥珀色验证点让这个抽象可以被检验，而不只是漂亮的概念图。这里强调：智能不是记住多少案例，而是能从经验中抽取出可复用结构，并在新情境中接受检验。

- Image: https://yingwen.io/art/yingwen-research-slogans/07-abstraction-not-samples.png
- 原始表述: 研究抽象，不只研究样本。

### 有用结构

淡淡的背景暗示完整复刻世界这一不可能的幻想。前景只保留有用结构：预测弧线、行动路径和验证点。解读是：世界模型的价值不在于看起来逼真，而在于是否帮助智能体规划、行动、发现变量并修正错误。

- Image: https://yingwen.io/art/yingwen-research-slogans/08-useful-world-structure.png
- 原始表述: 世界模型不是复刻世界，而是保存有用结构。

### 可扩展机制

上方结构不断重复和扩展，像一种能随算力、数据、环境或经验增长而复利的机制。下方红色技巧碎片彼此孤立，也许聪明、也许一时有用，但不是长期路线。它区分了 search、learning、planning、memory、verification 这类可扩展机制，以及只在某个榜单或产品形态里成立的人工补丁。

- Image: https://yingwen.io/art/yingwen-research-slogans/09-general-mechanisms-not-tricks.png
- 原始表述: 长期路线应扩展一般机制，而不是扩展人工技巧。

### 失败改写路线

黑色假设线撞上红色失败边界，并在琥珀色测试点之后弯折为绿色路线。失败不是附录，而是直接改变路径。它表达的是：一个研究方向只有在证据能够削弱它、重定向它，或迫使问题定义变清楚时，才真正接近科学。

- Image: https://yingwen.io/art/yingwen-research-slogans/10-failure-conditions.png
- 原始表述: 好研究要能改变自己的失败条件。

### 小实验，大假设

蓝色大圆是宏大假设，极小的琥珀点是关键实验。黑色针尖瞄准因果铰链，而不是试图覆盖整个空间。它的解读是：小实验如果能决定我们相信什么、放弃什么，就很有力量；规模应该让问题更尖锐，而不是把模糊命题藏在大 demo 后面。

- Image: https://yingwen.io/art/yingwen-research-slogans/11-minimal-experiment-max-hypothesis.png
- 原始表述: 用最小实验攻击最大假设。

### 验证阶梯

小方块是强验证起点，浅色大球是研究最终想进入的开放世界。中间的阶梯表示逐步放松假设，而不是突然跳跃。解读是：开放世界研究需要验证阶梯，先在反馈可靠的地方证明机制，再逐步加入模糊性、规模和真实复杂度。

- Image: https://yingwen.io/art/yingwen-research-slogans/12-verified-domain-to-open-world.png
- 原始表述: 开放世界很重要，但强验证域常常更适合起步。

### 流畅裂隙

流畅的蓝色带状物代表顺滑表达：连贯、好看，也很容易让人过度信任。下方绿色结构是可执行机制，中间的红色裂隙表示缺失的证据。它提醒我们：语言可以是智能的一部分，但只有连接到预测、行动、技能、错误恢复或新知识时，才更接近理解。

- Image: https://yingwen.io/art/yingwen-research-slogans/13-fluency-not-understanding.png
- 原始表述: 不要把流畅表达误认为理解。

### 选择性记忆

庞大的档案被故意退到背景，因为容量本身不是智能。只有一个记忆块穿过检索边界并连接到行动，红色过期碎片留在外面。解读是：记忆应该降低未来成本并改善决策；坏记忆会增加检索负担、冲突、过期和误用。

- Image: https://yingwen.io/art/yingwen-research-slogans/14-selective-memory.png
- 原始表述: 记忆不是越多越好。

### 双重边界

外层矩形是系统边界：模型、工具、检索、用户、协议和基础设施。内层矩形才是学习真正发生的位置。它提醒我们不要把整个系统行为都归因于模型；严肃研究必须问清楚哪部分在更新、哪部分在存储、哪部分在路由、哪部分只是脚手架。

- Image: https://yingwen.io/art/yingwen-research-slogans/15-system-learning-boundaries.png
- 原始表述: 系统边界和学习边界同样重要。

### 内生对齐

绿色学习线和紫色对齐线从一开始就交织，而不是在最后才被连接。漂浮在外的红色补丁表示把对齐当成表面修补的脆弱性。它的解读是：安全、可信和可控本质上是学习设计问题，会影响反馈、目标、行动空间、环境和更新规则。

- Image: https://yingwen.io/art/yingwen-research-slogans/16-alignment-inside-learning.png
- 原始表述: 对齐不是后处理，而是学习问题的一部分。

### 新证据生成

灰色档案包含旧知识，但中央绿色行动生成了新的蓝色证据点，并且只有经过琥珀色反馈后才成立。图像区分了文献压缩和科学发现。解读是：AI for Science 应逐步从总结论文走向提出实验、生成观察、归因失败，并产出可复现实验证据。

- Image: https://yingwen.io/art/yingwen-research-slogans/17-new-experience-for-science.png
- 原始表述: 科学发现需要生成新经验，不只是压缩旧知识。

### 多样性与反驳

三个智能体围绕共享验证点构成三角形，形成一种关于主张与检查的社会几何。红色矛盾没有被抹掉，而是改变结构，说明分歧可以成为信息。解读是：多智能体的价值不只是把 workflow 拆成角色，而是产生假设多样性、对抗检查、互证纠错和更丰富的反馈。

- Image: https://yingwen.io/art/yingwen-research-slogans/18-diversity-and-refutation.png
- 原始表述: 多智能体的价值不只是分工，而是产生多样性和反驳。

### 研究品味

灰色工具块铺在下方，数量很多、彼此可替换。上方蓝色结构、绿色行动和琥珀色验证之间的小平衡，才是真正稀缺的判断点。解读是：工具变化很快，研究品味是更慢的能力——选择真问题、拒绝噪声进展、设计能反驳自己的实验，并识别哪些机制会复利。

- Image: https://yingwen.io/art/yingwen-research-slogans/19-research-taste.png
- 原始表述: 研究品味比技术栈更稀缺。

### 累积之阶

蓝绿色验证块形成一段后来者可以继续攀登的阶梯。红色 demo 漂浮在旁边，也许醒目、也许令人印象深刻，但没有接入阶梯。解读是：研究只有留下可复用的问题定义、基线、失败、工具、概念和可被后续工作扩展或推翻的假设时，才真正可累积。

- Image: https://yingwen.io/art/yingwen-research-slogans/20-cumulative-science.png
- 原始表述: 做可累积的科学，不做不可复用的演示。

## AI 抽象概念研究

早期作品围绕从经验中学习、Rich Sutton 的苦涩教训、时间抽象与近似方法展开。

### 马列维奇 · AI从经验中学习

画面中心巨大的黑色几何体象征AI模型本身——一个深邃的"黑箱"。围绕它漂浮的红、黄、蓝碎片代表原始数据与零散经验，不断撞击、融入并重塑中心。贯穿画面的红色对角线象征梯度下降——模型在混乱的数据海洋中寻找最优解。至上主义中黑色方块是一切的起点，在这里它成为认知的诞生地。

- Image: https://yingwen.io/art/malevich-ai-learning-from-experience.png

### 蒙德里安 · AI从经验中学习

黑色网格象征AI的底层架构与算法约束——所有经验都必须填充到这个"逻辑网格"中。左侧孤立的红黄蓝小方块是原始数据点；右侧紧密互锁的彩色集群是训练完成的知识结构。构图从稀疏到密集，讲述了数据被"消化"为结构的过程。蒙德里安追求剥离表象、寻找底层逻辑——这正是AI通过特征提取和降维所做的事。

- Image: https://yingwen.io/art/mondrian-ai-learning-from-experience.png

### 康定斯基 · AI从经验中学习

一场从混沌到光明的精神之旅。左下角充斥着灰暗斑点和漩涡——未分化的原始数据。视线自然流向右上角发光的太阳状结构——这是训练完成的智慧核心。连接两端的"河流"中，形状变亮、线条连接圆圈，隐喻了特征提取与模式识别。锋利的黑色直线切入柔软的有机色块——冷峻的数学法则驯服庞杂的经验数据。这是关于"涌现"的故事。

- Image: https://yingwen.io/art/kandinsky-ai-learning-from-experience.jpg

### 马列维奇 · 苦涩的教训 — 泛化假设

视觉化Rich Sutton的泛化假设："未来将重现过去。"左下方大黑方块象征"过去"——积累的经验与知识基石。右上角小黑方块象征"未来"——形状和本质与过去完全相同，只是尚未完全展开。连接两者的红色对角线是泛化假设本身——因为两者本质相似，学到的经验可以迁移到未来。黄色圆点是智能体——处于过去与未来之间的观察者，依据泛化法则行动。

- Image: https://yingwen.io/art/malevich-the-bitter-lesson-1.png

### 马列维奇 · 苦涩的教训

压倒性的黑色十字架象征"算力"——粗暴但有效的规模。漂浮在周围的黄色、白色小碎片是"人类的启发式知识"——研究者试图手动加入AI的那些"聪明的小技巧"。在巨大的算力面前，这些人造规则显得琐碎、脆弱。构图稳固近乎压迫，传达历史的必然性。Sutton的教训是"苦涩"的——承认人类的思维并非构建AI的最佳蓝图，这是一颗难以下咽的药丸。

- Image: https://yingwen.io/art/malevich-the-bitter-lesson-2.png

### 马列维奇 · 单步陷阱

一个完美的小黑色正方形——理想化的"单步预测"——开启向右上方延伸的轨迹。但这条线很快崩溃：后续方块倾斜、散乱、漂移，撞入由红色、黑色和黄色碎片组成的爆炸性混乱云团——复合误差的指数级积累与随机世界的可能性之树。一个纯红色大矩形大胆地横跨整个画面，从起点直接跳跃到远端，完全绕过中间的混乱。这就是"时间抽象"——跳过步骤的能力，用选项和子目标而非逐步预测来思考。

- Image: https://yingwen.io/art/malevich-one-step-trap.png

### 拥抱近似

不完美解法之美。近似不是失败——它是通往大规模智能的唯一路径。精确方法在维度诅咒面前粉碎；近似方法弯曲、流动、生存。

- Image: https://yingwen.io/art/abstract-embracing-approximation.png

### 苦涩的教训

Rich Sutton核心论点的纯粹抽象表达：利用算力的通用方法终将最为有效。教训是苦涩的，因为它告诉我们——人类的聪明才智没有我们以为的那么重要。

- Image: https://yingwen.io/art/abstract-the-bitter-lesson.png
