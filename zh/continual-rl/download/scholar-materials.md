# 研究者的论文、课程、讲座与随笔

材料按教材问题归类。每项说明研究问题、核心内容、阅读方法和适用范围，并提供作者原始入口。

## Richard Sutton：个人主页、完整发表与讲座索引

个人主页与发表目录 · 跨时期 · 作者维护的原始索引

### 研究问题

从单个算法到完整智能体，哪些问题一直延续，哪些研究条件发生了变化？

### 内容与作用

作者主页将研究、授课、讲座、软件与 Incomplete Ideas 随笔放在一起。发表目录可沿 TD、Dyna、options、预测知识、步长适应、可塑性与架构向前追踪；其研究跨越多个机构，近期主页也分别说明 Oak Lab、Alberta 与 Openmind 的角色。

### 阅读与思考

先沿一个问题读两代原文，例如 TD→Horde 或 Dyna→option 模型，再回到课程与讲座理解问题选择。每次保留论文日期、作者与原文机构。

### 适用范围

个人目录包含独著、合作研究和单列的他人文献；收录关系不等于作者关系。历史工作及 Oak Lab 工作不能统一归为 Openmind 成果。

- [个人主页](http://incompleteideas.net/)
- [发表目录](http://incompleteideas.net/publications.html)
- [讲座目录](http://incompleteideas.net/Talks/Talks.html)

---

## Sutton 与 Barto：Reinforcement Learning: An Introduction

课程与讲座 · 1998／2018 · 作者提供的教材与配套入口

### 研究问题

预测、控制、规划与函数逼近怎样共享同一个交互问题表述？

### 内容与作用

教材从 bandit 与有限 MDP 建立目标、价值和策略概念，随后讲 MC、TD、多步方法、资格迹、规划、近似与策略梯度。它提供理解持续智能体各模块所需的共同语言，也让读者能辨认哪些经典分析依赖固定问题或固定表示。

### 阅读与思考

先用表格问题推导并实现价值预测和控制，再将同一个更新改为函数逼近。阅读持续学习章节时，把固定策略、分布与表示的假设显式列出。

### 适用范围

经典 RL 的系统基础不等于持续学习的完整解法。书中的收敛或策略改善结论必须连同其假设使用；Alberta RL MOOC 是 RLAI 与 Martha、Adam White 团队的课程，不应标成 Sutton 独立主讲。

- [作者教材主页](http://incompleteideas.net/book/the-book.html)
- [Alberta RL MOOC](https://www.coursera.org/specializations/reinforcement-learning)

---

## Learning to Predict by the Methods of Temporal Differences

论文 · 1988 · Machine Learning 原始论文，作者提供含勘误扫描版

### 研究问题

最终结果尚未出现时，怎样利用相邻预测的差异学习？

### 内容与作用

TD 把后继预测与即时信号组合成当前预测的学习目标，形成可随经验到达而执行的更新。理解其引导作用，是进入资格迹、GVF、Dyna 和 actor–critic 的共同起点。

### 阅读与思考

在小型随机游走上并列写出 MC 与 TD 的目标，固定数据轨迹比较更新发生的时刻；随后追问 bootstrap 的偏差和计算收益。

### 适用范围

原论文中的结果不能直接外推到任意非线性、离策略和不断变化的表示。预测有效也不自动意味着控制策略得到改善。

- [含勘误的原始论文](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
- [作者发表记录](http://incompleteideas.net/publications.html#TD_paper)

---

## Dyna：把学习、规划与行动接到同一条经验流

论文 · 1991 · 原始架构论文

### 研究问题

一次真实交互除了改变价值，还能怎样支持后续决策？

### 内容与作用

Dyna 让真实转移更新价值与环境模型，再从模型产生额外备份。它把直接经验学习与规划统一为作用于价值的更新，同时留下搜索控制问题：有限计算应该在哪些状态和动作上使用？

### 阅读与思考

画出真实转移、模型学习、模拟转移和价值备份四个接口。在 Blocking Maze 中分别记录真实交互数与规划备份数，观察模型陈旧时的代价。

### 适用范围

更多规划只有在模型与采样方式合适时才有用。固定真实交互却放开模型调用量，不能说明总计算效率更高。

- [Dyna, an Integrated Architecture for Learning, Planning and Reacting](http://incompleteideas.net/papers/sutton-91b.pdf)

---

## Between MDPs and Semi-MDPs：时间抽象的 options 框架

论文 · 1999 · Artificial Intelligence 原始论文；与 Doina Precup、Satinder Singh 合作

### 研究问题

怎样让一个持续若干步的行为，成为学习和规划可以调用的对象？

### 内容与作用

Option 用启动集合、内部策略和终止条件定义时间扩展行为。执行时长不固定，因此价值与后果模型必须处理随机持续时间。框架也使研究者能分别讨论技能的发现、学习、选择与中断。

### 阅读与思考

在 Four Rooms 中逐项实现一个到达门口的 option，检查累计奖励与终点模型。改变终止条件后，重新计算规划目标和实际用时。

### 适用范围

定义 options 并不自动发现有用技能。技能预训练、模型学习和选择的成本都应计入比较；长期收益也取决于目标变化。

- [原始论文](http://incompleteideas.net/papers/SPS-aij.pdf)

---

## IDBD：在线调整每个权重的学习步长

论文 · 1992 · Adapting Bias by Gradient Descent 原始论文

### 研究问题

一个特征应当学得更快还是更慢，能否由经验决定？

### 内容与作用

IDBD 为不同权重维护可适应的步长，通过学习过程的敏感度调整更新尺度。这为后来的在线元梯度与持续学习步长适应提供早期实例：优化对象包含学习规则，而非只有预测权重。

### 阅读与思考

从线性预测推导步长的参数化及敏感度递推。在无关特征和漂移目标上比较统一步长、逐特征固定步长和自适应步长。

### 适用范围

早期线性预测机制不能不经推导直接替换深度 actor–critic 的学习率。元步长、截断与稳定化仍需说明。

- [Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta](http://incompleteideas.net/papers/sutton-92a.pdf)

---

## The Alberta Plan：从增量学习到完整智能体的研究路线

研究计划 · 2022 · 研究路线论文；与 Michael Bowling、Patrick Pilarski 合作，使用 v1

### 研究问题

从哪个最小学习问题开始，才能逐步研究状态、技能、模型和规划之间的依赖？

### 内容与作用

Alberta Plan 按研究能力逐步构建智能体，并强调增量、持续、资源有限和经验驱动的学习。Openmind 官网将其列为研究起点，明确允许路线随证据改变。它适合用来发现模块之间尚未解决的接口。

### 阅读与思考

先给每个阶段写出输入、持久状态、每步更新和可测输出，再为相邻两个阶段设计最小联合实验。对照教材架构章节检查新增机制需要哪些证据。

### 适用范围

这是一份问题驱动的研究路线。阶段之间的逻辑衔接与整体愿景，不等于全部阶段已有可复现的成功实现。

- [Alberta Plan v1](https://arxiv.org/abs/2208.11173v1)
- [Openmind 研究方向](https://www.openmindresearch.org/)

---

## Loss of Plasticity in Deep Continual Learning

论文 · 2024 · Nature；Dohare 等与 Sutton 合作

### 研究问题

已有网络即使继续更新，为什么仍可能越来越难学会新问题？

### 内容与作用

论文在长时间、持续变化的学习问题中检验深度网络的学习能力下降，并研究包括 continual backpropagation 在内的干预。新问题上的学习速度与旧知识的保持是不同评价对象。

### 阅读与思考

把训练年龄与环境变化拆开，比较老网络、新网络和局部特征替换。再检查优化器状态、初始化与特征利用率是否解释差异。

### 适用范围

论文支持其任务和协议下的可塑性现象与干预结果。不能把局部重置的收益推广为任意长期控制环境中的保证，也不能用不遗忘替代能继续学习的证据。

- [Nature 原文](https://www.nature.com/articles/s41586-024-07711-7)

---

## Verification, the Key to AI：让知识能够由智能体自行检验

Blog · 2001 · 作者研究随笔

### 研究问题

当知识错了，需要人来发现，还是智能体可以从后续经验中发现？

### 内容与作用

文章把自我验证能力视为知识系统扩展的重要条件。把这一立场带回 RL，可以具体追问每条知识对应什么可观察结果、何种条件与何时收到的反馈；这与预测知识和模型学习构成自然联系。

### 阅读与思考

选择一条常识和一条 GVF，分别写出它们的检验程序。对无法由当前交互验证的主张，说明缺少何种观察或行动权限。

### 适用范围

这是研究原则与问题诊断，不是证明所有知识都能由单个标量预测表达的定理。可验证也不自动意味着对当前决策有用。

- [作者原文](http://incompleteideas.net/IncIdeas/KeytoAI.html)

---

## The Bitter Lesson：可扩展的搜索与学习

Blog · 2019 · 作者历史评论与研究立场

### 研究问题

增加计算预算时，方法能继续发现有效结构，还是只能重复设计者已写入的知识？

### 内容与作用

Sutton 从棋类、语音和视觉等历史例子提出偏向可扩展搜索与学习的研究立场。对持续 RL，它启发的具体问题是：经验和计算增长能否持续改善状态、知识和行为，而非只扩大一个静态训练过程。

### 阅读与思考

选一个含手工先验的方法和一个可扩展学习方法，明确在哪个资源轴上比较。分别画数据、计算和工程投入变化后的结果，避免只比单个预算点。

### 适用范围

历史论证不能替代特定算法的实验，也不意味着有限预算下所有结构先验都无效。持续学习还需要研究不可重置的交互和在线计算限制。

- [作者原文](http://incompleteideas.net/IncIdeas/BitterLesson.html)

---

## The One-Step Trap：何时应直接学习时间抽象的预测？

Blog · 2024 · 作者立场文章

### 研究问题

用一步模型反复展开，与直接学习一个长期后果，分别承担什么误差和计算成本？

### 内容与作用

文章批评只学习一步预测、再由展开获得全部长期知识的习惯，强调误差累积和分支计算，并提出以 options 与 GVF 学习时间抽象模型的方向。

### 阅读与思考

在同一个带噪模型上比较一步 rollout、直接多步预测和 option 后果模型。固定模型调用预算，同时报告长期预测误差与闭环控制收益。

### 适用范围

原文是立场性论述，不能推导出所有一步模型都不可用。模型精度、规划时域、近似采样方式和任务结构都会改变比较。

- [作者原文](http://incompleteideas.net/IncIdeas/OneStepTrap.html)

---

## Dynamic Deep Learning：让深度学习在运行中继续发生

课程与讲座 · 2024-10-30 · Imperial College London 讲座；作者目录提供讲义与视频

### 研究问题

训练完成后的网络，在继续交互时还能否保持学习能力？

### 内容与作用

讲座围绕持续学习中的可塑性损失和动态表示组织问题，连接长期训练诊断、特征替换与实时经验学习。它适合在读完具体可塑性实验后，用来讨论完整智能体为何需要持续更新内部结构。

### 阅读与思考

先读 Nature 可塑性论文，再用讲义找出从实验现象到一般架构主张的推理步骤。把每个架构主张改写为可检验的模块消融。

### 适用范围

报告中的研究展望需要与论文已经检验的结论分开；视频和讲义也不能替代算法实现与实验协议。

- [作者讲义](http://incompleteideas.net/Talks/DDL-Imperial.pdf)
- [作者目录链接的视频](https://youtu.be/75jr5E4OzEE)
- [讲座目录](http://incompleteideas.net/Talks/Talks.html#DDL)

---

## Planning and Action Selection in Options-based Agents

课程与讲座 · 2025-08-28 · Tea-Time Talk；作者目录提供讲义与视频

### 研究问题

智能体已经启动一个技能之后，何时应继续、何时应改选其他行为？

### 内容与作用

讲座用 options 与子问题组织规划和行动选择，并强调一个 option 可以只在仍有用时保持激活。它把技能模型、规划更新和真实执行的接口放到统一视角中，适合接在基础 options 框架之后。

### 阅读与思考

对同一组 options 比较执行到终止和每步重新评估两种控制器，明确高层选择、低层策略和终止信号分别何时更新。

### 适用范围

这是架构与决策机制的讲解，不是所有中断策略都优于执行到终止的实验结论。比较时应计入重新决策与规划成本。

- [作者讲义](http://incompleteideas.net/Talks/TTT-options-2025.pdf)
- [作者目录链接的视频](https://youtu.be/eJSoV2fSab4)

---

## Sutton 高级 RL 授课资料：上海课程与 CMPUT 609

课程与讲座 · 多学期／2026 · 作者主页直接提供的课程资料入口

### 研究问题

怎样把教材推导、研究论文与持续智能体问题连成一条学习路径？

### 内容与作用

Sutton 主页列出 2026 RL Course in Shanghai 与 University of Alberta 的 CMPUT 609 — Reinforcement Learning II 资料目录。它们适合作为高级学习材料索引，与教材中的具体算法章节及原始论文配合使用。

### 阅读与思考

按文件自身的课程、日期和讲次建立阅读顺序。每学一个更新规则，先做小问题推导与程序检查，再记录它进入持续学习系统时新增的假设。

### 适用范围

共享资料目录可能继续更新，也可能包含不同学期版本；不能把目录入口当作所有讲次或完整公开视频的证明。课程名称相近时仍需按授课时间区分。

- [2026 RL Course in Shanghai](https://drive.google.com/open?id=1wtHmiM5mKKLgl4K4XCUgEKdDVXAOLD9P)
- [CMPUT 609 资料目录](https://drive.google.com/drive/folders/0B3w765rOKuKANmxNbXdwaE1YU1k?usp=sharing)
- [作者课程入口](http://incompleteideas.net/)

---

## Randy Goebel：知识表示、推理与 xAI 研究索引

个人主页与发表目录 · 跨时期 · 大学个人目录、作者旧主页与实验室发表列表

### 研究问题

一个表示除了能够预测，还能否支持推理、修正与可验证的解释？

### 内容与作用

Goebel 的研究从知识表示、溯因、假设推理和信念修正延伸到 XAI、法律与医学应用。学校当前目录说明其研究兴趣，旧个人主页保留早期问题脉络，xAI Lab 列表连接到原始论文。

### 阅读与思考

先区分知识的形式、推理操作和解释对象，再选择一个应用论文，检查它怎样定义解释正确性与评价。

### 适用范围

实验室目录含多个作者及项目，并非 Goebel 的独著清单，也不是 Openmind 的机构成果列表。其与持续 RL 的联系主要在知识、模型和解释接口。

- [University of Alberta 当前目录](https://apps.ualberta.ca/directory/person/rgoebel)
- [个人学术主页](https://webdocs.cs.ualberta.ca/~goebel/)
- [xAI Lab 发表列表](https://sites.ualberta.ca/~amiixai/publications.html)

---

## Explainable AI: The New 42?

论文 · 2018 · CD-MAKE 会议论文；Goebel 与合作者

### 研究问题

解释为什么不能只靠展示模型参数或一张显著性图？

### 内容与作用

文章把 XAI 放到溯因和专家系统的长期历史中，讨论现代学习系统为何需要显式的表示与推理工具来说明结果。它使“知道模型怎么算”与“解释某个结果为何成立”的区别更清楚。

### 阅读与思考

选择一个预测错误，分别写出机制说明、支持证据和对使用者有意义的解释；再问哪些内容可以通过新实验检验。

### 适用范围

这是关于解释要求的研究论述，不提供持续 RL 更新规则，也不能仅凭可解释性主张推出闭环控制可靠。

- [出版社原文](https://link.springer.com/chapter/10.1007/978-3-319-99740-7_21)
- [会议提供的作者稿](https://2018.cd-make.net/wp-content/uploads/2018/09/GOEBEL-et-al-2018-Explainable-AI-the-new-42.pdf)

---

## A Multi-component Framework for the Analysis and Design of XAI

论文 · 2020／2021 · 2020 预印本；实验室目录另列 2021 Machine Learning and Knowledge Extraction 期刊版

### 研究问题

怎样在设计解释系统之前，把解释需求写成可评价的对象？

### 内容与作用

多组件框架从解释的历史与需求出发，组织不同层次的 XAI，强调可复现、可测量、因果及面向解释接收者的问题。它提供一套分析维度，帮助区分模型解释、结果解释与评价要求。

### 阅读与思考

给一个持续更新的 agent 做解释设计：标明当前预测、解释接收者、证据、模型版本与检验方式，再用干预测试解释是否追踪了实际决策依据。

### 适用范围

框架不是对所有解释方法的统一正确性保证。它到持续 RL 的迁移属于教学分析，需要另外检验非平稳状态和模型更新带来的变化。

- [预印本原文](https://arxiv.org/abs/2005.01908)
- [实验室期刊版索引](https://sites.ualberta.ca/~amiixai/publications.html)

---

## Goebel 在 AISUM 2020：从 AI 理论走向自主系统

课程与讲座 · 2020-10-20 · Amii 官方介绍与会议录像入口

### 研究问题

可靠表现来自智能体自身，还是来自基础设施和受控运行条件？

### 内容与作用

官方介绍列出理论到应用、自动驾驶、智能体与基础设施之间的取舍，以及 XAI 的角色。这些议题帮助持续 RL 学习者辨认完整系统中哪些能力来自学习器，哪些由环境设计提供。

### 阅读与思考

把一个自主系统分为学习器、感知与执行、基础设施和人工支持；删去其中一项，预测性能会怎样变化，并设计对照。

### 适用范围

会议报告提供研究视角，不能替代某个自动驾驶系统或 CRL 算法的实测结果。

- [Amii 官方报告介绍](https://www.amii.ca/updates-insights/randy-goebel-aisum-summit)
- [会议录像，从官方指定时间开始](https://youtu.be/WRyk87zYAsc?t=4342)

---

## Goebel：神经符号方法与基础模型的形式化

课程与讲座 · 2025 · HKUST Frontiers Forum on AI + Formal Methods 官方讲座摘要

### 研究问题

把多个推理与学习模块接在一起后，怎样定位错误并检验可靠运行？

### 内容与作用

报告主张根据应用对正确性的要求组合不同形式体系，讨论逻辑、概率、RL 与 transformer 等方法的接口。其关注点是调试和可靠性的基础，而非把某一种模型当作所有问题的统一答案。

### 阅读与思考

从摘要中抽取一个可检验问题：当状态估计正确但决策错误时，哪些模块可以被独立替换，怎样判断修复来自何处？

### 适用范围

来源是报告题目与摘要，适合标引问题与立场；它不是完整论文、课程录像或新算法实证。

- [主办方讲座页面](https://cse.hkust.edu.hk/ai-formal/2025fm/randy.html)

---

## Goebel 授课索引：形式系统、逻辑与计算中的证明

课程与讲座 · 按学期更新 · University of Alberta 官方课程目录；2026 秋季列为 CMPUT 272 主讲之一

### 研究问题

怎样把自然语言中的研究主张转成具有假设和反例的精确命题？

### 内容与作用

CMPUT 272 涵盖集合、函数、命题与谓词逻辑、证明系统、归纳和程序正确性。它是推理与算法分析的基础索引，可帮助读者检查持续学习论文的量词、条件和证明对象。

### 阅读与思考

选择一个教材结论，显式写出“对所有”与“存在”的对象，再构造违反假设的最小反例。算法实现检查与数学命题检查应分别开展。

### 适用范围

官方目录说明课程主题与任课关系，不代表公开了全部讲义、作业或录像；它也不是一门持续 RL 专门课程。

- [个人授课目录](https://apps.ualberta.ca/catalogue/instructor/rgoebel)
- [CMPUT 272 官方课程说明](https://apps.ualberta.ca/catalogue/course/cmput/272)

---

## Kris De Asis：个人主页、发表目录与研究经历

个人主页与发表目录 · 2012–2026 · 作者维护的个人目录；早年工作与现任机构分开理解

### 研究问题

多步预测、目标定义与实时机器人学习，怎样构成一条研究主线？

### 内容与作用

目录连接机电工程与运动学习背景、Alberta 时期的增量价值学习，以及 2024 年起的 Openmind 工作。早期论文通常署 University of Alberta 等机构，不能因作者当前任职而追溯归属 Openmind。

### 阅读与思考

先看研究经历，再沿 Q(σ) → 控制变量 → 固定时域 → 价值感知校正 → 时间离散化阅读；2026 年论文用正式会议记录确定版本。

### 适用范围

个人目录是发现入口，不是完整性保证；任职、共同作者和论文机构署名是三个不同字段。

- [个人主页](https://kris.pengy.ca/)
- [研究与发表目录](https://kris.pengy.ca/research)
- [作者代码目录](https://github.com/MeepMoop)

---

## Arsalan Sharifnassab：个人主页与发表目录

个人主页与发表目录 · 2012–2026 · 作者主页；较早目录中的投稿状态须由正式论文记录更新

### 研究问题

扰动稳定性和受限优化，如何延伸为在线价值估计与学习规则适应？

### 内容与作用

其历史工作包括网络调度、混合动力系统、稀疏恢复和通信受限分布式优化；后续转向 TD 优化、逐特征步长、元优化和流式 RL。旧目录把 MetaOptimize、SwiftTD 等写作投稿，不代表当前发表状态。

### 阅读与思考

先区分研究问题的约束：通信位数、轨迹扰动、Bellman 残差和在线元目标不是同一个损失；再读本库的正式 PMLR、RLJ 与 TMLR 入口。

### 适用范围

早年 Sharif/MIT 与 Alberta 工作不应统称 Openmind 成果；尚在准备中的标题也不等于论文结果。

- [个人主页](https://sites.google.com/view/sharifnassab/home)
- [作者发表目录](https://sites.google.com/view/sharifnassab/publications)
- [Openmind 当前研究目录](https://www.openmindresearch.org/research)

---

## Multi-Step Reinforcement Learning: A Unifying Algorithm（Q(σ)）

论文 · 2017 预印本；2018 AAAI · AAAI 2018 正式论文；De Asis、Hernandez-Garcia、Holland、Sutton

### 研究问题

备份必须全采样或全取期望吗？

### 内容与作用

σ 在每一步调节采样与期望备份，连接多步 Sarsa 与 Tree-backup；一步纯期望端点对应 Expected Sarsa。多步长度 n 与 σ 控制不同的设计维度，不能合称一个时域参数。

### 阅读与思考

从一步混合 TD 误差展开到多步回报，再比较固定和动态 σ 的实验。配合本章控制变量节检查：降低采样噪声，不等于消除初始价值误差。

### 适用范围

论文实验重点为 on-policy；中间 σ 在所测任务有效，不是普遍最优或通用深度 off-policy 稳定性定理。未确认论文专属作者实验仓库。

- [AAAI 正式论文](https://cdn.aaai.org/ojs/11631/11631-13-15159-1-2-20201228.pdf)
- [作者预印本与版本](https://arxiv.org/abs/1703.01327)

---

## Per-decision Multi-step Temporal Difference Learning with Control Variates

论文 · 2018 · UAI 2018 正式论文；De Asis 与 Sutton

### 研究问题

重要性采样放大多步噪声时，能否保留真实路径又借助已知期望？

### 内容与作用

在每个决策点加入条件均值为零的校正，将采样尾部写成目标策略期望加上重要性加权的残差。动作价值版本即使 on-policy 也可能有控制变量；状态价值版本在 on-policy 时相应项消失。

### 阅读与思考

重点读式 (21) 的动作价值递推，再读状态价值与 λ-return 的后向关系。自己枚举两动作例子，分别检查均值保持和方差变化。

### 适用范围

控制变量系数与价值估计精度影响效果；无偏校正不消除原有 bootstrap 偏差，也不提供任意非线性函数逼近的收敛保证。未确认论文专属作者仓库。

- [UAI 论文](http://auai.org/uai2018/proceedings/papers/282.pdf)
- [作者预印本](https://arxiv.org/abs/1807.01830)

---

## Predicting Periodicity with Temporal Difference Learning

论文 · 2018 预印本；2019 RLDM 摘要 · De Asis、Brendan Bennett、Sutton；预印本与会议扩展摘要

### 研究问题

价值预测能否表示未来信号的周期结构，而不只表示一个时间尺度的总量？

### 内容与作用

将折扣扩展为复数，借助信号处理视角在线估计感兴趣信号的离散傅里叶变换。它扩展的是预测知识的语义，不是把复数直接当作控制任务的好坏排序。

### 阅读与思考

先复习普通折扣加权和，再检查复数旋转因子如何使不同频率发生相长或抵消。与 GVF 的累积信号、延续规则分开比较。

### 适用范围

属于特定预测构造；周期检测不是通用非平稳适应能力。RLDM 摘要不应标成完整同行评审长论文；未确认专属实验仓库。

- [作者预印本](https://arxiv.org/abs/1809.07435)
- [RLDM 扩展摘要集（第 108 页）](https://cs.brown.edu/people/mlittman/ftp/extendedabstracts.pdf#page=108)

---

## Fixed-Horizon Temporal Difference Methods for Stable Reinforcement Learning

论文 · 2019 预印本；2020 AAAI · AAAI 2020；De Asis、Chan、Pitis、Sutton、Daniel Graves

### 研究问题

不让一个预测器从自身 bootstrap，能否改变离策略不稳定的结构？

### 内容与作用

为不同剩余时域建立价值预测：h 步预测从 h−1 步预测 bootstrap，形成按时域排列的依赖链。代价是额外预测器；n 步备份、并行计算和共享参数用于降低开销。

### 阅读与思考

先画 h=0、1、2 的依赖图，再读线性条件下的矩阵与收敛分析。随后单独检查深度实验的共享表示和 replay 协议。

### 适用范围

固定时域目标不等于无限时域折扣目标；理论假设不能无条件移到共享非线性网络。此论文不能据标题视为严格无样本存储算法。未确认专属作者仓库。

- [AAAI 正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/5784)
- [论文全文](https://ojs.aaai.org/index.php/AAAI/article/download/5784/5640)
- [作者预印本](https://arxiv.org/abs/1909.03906)

---

## Inverse Policy Evaluation for Value-based Sequential Decision-making

论文 · 2020 预印本；2022 RLDM 摘要 · Alan Chan、Kris De Asis、Sutton；预印本与 RLDM 扩展摘要

### 研究问题

近似价值函数未必属于任何策略，直接贪心改善依据的是什么？

### 内容与作用

逆策略评价从给定价值估计反求相容的策略，探索价值到行为的另一种映射。它不同于逆强化学习：待求对象是策略，不是解释专家行为的奖励。

### 阅读与思考

先读为何近似价值迭代可能离开策略价值集合，再读反求策略的增量过程；把 2020 完整预印本与 2022 摘要分开。

### 适用范围

映射可行性及所测控制结果，不等于任意逼近价值都能获得全局策略改善保证；未确认专属作者实验仓库。

- [完整预印本](https://arxiv.org/abs/2008.11329)
- [RLDM 2022 正式摘要与节目（1.153）](https://rldm.org/wp-content/uploads/2022/05/programme_final_27_May-2022.pdf)

---

## Value-aware Importance Weighting for Off-policy Reinforcement Learning

论文 · 2023 · CoLLAs 2023，PMLR 232:2086–2112；De Asis、Eric Graves、Sutton

### 研究问题

校正一个特定价值的期望，是否一定要使用对所有随机变量通用的概率比？

### 内容与作用

把各动作的当前价值也纳入权重构造，在保持其目标期望及权重期望约束下优化权重方差。Sparho 是该框架的实例，随后被带入多步与资格迹预测。

### 阅读与思考

读第 3 节的两条约束与拉格朗日推导，再读多步校正。比较“保持当前 Q 的期望”与“对任意未知回报都无偏”的不同要求。

### 适用范围

论文主要校正更新目标而非状态访问分布；Emphatic TD 部分是初步实验。估计值依赖与退化情形需要按原文处理，不能把方差结果无限外推。未确认专属作者仓库。

- [CoLLAs 正式记录](https://proceedings.mlr.press/v232/de-asis23a.html)
- [全文与推导](https://arxiv.org/html/2306.15625v1)

---

## An Idiosyncrasy of Time-discretization in Reinforcement Learning

论文 · 2024 · RLC 2024 / RLJ；De Asis 与 Sutton

### 研究问题

物理时间被切成步之后，即时奖励的折扣时间戳是否一致？

### 内容与作用

从连续时间目标出发，指出常见离散写法可能混用奖励与折扣的取样端点。固定时长下差异可能只是全局尺度，动作或状态依赖的可变时长下却可能改变策略偏好。

### 阅读与思考

跟着连续积分到离散求和的转换逐项核对时间戳；再区分奖励率与区间累计奖励。用同一物理轨迹改变采样频率作一致性检查。

### 适用范围

并非否定所有离散 MDP；离散问题可以有自洽的独立定义。论文针对连续目标的近似，不自动解决计算延迟、异步观测或可变步长数值积分。未确认专属作者代码。

- [RLC 正式记录](https://rlj.cs.umass.edu/2024/papers/Paper164.html)
- [作者最终预印本](https://arxiv.org/html/2406.14951v2)

---

## Extending Differential Temporal Difference Methods for Episodic Problems

论文 · 2026 · RLC 2026 / RLJ；Kris De Asis、Mohamed Elsayed、Jiamin He

### 研究问题

把差分 TD 的中心化优势带到回合任务时，怎样避免悄悄改变目标？

### 内容与作用

直接每步减常数会按回合长度改变回报。论文利用终止处的特殊修正和价值偏置参数化，形成与原回合预测目标相容的更新；关键不是把 episodic 任务强行改成平均奖励任务。

### 阅读与思考

先自己构造长短路径偏好翻转，再读 potential-based 解释、终止值约束与 bias 更新；线性证明和深度实验分开读。

### 适用范围

固定偏置下的目标等价性不等于所有动态偏置更新都全局收敛；γ=1 的有限回合情形也不是平均奖励最优性结论。未确认论文专属作者仓库。

- [RLC 正式记录](https://rlj.cs.umass.edu/2026/papers/Paper33.html)
- [作者论文](https://arxiv.org/abs/2605.04368)

---

## Intentional Updates for Streaming Reinforcement Learning

论文 · 2026 · ICML 2026，PMLR 306:110478–110496

### 研究问题

参数步长相同，为什么价值或策略的实际变化却很不相同？

### 内容与作用

Sharifnassab、Elsayed、De Asis、Mahmood 与 Sutton 先指定输出上的意图，再以局部导数求近似所需步长。TD 关注误差的相对缩减，策略更新关注概率变化尺度，并结合资格迹和对角缩放。

### 阅读与思考

先读一阶输出变化与步长求解，再对照资格迹的累计尺度和代码中的统计量。与 MetaOptimize 比较：一个控制本次输出变化，另一个利用后续损失调整学习规则。

### 适用范围

局部一阶近似、样本依赖缩放和共享参数耦合仍存在；典型更新尺度不等于逐样本硬上界，局部 KL 近似不等于精确 trust region。仍有元尺度参数。

- [ICML 正式记录](https://proceedings.mlr.press/v306/sharifnassab26a.html)
- [作者论文](https://arxiv.org/abs/2604.19033)
- [作者代码](https://github.com/sharifnassab/Intentional_RL)

---

## A Unified View of Multi-step Temporal Difference Learning

论文 · 2018 · University of Alberta 硕士论文；作者目录与机构存档入口

### 研究问题

怎样把看似分散的多步 TD 更新放进统一的备份框架？

### 内容与作用

这篇学位论文是 Kris 早期多步 TD 研究的系统入口，可与同年的 Q(σ) 和逐决策控制变量论文建立对应关系。

### 阅读与思考

先读已公开的两篇会议论文，再用学位论文的定义与章节组织核对统一视角。

### 适用范围

此条提供作者确认的学位论文入口，不把尚未逐章复核的内容列成额外发现；学位论文和会议论文有内容重叠。

- [Alberta 学位论文存档](https://era.library.ualberta.ca/items/0d6b663a-4430-4c4a-8960-f1ae48a380c1)
- [作者目录](https://kris.pengy.ca/research)

---

## Explorations in the Foundations of Value-based Reinforcement Learning

论文 · 2024 · University of Alberta 博士论文；作者目录与机构存档入口

### 研究问题

价值预测的时域、离策略校正和行为导出，分别改变了什么？

### 内容与作用

博士论文为阅读其价值学习基础研究提供总入口。应由具体章节追到对应公开论文，而不是把论文集式研究压缩成一种万能稳定算法。

### 阅读与思考

围绕预测目标、备份依赖、估计分布、价值到策略四个问题建立章节索引，再与正式论文版本核对。

### 适用范围

这里只确认题名、年份和学位类型；不据标题推断每章结果。2024 年学位研究不应整体改标为 Openmind 机构成果。

- [Alberta 学位论文存档](https://era.library.ualberta.ca/items/e2d7890f-9a86-4bc7-b05d-0108e7a661f2)
- [作者目录](https://kris.pengy.ca/research)

---

## MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters

论文 · 2024 预印本；2025 ICML · ICML 2025，PMLR 267:54300–54325；Sharifnassab、Salehkaleybar、Sutton

### 研究问题

后来的误差怎样归因于较早的步长选择，而不只给当前梯度做归一化？

### 内容与作用

从折扣的未来损失出发，以前向敏感度累积过去元参数对当前内层状态的影响。框架联合表示内层优化器、元优化器及敏感度；2×2、L 和 Hessian-free 等近似改变所保留的导数路径。

### 阅读与思考

先分清损失时间与元参数时间，再读完整状态递推，最后进入近似。比较非平稳逐样本 CIFAR-100 与常规批量图像/语言训练，勿混用协议。

### 适用范围

有限元步长下的前后向关系和近似均有边界；不是去掉全部超参数。论文保留 ImageNet 分块步长未改善等结果，未提供持续 actor–critic 收益保证。

- [ICML 正式记录](https://proceedings.mlr.press/v267/sharifnassab25a.html)
- [最终作者版本](https://arxiv.org/html/2402.02342v6)
- [作者代码](https://github.com/sabersalehk/MetaOptimize)

---

## Step-size Optimization for Continual Learning

论文 · 2024 · 公开预印本；Degris、Javed、Sharifnassab、Liu、Sutton

### 研究问题

归一化更新与依据学习结果优化步长，是否是同一件事？

### 内容与作用

比较归一化型方法和元梯度型步长适应，强调在持续变化的小问题上检验跟踪能力。它为理解 IDBD、现代自适应优化器与后来 MetaOptimize 的关系提供问题背景。

### 阅读与思考

先读两类方法优化或控制的对象，再看受控任务中信号尺度和目标变化。把组合二者的动机与已完成的比较实验分开。

### 适用范围

小型持续学习任务不等于深度 RL 全面验证；混合机制的研究主张不等于已证明普遍优越。未确认独立论文专属作者代码入口。

- [作者预印本](https://arxiv.org/abs/2401.17401)

---

## SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning

论文 · 2024 · RLC 2024 / RLJ 2:840–863；Khurram Javed、Sharifnassab、Sutton

### 研究问题

资格迹传播信用时，逐特征步长怎样既适应又避免过大的在线更新？

### 内容与作用

SwiftTD 将 true-online TD(λ)、步长适应和更新控制组合成在线预测方法，并在 Atari Prediction Benchmark 等任务比较。学习的是给定行为下的预测，不是通过 Atari 分数证明控制能力。

### 阅读与思考

先对照 true-online TD 的修正项，再标出步长更新和稳定化机制；用作者交互演示观察 λ、步长与跟踪误差。

### 适用范围

线性像素特征或神经网络最后一层预测，与端到端持续深度控制不同；必须保留固定策略和表征条件。

- [RLC 正式记录](https://rlj.cs.umass.edu/2024/papers/Paper111.html)
- [作者全文](https://khurramjaved.com/papers/SwiftTD.pdf)
- [作者代码](https://github.com/kjaved0/swifttd)
- [作者交互演示](https://khurramjaved.com/swifttd.html)

---

## Toward Efficient Gradient-Based Value Estimation（RAN / RANS）

论文 · 2023 · ICML 2023，PMLR 202:30827–30849；Sharifnassab 与 Sutton

### 研究问题

Bellman 残差目标可以下降，为什么价值估计仍可能非常慢？

### 内容与作用

研究均方 Bellman 误差的病态优化几何，以近似 Gauss–Newton 和近端思想构造 RAN，并用离群更新拆分形成 RANS。它聚焦残差最小化的效率，而非另设平均奖励目标。

### 阅读与思考

先辨别 MSBE、采样 TD 误差平方及 double-sampling 问题，再读曲率近似和算法 3–4 的 outlier buffer。

### 适用范围

“batch-free”不等于本教材不保留样本的严格流式：RANS 缓存并在以后处理离群样本贡献。理论与实验条件不可直接搬到任意非线性 off-policy 控制；未确认专属作者仓库。

- [ICML 正式记录](https://proceedings.mlr.press/v202/sharifnassab23a.html)
- [论文与附录](https://proceedings.mlr.press/v202/sharifnassab23a/sharifnassab23a.pdf)
- [作者预印本](https://arxiv.org/abs/2301.13757)

---

## Soft Preference Optimization: Aligning Language Models to Expert Distributions

论文 · 2024 预印本；2026 TMLR · 正式版本发表于 TMLR（05/2026）；正式作者为 Sharifnassab、Salehkaleybar、Schuurmans

### 研究问题

从偏好对学习策略时，怎样区分偏好拟合、参考策略约束和分布的软度？

### 内容与作用

SPO 以成对偏好损失和全局 KL 正则构造训练目标，并用软度参数描述专家分布；在特定偏好生成模型下分析目标策略。它可作为最大熵目标与偏好学习的桥梁。

### 阅读与思考

先读正式版本的问题假设与损失，再对照 2024 arXiv。旧稿含 Ghiassian、Kanoria 两位共同作者，不能把不同版本的作者列表与年份拼成一个记录。

### 适用范围

这是语言模型偏好对齐，不是持续在线 RL 的验证；理论依赖偏好模型等条件。Openmind 目录仍列旧预印本，正式状态以 TMLR 记录为准；未确认专属作者代码。

- [TMLR 正式记录](https://openreview.net/forum?id=EUPIcAkrSR)
- [正式论文](https://openreview.net/pdf?id=EUPIcAkrSR)
- [2024 作者预印本](https://arxiv.org/abs/2405.00747)
- [合作者所在课题组发表目录](https://eda.liacs.leidenuniv.nl/publications/)

---

## Order Optimal One-Shot Distributed Learning

论文 · 2019 · NeurIPS 2019；Sharifnassab、Salehkaleybar、Golestani；Openmind 之前

### 研究问题

每台机器只能发送一次消息时，统计优化误差受什么限制？

### 内容与作用

研究分布式统计学习的信息限制，并构造多分辨率估计器，在给定条件下达到忽略对数因子的阶最优误差。这里的 one-shot 指一次通信，不是从一条经验学会控制。

### 阅读与思考

先写清机器数、每机样本数、维数与消息预算，再对齐上下界的渐近变量；随后读 2021 JMLR 扩展。

### 适用范围

凸统计优化与通信限制不是持续非平稳 RL 的数据协议；不能用阶最优名称推出其在 RL 中最优。未确认专属作者代码。

- [NeurIPS 正式论文](https://papers.nips.cc/paper_files/paper/2019/file/018b59ce1fd616d874afad0f44ba338d-Paper.pdf)
- [作者预印本](https://arxiv.org/abs/1911.00731)

---

## One-Shot Federated Learning: Theoretical Limits and Algorithms to Achieve Them

论文 · 2021 · JMLR 22(189):1–47；Salehkaleybar、Sharifnassab、Golestani；Openmind 之前

### 研究问题

有限消息位数怎样改变可达到的估计精度？

### 内容与作用

每机观察独立同分布样本后发送 B 比特消息，服务器估计期望凸损失的最小点；MRE 在规定的预算条件下匹配误差下界到多对数因子。

### 阅读与思考

与 2019 版本比较通信预算的显式条件，并区分固定每机样本数、增加机器数与增加总数据量三个极限。

### 适用范围

这是通信受限静态统计估计，不是无任务边界的跟踪定理；独立样本和凸性条件很关键。正式页面未给出论文专属代码入口。

- [JMLR 正式记录](https://www.jmlr.org/papers/v22/19-1048.html)
- [论文全文](https://www.jmlr.org/papers/volume22/19-1048/19-1048.pdf)

---

## Order Optimal Bounds for One-Shot Federated Learning over non-Convex Loss Functions

论文 · 2021 预印本；2023 在线发表 / 2024 卷期 · IEEE Transactions on Information Theory 70(4):2807–2830；Sharifnassab、Salehkaleybar、Golestani

### 研究问题

失去凸性后，一次通信的全局估计误差怎样依赖维数与消息长度？

### 内容与作用

给出非凸损失情形的误差下界，并构造 MRE-NC 匹配机器数和样本数上的阶，揭示维数对受限通信学习的代价。

### 阅读与思考

以 2021 arXiv 为开放全文入口，逐项对照损失函数类与 B 比特消息条件；引用时区分在线年份与卷期年份。

### 适用范围

非凸函数类上的统计估计结论不是一般神经网络 SGD 的优化保证，更不是强化学习交互样本复杂度结果。未确认专属作者代码。

- [作者预印本](https://arxiv.org/abs/2108.08677)
- [期刊 DOI](https://doi.org/10.1109/TIT.2023.3344141)
- [作者发表目录（记在线年份）](https://sites.google.com/view/sharifnassab/publications)

---

## Bounds on Over-Parameterization for Guaranteed Existence of Descent Paths in Shallow ReLU Networks

论文 · 2020 · ICLR 2020；Sharifnassab、Salehkaleybar、Golestani；Openmind 之前

### 研究问题

更宽的网络在何种意义上具有通往较低损失的路径？

### 内容与作用

研究浅层 ReLU 网络损失面的下降路径与过参数化边界。其价值在于把几何上存在一条路，与具体优化器能否找到这条路分开。

### 阅读与思考

先核对网络层数、样本数与宽度的条件，再检查存在性结论的量词。用于反思“参数多就不会失去可塑性”的过强推论。

### 适用范围

不是持续任务下可塑性保持定理，也不保证梯度下降沿所证明路径运动；未确认论文专属作者代码。

- [ICLR OpenReview 记录](https://openreview.net/forum?id=BkgXHTNtvS)
- [作者提供的论文](https://drive.google.com/file/d/1FoHKnWGX4lmTL4tLwknMvndwwLgumluD/view)

---

## Sensitivity to Cumulative Perturbations for a Class of Piecewise Constant Hybrid Systems

论文 · 2018 预印本；2019 在线 / 2020 卷期 · IEEE Transactions on Automatic Control 65(3):1057–1072；Sharifnassab、Tsitsiklis、Golestani

### 研究问题

瞬时扰动可以很大时，累计扰动小能否仍保证轨迹接近？

### 内容与作用

对特定有限分片混合动力系统建立累计扰动敏感度界，连接分片线性凸势函数与网络调度流体模型。这是其研究更新稳定性之前的一条动力系统线索。

### 阅读与思考

先区分扰动的瞬时范数与累计和的范数，再核对分片数、非扩张性和边界动力学条件。

### 适用范围

结论针对指定系统类，不能直接当作任意神经网络随机梯度或 TD 轨迹的鲁棒性定理。未确认专属作者代码。

- [作者预印本](https://arxiv.org/abs/1807.08139)
- [合作者 MIT 全文](https://web.mit.edu/jnt/www/Papers/J165-20-AS-final.pdf)
- [期刊 DOI](https://doi.org/10.1109/TAC.2019.2922495)

---

## When do Trajectories have Bounded Sensitivity to Cumulative Perturbations?

论文 · 2019 · 公开预印本；Sharifnassab 与 Golestani；Openmind 之前

### 研究问题

哪些看起来温和的动力系统仍会对累计扰动高度敏感？

### 内容与作用

以充分条件、变换性质和反例刻画累计扰动敏感度；特别提醒不能把有限分片系统结论任意扩展到一般凸梯度动力系统。

### 阅读与思考

先读反例，再回看先前混合系统结果究竟使用了哪些有限性与结构条件；把这些条件做成假设检查表。

### 适用范围

“凸”并不足以推出所有轨迹稳定性质；与流式 RL 的联系是分析方法启发，而非已经验证的算法迁移。未确认专属作者代码。

- [作者预印本](https://arxiv.org/abs/1905.11746)

---

## Nonexpansive Piecewise Constant Hybrid Systems are Conservative

论文 · 2019 · 公开预印本；Sharifnassab、Tsitsiklis、Golestani；Openmind 之前

### 研究问题

分片恒定的非扩张动力学，是否隐含一个势函数？

### 内容与作用

在有限多面体分片等设定下，将非扩张混合系统与分片线性凸势函数的负次梯度流联系起来。这提供了从更新规则反推优化结构的例子。

### 阅读与思考

画二维分片与每片漂移向量，再读非扩张到保守性的推导；注意轨迹在分片边界上的定义。

### 适用范围

“保守”在此是势函数结构的数学术语，不是安全探索保证；不适用于任意学习更新。未确认专属作者代码。

- [作者预印本](https://arxiv.org/abs/1905.12361)
- [MIT 合作者全文](https://www.mit.edu/~jnt/Papers/P-19-conservative.pdf)

---

## Fluctuation Bounds for the Max-Weight Policy with Applications to State Space Collapse

论文 · 2018 预印本；2020 期刊 · Stochastic Systems 10(3):223–250；Sharifnassab、Tsitsiklis、Golestani

### 研究问题

离散排队过程偏离流体近似多少，能否由累计到达扰动控制？

### 内容与作用

为 Max-Weight 调度建立队列与流体轨迹偏差界，并据此分析状态空间塌缩的时间尺度上下界。作者个人旧目录写 2019，正式期刊在线发表为 2020。

### 阅读与思考

先识别队列长度、流体解和累计到达量，再读时间尺度结论；不要把流体缩放与实际控制周期混为一谈。

### 适用范围

这里的 state space collapse 是重载排队的数学现象，不是表示塌缩或可塑性损失；未确认专属作者代码。

- [期刊正式记录](https://pubsonline.informs.org/doi/abs/10.1287/stsy.2019.0038)
- [作者预印本](https://arxiv.org/abs/1810.09180)

---

## Jumping Fluid Models and Delay Stability of Max-Weight Dynamics Under Heavy-Tailed Traffic

论文 · 2021 预印本；2023 期刊 · Stochastic Systems 13(4):399–437；Sharifnassab 与 Tsitsiklis

### 研究问题

少数重尾到达流是否会使其他队列的平均延迟也失控？

### 内容与作用

用带跳跃的流体模型研究 Max-Weight 系统的鲁棒延迟稳定性，刻画重尾输入影响其他队列的条件。它延续累计扰动和网络调度主线。

### 阅读与思考

先读 delay stability 的期望定义和流量尾部假设，再看跳跃流体模型如何编码罕见大到达。

### 适用范围

对重尾队列的判据不是一般 RL 更新离群值的处理算法；二者可比较问题结构，不能直接转移定理。未确认专属作者代码。

- [期刊正式记录](https://pubsonline.informs.org/doi/10.1287/stsy.2023.0110)
- [作者预印本](https://arxiv.org/abs/2111.07420)
- [MIT 合作者全文](https://www.mit.edu/~jnt/Papers/P-21-JF.pdf)

---

## Design for Learning

Blog · 2025-10-23 · 作者立场文章；源于 AI Malaysia 前置研讨会报告

### 研究问题

若机器人要在运行中学习，硬件本身应该允许什么？

### 内容与作用

Kris 主张算法与身体共同设计：便于维护、可以恢复、能承受探索动作；物理时间和无人工干预的可继续性应成为问题的一部分。该主张连接连续学习与机器人实验设计。

### 阅读与思考

把三类硬件主张转成可测指标：恢复时间、维修成本、探索损伤；再读作者对冻结 sim2real 策略边界的论证。

### 适用范围

文章是研究立场和设计建议，不是这些设计原则已普遍提升学习收益的受控实验；也不主张废弃仿真。

- [作者原文](https://kris.pengy.ca/designforlearning)
- [作者博客日期索引](https://kris.pengy.ca/blog)

---

## Kris De Asis · Predicting Periodicity with Temporal Difference Learning

课程与讲座 · 2018-05-31 · Amii Tea Time Talk；主办方目录与录像

### 研究问题

折扣除了指定预测远近，还能编码频率吗？

### 内容与作用

报告以数字信号处理视角解释复数折扣的动机，适合作为周期预测论文的直观入口。

### 阅读与思考

先听问题与例子，再回论文检查复数折扣和 DFT 的条件；主办方页面给出准确报告日期。

### 适用范围

这是研究报告，不是一门完整强化学习课程；录像中的讲解不替代论文推导与实验条件。

- [主办方节目与摘要](https://amiithinks.github.io/tea-time-talks/2018/)
- [主办方链接的录像](https://www.youtube.com/watch?v=YZFS1DbglqY)

---

## TalkRL · RLC 2024 Posters and Hallways 3：时间离散化访谈

课程与讲座 · 2024-09-18 · 原播客发布页；Kris 段落 0:01–2:23

### 研究问题

连续物理目标与离散 RL 公式之间，容易忽略哪一个时间问题？

### 内容与作用

Kris 在 RLC 现场短访谈中介绍时间离散化研究。可作为论文阅读前的问题引入，后接教材中的可变时长反例。

### 阅读与思考

先听首段，再对照论文连续积分与离散回报中的奖励时间戳；节目后续段落是其他研究者，勿混归作者。

### 适用范围

两分钟口头摘要不是完整证明，也不是课程或独立算法实验。

- [TalkRL 原发布页与转录入口](https://www.talkrl.com/episodes/rlc-2024-posters-and-hallways-3)

---

## Bounded Perturbations in Hybrid Dynamical Systems with Applications to Network Scheduling

课程与讲座 · 2017-02-02 · MIT LIDS Student Conference 2017 报告记录

### 研究问题

外部输入波动怎样传递成网络调度轨迹的偏差？

### 内容与作用

主办方日程和摘要记录了 Arsalan 关于混合系统累计扰动的早期报告，可与后来 TAC 和 Stochastic Systems 论文连读。

### 阅读与思考

读主办方摘要后转到正式论文中的系统类、范数与边界条件，观察口头问题如何发展成定理。

### 适用范围

此入口确认报告身份与内容摘要，未确认可公开访问的录像；不能将其写作完整课程。

- [MIT 原日程](https://studentconf.lids.mit.edu/2017/schedule.html)
- [MIT 原摘要](https://studentconf.lids.mit.edu/2017/abstracts.html)

---

## Kris 的 Tile Coding 与 MDPy 教学实现

代码 · 作者公开代码 · 作者个人仓库；通用表示和小 MDP 工具

### 研究问题

如何用可检查的小实现，把特征表示与精确价值迭代分开？

### 内容与作用

Tile Coding 将连续输入映射到稀疏二元特征；MDPy 用显式状态、动作和转移构造小 MDP 并作价值迭代。二者适合验证教材手算例子。

### 阅读与思考

先跑 README 的最小构造，检查激活特征数及转移概率，再写自己的边界测试；留意 MDPy 示例的旧版 Python 语法。

### 适用范围

这些是作者工具，不应自动标为 Q(σ)、控制变量或后续论文的实验复现包；小 MDP 求解也不是交互式未知模型学习。

- [Tile Coding](https://github.com/MeepMoop/tilecoding)
- [MDPy](https://github.com/MeepMoop/MDPy)
- [Kanerva Coding](https://github.com/MeepMoop/kanervacoding)

---

## MetaOptimize 作者实现

代码 · 2025 · 论文链接的作者代码仓库

### 研究问题

论文中的完整框架和各级近似，实际由哪些状态变量实现？

### 内容与作用

仓库提供 MetaOptimize 优化器实现和示例，是把元状态、内层状态、敏感度与自动微分路径对应起来的入口。

### 阅读与思考

从示例中的基优化器/元优化器组合进入实现，再检查 detach、Hessian 路径、步长分块和超参数默认值是否对应所读变体。

### 适用范围

同一框架的不同近似不是同一算法；运行示例成功不证明在新 RL 协议下有效。复现应固定提交版本与任务序列。

- [作者仓库](https://github.com/sabersalehk/MetaOptimize)
- [正式论文](https://proceedings.mlr.press/v267/sharifnassab25a.html)

---

## Intentional_RL 作者实现

代码 · 2026 · ICML 论文链接的公开实验代码

### 研究问题

输出意图、资格迹与在线尺度统计，在实现里是否使用同一种归一化？

### 内容与作用

代码用于对应 Intentional TD 与 Intentional Policy Gradient 的实际更新、迹和对角缩放。应同时检查数学累计量与实现的归一化统计，不能仅凭符号名称认定完全相同。

### 阅读与思考

沿环境单步到 TD 误差、梯度、迹、尺度统计、参数写回的顺序追踪；保持 batch size、replay、终止处理和元尺度一致再做对照。

### 适用范围

作者结果是指定任务与协议下的实证；代码存在不等于所有默认设置都可直接迁移至真实机器人，也不替代延迟和稳定性测试。

- [作者实验仓库](https://github.com/sharifnassab/Intentional_RL)
- [正式论文](https://proceedings.mlr.press/v306/sharifnassab26a.html)

---

## SwiftTD 作者代码与在线预测演示

代码 · 2024 · 合作者 Khurram Javed 发布的实现与交互说明

### 研究问题

资格迹与逐特征步长组合后，预测误差和更新尺度如何变化？

### 内容与作用

演示与仓库将 SwiftTD 的机制变成可观察的在线预测过程，适合在读完 true-online TD 后逐项对照。

### 阅读与思考

先固定数据流比较误差，再改动步长或迹衰减；回到论文确认演示所省略的实验条件。

### 适用范围

演示展示机制，不是对任意非平稳序列的保证；资源由共同作者发布，不应标成 Arsalan 独立授课。

- [作者代码](https://github.com/kjaved0/swifttd)
- [作者交互演示](https://khurramjaved.com/swifttd.html)

---

## Kris De Asis · Openmind Fellowship Proposal

研究计划 · 2024 前后 · 公开 fellowship 研究计划；包含初步例子，不是完整成果论文

### 研究问题

在没有明确回合边界且计算有界的情况下，如何构建实时学习智能体？

### 内容与作用

提出持续环境、在线增量计算与增量策略梯度等方向，并用 AdaptChain 初步例子解释回合内适应的价值。其约束有助于把“在线”落实到协议。

### 阅读与思考

先抄出不能依赖的资源：完整回合轨迹、无界计算、停住世界等待更新；再把计划中的问题与后来公开论文一一对照。

### 适用范围

方向、初步图示与后续同行评审结果须分开；不能声称计划已实现完整原型 AI 或通用机器人学习。

- [Openmind 原始计划](https://www.openmindresearch.org/s/kdeasis_proposal_web.pdf)

---

## Arsalan Sharifnassab · Continual Meta Learning Proposal

研究计划 · 2024 前后 · 公开 fellowship 研究计划，不是已完成结果

### 研究问题

智能体能否从不断到来的经验改善步长及其他学习规则？

### 内容与作用

计划围绕持续元学习与步长优化展开，并讨论平均奖励价值估计等后续方向。可据此理解研究动机，但正式证据仍应落到 MetaOptimize、SwiftTD 等具体论文。

### 阅读与思考

把每个研究问题改写为可测试假设，单独列出所需资源、目标与评估协议；特别区分平均奖励方向和已发表残差优化方法。

### 适用范围

计划中的平均奖励工作不等于 RANS 已解决平均奖励问题；预期结果、初步证据与完成结果必须保持不同状态。

- [Openmind 原始计划](https://www.openmindresearch.org/s/Proposal_Arsalan_web.pdf)

---

## Meta-descent Normalization / Tabular TD in L1 Geometry：公开准备中方向

研究计划 · 2026 目录状态 · Openmind 标为 In Prep.；不是已发表论文

### 研究问题

元下降与归一化怎样结合，TD 能否从另一种局部优化几何理解？

### 内容与作用

机构目录列出 Meta-descent normalization in continual learning，以及 Tabular TD as local optimization in L1 geometry 两个准备中标题。这里只把标题作为研究问题入口。

### 阅读与思考

先学习本教材已有元梯度、归一化和 TD 更新机制；待原稿公开后再核对目标、公式、假设和证据。

### 适用范围

未公开的推导、算法、实验与结论不得由标题推测；不把 In Prep. 计作验证过的结果。

- [Openmind 研究目录与状态](https://www.openmindresearch.org/research)

---

## On the Possibility of Network Scheduling With Polynomial Complexity and Delay

论文 · 2017 · IEEE/ACM Transactions on Networking；Sharifnassab 与 Golestani；早期跨领域工作

### 研究问题

一个调度策略能否同时有可控计算复杂度和低延迟？

### 内容与作用

这一条保留作者在网络调度中的问题背景：单步计算资源和长期运行性能需要同时分析，不能只追求吞吐量或只计算一次更新的耗时。

### 阅读与思考

先从作者全文核对网络、到达过程与复杂度的变量，再区分理论可行性和某个实际调度器的表现。

### 适用范围

这里提供历史论文入口，不把网络调度结论外推成 RL 实时性保证；未确认专属作者实验仓库。

- [作者提供的全文](https://drive.google.com/file/d/1PzfC-szmY6kq2nEfm58YBf8q0sIH1G_L/view)
- [IEEE Communications Society 当期目录](https://www.comsoc.org/system/files/2018-01/Publications_Contents_Digest_2017_Dec.pdf)

---

## Invariancy of Sparse Recovery Algorithms

论文 · 2017 · IEEE Transactions on Information Theory；Kharratzadeh、Sharif-Nassab、Babaie-Zadeh；早期工作

### 研究问题

输入方程的变换会不会改变稀疏恢复算法的行为？

### 内容与作用

该论文研究稀疏恢复算法的不变性，属于其优化与信号处理背景。它提供一个有用提问方式：问题的等价重表达，是否也被算法视为等价？

### 阅读与思考

按作者原文区分等价问题、具体求解器与数值条件，再类比教材中的特征尺度变换；类比不是直接定理。

### 适用范围

不能把稀疏恢复中的不变性直接称为神经网络表示不变性或 TD 收敛性；未确认专属作者代码。

- [作者提供的论文](https://drive.google.com/file/d/1pm3VFQu-8afu9yjBGNQ8PZaY1_q_0tWt/view)
- [期刊 DOI](https://doi.org/10.1109/TIT.2017.2686428)
- [IEEE IT Society 正式目录](https://www.itsoc.org/sites/default/files/2021-01/67nits03-SeptWeb.pdf)

---

## Distributed Voting/Ranking with Optimal Number of States per Node

论文 · 2015 期刊；2017 开放预印本 · IEEE Transactions on Signal and Information Processing over Networks；Salehkaleybar、Sharif-Nassab、Golestani

### 研究问题

节点只保留有限局部状态，能否分布式恢复所有选项的全局排序？

### 内容与作用

DMVR 通过相遇节点的集合并与交运算汇聚信息，分析状态数和时间复杂度。排序情形的状态数最优性被证明；投票情形对应最优性在文中是猜想。

### 阅读与思考

先区分投票与完整排序，再读有限状态表示及完全图上的时间分析，避免把不同网络条件合并。

### 适用范围

局部状态复杂度和通信协议不等于学习智能体的最小充分状态；必须保留猜想与定理的区别。未确认专属作者代码。

- [作者开放全文](https://arxiv.org/abs/1703.08838)
- [期刊 DOI](https://doi.org/10.1109/TSIPN.2015.2477777)

---

## Connectivity Analysis of One-Dimensional Ad Hoc Networks with Arbitrary Spatial Distribution for Variable and Fixed Number of Nodes

论文 · 2012 · IEEE Transactions on Mobile Computing；Sharif-Nassab 与 Farid Ashtiani；早期论文入口

### 研究问题

节点数和空间分布改变时，怎样定义并计算网络连通概率？

### 内容与作用

作者早期发表记录中的网络概率分析工作，分别研究节点数可变与固定的设定。阅读价值是学习把随机对象、条件和评价事件明确分开。

### 阅读与思考

从作者提供的全文核对空间分布与独立性假设，再看解析概率和仿真的关系。

### 适用范围

不将连通性分析归为强化学习成果，也不据标题断言它适用于任意相关节点模型；未确认专属作者代码。

- [作者提供的论文](https://drive.google.com/file/d/1ADEHMBvEfcNmnCpVlTHK3OI-D00MRsNx/view)
- [期刊 DOI](https://doi.org/10.1109/TMC.2011.188)

---

## How to Use Real-Valued Sparse Recovery Algorithms for Complex-Valued Sparse Recovery?

论文 · 2012 · EUSIPCO 2012；Sharif-Nassab、Kharratzadeh、Babaie-Zadeh、Jutten

### 研究问题

复数问题转成实数问题后，原来的稀疏结构是否仍被保留？

### 内容与作用

分析用实值求解器处理复值稀疏恢复的转换，并给出保证与修正方式。方程等价不自动代表稀疏性目标和求解保证也完全相同。

### 阅读与思考

先检查实部/虚部展开后的维度，再比较唯一性与具体求解算法的充分条件；留意 LP 和 SOCP 的差别。

### 适用范围

这是表示转换与稀疏求解理论，不是在线 RL 算法；转换后的维数和充分条件有代价。未确认专属作者实验仓库。

- [EURASIP 正式论文](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2012/Conference/papers/1569581759.pdf)

---

## Individualized Challenge Point Practice as a Method to Aid Motor Sequence Learning

论文 · 2018 在线发表 · Journal of Motor Behavior；Wadden、Hodges、De Asis、Neva、Boyd；早期跨领域合著

### 研究问题

运动学习实验如何将练习难度与个体差异联系起来？

### 内容与作用

保留 Kris 机电与运动学习背景中的合著记录。它研究人的运动序列学习，不能因为作者后来研究 RL 就改标为强化学习算法论文。

### 阅读与思考

从研究设计、参与者、难度操控和测量时点读起，再决定与算法任务难度设计的类比是否成立。

### 适用范围

这里只提供原始研究入口，不形成临床或训练建议，也不将人类实验效应外推为机器持续学习结论；未确认公开分析代码。

- [期刊原文](https://www.tandfonline.com/doi/full/10.1080/00222895.2018.1518310)
- [作者发表目录](https://kris.pengy.ca/research)

---

## Predicting Motor Sequence Learning in Individuals with Chronic Stroke

论文 · 2016 在线发表 · Neurorehabilitation and Neural Repair；Wadden、De Asis 等；早期跨领域合著

### 研究问题

对人的运动学习能力作预测时，样本、结局和测量条件应怎样说明？

### 内容与作用

该合著记录来自 Kris 的运动学习与实验系统背景，补足其转向价值预测之前的研究经历；这里不将其技术内容并入 RL 更新机制。

### 阅读与思考

按原文分别查看预测变量、学习指标和适用人群，练习区分预测相关性与干预效果。

### 适用范围

本条不是康复建议，也不主张结论能迁移到不同人群或机器学习系统；未确认公开分析代码。

- [期刊原文](https://journals.sagepub.com/doi/pdf/10.1177/1545968316662526)
- [作者发表目录](https://kris.pengy.ca/research)

---

## John D. Martin：个人发表、履历与教学目录

个人主页与发表目录 · 2017—2026 · 作者维护的主页、精选论文与 2026 CV；2026-10-05 核查

### 研究问题

从处理机器人不确定性，到让经验改变计算结构，研究问题如何演进？

### 内容与作用

早期工作涉及概率模型、海洋机器人与分布式回报，随后转向模型的学习效用、在线结构适应和环境外部记忆。CV 还收录机器人感知论文、2021 NAAMII RL Lecture Series 主讲及历次报告。

### 阅读与思考

按年份并读论文与当时的机构信息；CV 的邀请报告是资源线索，不等于已公开录像。

### 适用范围

主页是精选目录，CV 个别审稿状态滞后；不能把其 Intel、Alberta、DeepMind 等历史工作全部归为 Openmind 成果。

- [主页](https://jdmartin86.github.io/)
- [精选论文](https://jdmartin86.github.io/research/)
- [CV 与报告/教学记录](https://jdmartin86.github.io/assets/cv/2026_martin_cv.pdf)

---

## Reinforcement Learning Algorithms for Representing and Managing Uncertainty in Robotics

论文 · 2021 · 博士学位论文；作者实验室存档

### 研究问题

机器人中的环境随机性与知识不足，为什么需要不同的表示和决策规则？

### 内容与作用

学位论文把回报分布、感知与数据采集、期望回报的认识不确定性放在同一研究路线中，连接分布式 RL、概率推断与导航。

### 阅读与思考

先读章节路线图，再对照期刊/会议论文；特别核对 aleatoric 与 epistemic uncertainty 各进入哪个运算。

### 适用范围

是历史研究整合，不是通用持续学习系统。未核实其提出新的 Thompson sampling 方法，故不作该归属。

- [作者实验室全文](https://robustfieldautonomylab.github.io/John_Martin_PhD_Thesis.pdf)

---

## Extending Model-based Policy Gradients for Robots in Heteroscedastic Environments

论文 · 2017 · CoRL 2017

### 研究问题

状态不同会改变动力学噪声，模型如何避免把噪声一律当成常数？

### 内容与作用

通过相连接的高斯过程描述动力学均值与输入依赖的方差，把异方差不确定性带入基于模型的策略学习。

### 阅读与思考

从噪声模型读到策略评价，检查忽略异方差的基线为何产生不同决策。

### 适用范围

结果来自论文设定的机器人任务；不等于在线结构学习，也不自动保证变化环境中的稳定适应。

- [PMLR 原文入口](https://proceedings.mlr.press/v78/martin17a.html)

---

## Sparse Gaussian Process Temporal Difference Learning for Marine Robot Navigation

论文 · 2018 · CoRL 2018

### 研究问题

价值函数带概率不确定性时，如何控制机器人在线计算的规模？

### 内容与作用

用稀疏高斯过程近似 TD 价值估计，研究用于策略改进的 SPGP-SARSA 与结合模型的 SPGP-TD，并在海洋机器人导航中检验。

### 阅读与思考

跟踪伪输入、后验估计和策略选择之间的关系，再比较模拟与真实导航证据。

### 适用范围

稀疏近似降低计算量，不代表任意长生命期均有固定总资源；回报不确定性也不能直接等同安全保证。

- [原文](https://arxiv.org/abs/1810.01217)
- [PMLR 全文](https://proceedings.mlr.press/v87/martin18a/martin18a.pdf)

---

## Stochastically Dominant Distributional Reinforcement Learning

论文 · 2020 · ICML 2020

### 研究问题

当动作期望回报相近时，怎样比较整个结果分布的风险？

### 内容与作用

学习回报分布并以随机占优关系作决策比较，将分布式 RL 的估计与风险偏好联系起来。

### 阅读与思考

先区分回报随机性与参数后验，再读随机占优条件和分布更新；不要把它当作 Thompson sampling。

### 适用范围

特定分布比较与实验结果不证明所有风险目标等价，也不是持续探索的通用解。

- [PMLR 全文](https://proceedings.mlr.press/v119/martin20a/martin20a.pdf)
- [作者预印本](https://arxiv.org/abs/1905.07318)

---

## Autonomous Exploration Under Uncertainty via Deep Reinforcement Learning on Graphs

论文 · 2020 · IROS 2020；作者预印本

### 研究问题

建图机器人如何同时获取信息并抑制自身定位不确定性？

### 内容与作用

用图表示探索相关的信息，再以 GNN 与深度 RL 选择感知行动，避免每一步都展开昂贵的信念空间规划。

### 阅读与思考

核对图中哪些信息来自估计器、奖励如何衡量信息与定位，以及训练地图与测试地图的区别。

### 适用范围

这是一项有结构的机器人探索任务；图结构是设计的一部分，不能据此推断无先验输入结构下的普适探索能力。

- [原文](https://arxiv.org/abs/2007.12640)

---

## On Catastrophic Interference in Atari 2600 Games

论文 · 2020 · 作者预印本；公开实验代码

### 研究问题

即使不切换任务，学习游戏后段会不会破坏到达它所需的前段能力？

### 内容与作用

通过分段训练和受控干扰实验，将单一环境内的预测冲突与学习平台期联系起来。与跨任务遗忘不同，数据由当前策略产生时，前段退化还能阻止再访后段。

### 阅读与思考

先读如何人为控制干扰，再看性能变化和跨片段预测误差，避免仅凭曲线平台诊断遗忘。

### 适用范围

受控实验支持一种机制，不证明所有深度 RL 的样本低效都由干扰造成。

- [原文](https://arxiv.org/abs/2002.12499)
- [论文链接的作者代码](https://github.com/google-research/google-research/tree/master/memento)

---

## Adapting the Function Approximation Architecture in Online Reinforcement Learning

论文 · 2021 · 作者预印本；frogseye 代码

### 研究问题

不知道传感器空间顺序时，在线预测器能否自己发现有用的非线性连接？

### 内容与作用

Martin 与 Modayil 让函数近似结构随经验适应，在高维、随机且空间关系未直接给定的观测中构造预测特征。它为后来的 Nibbler 提供一条结构适应的前史。

### 阅读与思考

比较固定架构、自适应架构与得到额外结构信息的基线；进入代码检查输入选择与更新时序。

### 适用范围

验证集中于特定空间预测域，并非完整控制系统或任意输入结构的恢复定理。

- [原文](https://arxiv.org/abs/2106.09776)
- [作者代码](https://github.com/jdmartin86/frogseye)

---

## Should Models Be Accurate?

论文 · 2022 · RLDM 2022

### 研究问题

用于更新价值函数的模型，目标应是复现世界还是帮助学习？

### 内容与作用

在 Dyna 式预测中，元学习直接调整模型以提高下游学习效用；简单非平稳环境中，即使准确模型已知，使用它也未必带来最快的学习。

### 阅读与思考

沿“模型产生更新—预测器改变—真实误差改变”读元目标，区分模型误差与学习误差。

### 适用范围

结果不主张模型可以随意错误；它依赖具体学习器、规划预算与外层目标，不是全面否定准确建模。

- [原文](https://arxiv.org/abs/2205.10736)

---

## Meta-Gradient Search Control: A Method for Improving the Efficiency of Dyna-style Planning

论文 · 2024 · 作者预印本；前身为 2022 元学习 workshop 工作

### 研究问题

规划预算有限且模型不完美时，该把模拟更新花在哪些查询上？

### 内容与作用

元梯度调整搜索控制，让模型查询分布服务于真实学习进展；研究模型错误和动态变化下的规划效率。

### 阅读与思考

把查询分布、模型本身与价值学习器分开画出，再检查外层损失使用哪一类真实经验。

### 适用范围

改善查询分配不自动纠正模型，也不保证元学习计算免费；前身与后续论文不能算作两次独立验证。

- [原文](https://arxiv.org/abs/2406.19561)

---

## MOTO: Offline Pre-training to Online Fine-tuning for Model-based Robot Learning

论文 · 2023 · CoRL 2023；arXiv 版本 2024

### 研究问题

离线学到的模型遇到在线分布变化时，如何继续利用旧经验而不被模型误差误导？

### 内容与作用

结合模型价值展开、策略正则化和认识不确定性控制，在图像输入的机器人操纵环境中研究离线预训练到在线微调。

### 阅读与思考

重点看离线/在线阶段数据与目标的变化，再读去掉不确定性控制或正则项的消融。

### 适用范围

是有离线数据与阶段划分的流程，不是从零开始的严格流式持续 RL；项目页的视频也不能替代实验统计。

- [PMLR 原文](https://proceedings.mlr.press/v229/rafailov23a.html)
- [作者项目页](https://sites.google.com/view/mo2o/)

---

## Settling the Reward Hypothesis

论文 · 2023 · ICML 2023；共同第一作者工作

### 研究问题

“所有目标都能写成奖励最大化”在什么数学条件下成立？

### 内容与作用

把偏好与可表示性变成形式化问题，帮助区别奖励假设的口号和对偏好结构的具体要求。它与后来的感知接地问题互补：可表示不等于能够识别奖励所指事件。

### 阅读与思考

先核对偏好对象、假设与表示结论，再构造违反假设的反例。

### 适用范围

数学可表示性不提供自动奖励发现，也不证明感知接地或价值对齐已经解决。

- [PMLR 原文](https://proceedings.mlr.press/v202/bowling23a.html)

---

## On the Interplay Between Sparsity and Training in Deep Reinforcement Learning

论文 · 2025 · 作者预印本

### 研究问题

连接稀疏本身有用，还是训练方式改变了它的作用？

### 内容与作用

比较相近容量下的结构和训练选择，强调固定特征与可训练隐藏权重可能改变架构的相对表现。

### 阅读与思考

把稀疏初始化、稀疏连接和冻结特征分别列为实验变量，再读控制预算后的比较。

### 适用范围

不能把该工作简化为“越稀疏越好”；结构选择、容量与学习规则相互作用。

- [原文](https://arxiv.org/abs/2501.16729)

---

## Time to Take Embodiment Seriously

论文 · 2022 · RLDM agency workshop 立场短文

### 研究问题

把智能体当作抽象程序时，哪些身体与环境提供的计算被忽略了？

### 内容与作用

以具身性重新审视智能体划界和研究方法，是理解 Martin 后来 external artifacts 研究的问题意识的入口。

### 阅读与思考

与 2026 Artifacts 的定义和实验并读，区分哲学/方法论主张与已操作化的假设。

### 适用范围

立场短文不是性能验证；不能把思想上的延续写成其后机制的既有证明。

- [作者全文](https://jdmartin86.github.io/assets/papers/2022_rldm_agency_workshop.pdf)

---

## Artifacts as Memory Beyond the Agent Boundary

论文 · 2026 · 2026-04 预印本；观测型 artifact 的定义、定理与受控实验

### 研究问题

同样的决策能力，需要多少内部容量是否取决于环境能保存什么历史？

### 内容与作用

把当前观测对过去观测的确定性提示定义为 artifact；在规定条件下缩减历史而保留下一观测信息，再以路径、地标和动态痕迹研究参数容量与表现。

### 阅读与思考

先读定义/定理的条件，再检查有无 artifact 时的任务信息和容量计量；与讲座的边界直觉对照。

### 适用范围

不是任意行动条件下的控制充分性定理；参数数目排除了固定回放开销。写读成本、无界环境存储与噪声消融是教材后续问题；未核实作者代码。

- [原文](https://arxiv.org/html/2604.08756v1)

---

## Artifacts as Memory：PRL Symposium 讲座

课程与讲座 · 2026 · 主办方日程与作者 slides；非新增实验论文

### 研究问题

为什么应把智能体的记忆边界与物理边界分开讨论？

### 内容与作用

讲座用环境中的痕迹组织外部记忆的概念，适合在进入预印本形式化定义前建立直觉。

### 阅读与思考

先看例子怎样改变内部需求，再回到原文核查信息关系与容量度量。

### 适用范围

slides 是教学/报告材料；未把其他 CV 所列报告视为已取得完整录像或转录。

- [主办方活动页](https://all.cs.umass.edu/PRL-Symposium/)
- [作者 slides](https://all.cs.umass.edu/PRL-Symposium/2026/slides/martin.pdf)

---

## Joseph Modayil：感知、预测知识与在线架构的研究目录

个人主页与发表目录 · 2008—2026 · 作者主页与按作者检索的原始发表入口；2026-10-05 核查

### 研究问题

从未经解释的传感器流到可行动的知识，哪些接口一直没有消失？

### 内容与作用

早期对象发现、活动识别与传感器几何，连接到 Horde/nexting 的预测知识，再到深度 RL 的干扰/稳定性、自适应结构与真实平台。主页另注明 Openmind、Keen AGI 与此前 DeepMind 经历。

### 阅读与思考

主页保存早期全文；arXiv 作者页补充近年论文，逐条核对作者避免同名误配。

### 适用范围

主页不是穷尽书目，历史工作不能自动追溯归为 Openmind。未核实其独立个人博客或完整公开授课课程。

- [作者主页](https://josephmodayil.com/)
- [作者 arXiv 索引](https://arxiv.org/search/cs?searchtype=author&query=Modayil,+J)

---

## The Initial Development of Object Knowledge by a Learning Robot

论文 · 2008 · Robotics and Autonomous Systems；作者存档

### 研究问题

机器人能否从自己的感知运动经验形成对象，而不是先接受人类对象标签？

### 内容与作用

把时空跟踪、对象属性、类别和可改变属性的动作组合起来，再测试对象识别和需要规划/控制的外部任务。

### 阅读与思考

跟随“经验片段—持续对象—属性—行动”链条，检查哪部分是学习获得、哪部分是设计提供。

### 适用范围

是具体机器人上的结构化构造，不是端到端普适对象学习；概念来自经验也不意味着与人类语义天然一致。

- [作者全文](https://josephmodayil.com/papers/Modayil-ras-08draft.pdf)

---

## Improving the Recognition of Interleaved Activities

论文 · 2008 · UbiComp 2008

### 研究问题

多个活动交错发生时，怎样保存足够历史而不枚举巨大联合状态？

### 内容与作用

Interleaved HMM 同时表达活动内部与活动之间的变化，以近似推断识别 RFID 物体使用序列中的交错活动。

### 阅读与思考

比较只记当前活动与为各活动保留历史的状态设计，核对近似计算和识别误差。

### 适用范围

是活动识别而非 RL 控制；不能把其状态压缩经验直接当作持续控制收益。

- [作者全文](https://josephmodayil.com/papers/Modayil-ubicomp-08.pdf)

---

## Factoring the Mapping Problem: Mobile Robot Map-building in the Hybrid Spatial Semantic Hierarchy

论文 · 2009 · IJRR；作者保存的论文摘要入口

### 研究问题

机器人建图中的几类不确定性是否应该由同一种地图表示承担？

### 内容与作用

Hybrid Spatial Semantic Hierarchy 用不同层次处理局部几何与全局拓扑，研究分解地图构建问题的价值。

### 阅读与思考

先看各层表达什么及层间约束，再与现代端到端状态表示比较。

### 适用范围

分层结构包含设计知识；它是导航/建图历史工作，不能改称无模型持续 RL 算法。

- [作者论文入口](https://josephmodayil.com/papers/Beeson-ijrr-09.html)

---

## Discovering Sensor Space: Constructing Spatial Embeddings That Explain Sensor Correlations

论文 · 2010 · ICDL；作者全文

### 研究问题

没有传感器排列说明书，智能体怎样发现有意义的邻接关系？

### 内容与作用

从相关性推得传感器距离，保留近邻约束，再以快速最大方差展开构造低维嵌入；展示数千传感器和弯曲流形的例子。

### 阅读与思考

把统计相关距离、嵌入距离与真实物理距离分清，再读对传感器朝向和用途的讨论。

### 适用范围

相关几何不必等距于真实位置；Gaussian process 等建模假设与观测分布限制了推断，不能宣称恢复唯一物理结构。

- [作者全文](https://josephmodayil.com/papers/Modayil-icdl-10.pdf)

---

## Horde: A Scalable Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction

论文 · 2011 · AAMAS 2011；作者全文

### 研究问题

一条机器人经验流能否同时回答许多不同策略下的预测问题？

### 内容与作用

多个 demon 共享经验，各自拥有策略、累计信号与终止语义，使用离策略 TD 学习知识；架构把知识规格与学习实现连接起来。

### 阅读与思考

先读每个问题由哪些对象定义，再看共享行为经验如何支持不同目标策略。

### 适用范围

固定问题集合时单步成本不随生命长度增长，但仍随问题数/特征数增长；不自动发现问题，也不等于完整自主控制器。

- [作者全文](https://josephmodayil.com/papers/horde-final.pdf)

---

## Multi-timescale Nexting in a Reinforcement Learning Robot

论文 · 2011—2014 · 2011 预印本；2012 SAB；2014 Adaptive Behavior 期刊版

### 研究问题

机器人能否实时预测大量近未来感觉，而不先建一个完整世界模型？

### 内容与作用

把传感器信号作为累计目标，配以不同折扣，用线性 TD(λ) 学习多时间尺度预测；论文报告约两千个预测在笔记本上超过 10 Hz 更新。

### 阅读与思考

看 0.1—8 秒的尺度如何由采样与折扣决定，再读与离线参照的误差比较。

### 适用范围

展示的是预测可行性；同一行为策略下的许多预测，不等于许多反事实策略模型，也不直接证明控制收益。

- [早期原文](https://arxiv.org/abs/1112.1133)
- [作者期刊全文](https://josephmodayil.com/papers/Modayil-Nexting-AdaptiveBehavior-2014.pdf)

---

## Scaling Life-long Off-policy Learning

论文 · 2012 · 作者预印本

### 研究问题

评估每一个目标策略都要暂停并执行它时，多策略学习怎样扩展？

### 内容与作用

用 GTD(λ) 从共同随机行为中学习目标策略预测，再构造 MSPBE 的在线估计器，降低对穿插 on-policy 测试的依赖；展示真实机器人上的千策略规模。

### 阅读与思考

特别读评估机制如何从“执行目标策略”转为在线目标估计，这与并行多头本身不同。

### 适用范围

MSPBE 是特定近似空间中的学习目标，不等同所有使用者关心的误差；目标策略和特征仍是预先指定的。

- [原文](https://arxiv.org/abs/1206.6262)

---

## Rainbow: Combining Improvements in Deep Reinforcement Learning

论文 · 2017—2018 · 2017 预印本；AAAI 2018

### 研究问题

六种 DQN 改进放在一起，哪些互补、哪些贡献依赖其他部件？

### 内容与作用

把多步回报、分布式价值等 DQN 扩展组合，并用 Atari 消融分析组合表现，为研究整体算法而非孤立技巧提供实例。

### 阅读与思考

先画出组件作用于目标、采样、表示还是探索，再读逐项移除的消融。

### 适用范围

其经验回放和训练协议不同于严格流式学习；Atari 组合结果不能直接移植成真实时间持续架构。

- [原文](https://arxiv.org/abs/1710.02298)

---

## Deep Reinforcement Learning and the Deadly Triad

论文 · 2018 · 作者预印本；稳定性经验研究

### 研究问题

函数近似、自举与离策略相遇时，深度网络中的实际不稳定怎样出现？

### 内容与作用

围绕带经验回放的 DQN 家族，检验不同系统部件与价值估计不稳定、性能之间的关系，把理论风险转成实证诊断。

### 阅读与思考

区分价值爆炸、暂时不稳定和控制回报下降，再核对各消融改变了什么。

### 适用范围

经验诊断不是非线性离策略 TD 的一般收敛证明；某次没有发散也不消除 deadly triad 风险。

- [原文](https://arxiv.org/abs/1812.02648)

---

## Ray Interference: a Source of Plateaus in Deep Reinforcement Learning

论文 · 2019 · RLDM 摘要的完整预印本；机制分析

### 研究问题

为什么多个能力本可同时改善，学习却长时间停在一个个平台？

### 内容与作用

分析函数近似与策略控制数据分布之间的耦合，在受限设定中推导学习动态，并把 ray interference 与鞍点相联系。

### 阅读与思考

先检查简化模型的成立条件，再问真实实验是否同时满足参数共享与数据分布反馈。

### 适用范围

主要贡献是诊断与分析，不是通用新算法；不应把所有平台期都归因于同一机制。

- [原文](https://arxiv.org/abs/1904.11455)

---

## On Inductive Biases in Deep Reinforcement Learning

论文 · 2019 · 作者预印本

### 研究问题

去掉为一个领域精调的偏置，是否只能损失表现？

### 内容与作用

用自适应替代一些目标与环境接口中的领域设定，并在不额外调参的新连续控制任务上比较迁移；结果包含改善也包含退步。

### 阅读与思考

将领域知识、调参预算和自适应模块成本一起核算，再区分本域表现与跨域泛化。

### 适用范围

并非所有偏置都坏或适应总更好；结论受算法家族和任务集合约束。

- [原文](https://arxiv.org/abs/1907.02908)

---

## The Barbados 2018 List of Open Issues in Continual Learning

研究计划 · 2018 · workshop 讨论形成的研究议程，不是完成报告

### 研究问题

通向能在复杂世界自主学习的系统，哪些关键问题仍未解决？

### 内容与作用

汇集持续学习的开放问题和研究方向，适合作为历史议程与当代证据之间的对照。

### 阅读与思考

逐条问问题如今由哪个可复现实验回答；给仍缺证据的条目标注未解决。

### 适用范围

多人讨论稿不代表已实现统一架构；合作者名单也不是任何现有机构的成员名单。

- [原始报告](https://arxiv.org/abs/1811.07004)

---

## Building Machines that Learn and Think for Themselves

论文 · 2017 · Behavioral and Brain Sciences 评论文章

### 研究问题

智能系统应接受人类预制模型，还是自主构造并使用自己的模型？

### 内容与作用

评论在模型推理之外强调自主性，用研究实例讨论减少人类手工知识的重要性。

### 阅读与思考

与后来的传感器结构学习、GVF 问题构造并读，追踪主张如何变成可测机制。

### 适用范围

属于观点/评论，不是新系统性能或自主性的完整定义与证明。

- [作者预印本](https://arxiv.org/abs/1711.08378)

---

## Frost Hollow：预测信号怎样帮助另一个行动者

论文 · 2022 · 同一研究线的短文与长篇实验报告

### 研究问题

当两个行动者感知不同，一个人的时间预测能否成为另一个人的有用信号？

### 内容与作用

在部分可观测、稀疏奖励且存在定时危险的环境中，把预测学习者与控制者耦合，比较时间表示对机器—机器及人—机器协调的影响。

### 阅读与思考

先读短文的 signalling 接口，再读长篇对时间混叠与接收者差异的实验；检查信号是否真的被使用。

### 适用范围

不是完全自发语言涌现；信号通道、任务结构与接收方式已有设计，两篇不能算两次独立验证。

- [Pavlovian Signalling 短文](https://arxiv.org/abs/2201.03709)
- [The Frost Hollow Experiments](https://arxiv.org/abs/2203.09498)

---

## Assessing Human Interaction in VR With Continually Learning Prediction Agents: A Pilot Study

论文 · 2021—2022 · 先导研究；2021 初稿，2022 修订

### 研究问题

预测器还在学习、行为会改变时，人如何调整信任和策略？

### 内容与作用

虚拟现实任务比较两种时间预测架构以及参与者的行为和反馈，把持续学习过程本身纳入人机交互观察。

### 阅读与思考

先读 pilot 的设计和限制，再看早期互动、信任与后续策略之间的线索。

### 适用范围

作者明确不作确定性的人类信任结论；不能把先导样本的趋势写成已确立的普遍规律。

- [原文](https://arxiv.org/abs/2112.07774)

---

## Towards model-free RL algorithms that scale well with unstructured data（Nibbler）

论文 · 2023 · 作者预印本；合成规模实验

### 研究问题

没有预给观测结构时，怎样让辅助预测选择自己的问题和局部输入？

### 内容与作用

Nibbler 选择奖励相关 GVF 及输入子集，把学到的非线性预测特征交给主价值学习器；组合型任务允许有控制地放大观测与状态空间。

### 阅读与思考

先读问题族为何可被有效分解，再核对问题选择、特征使用和尺度曲线。

### 适用范围

实验中的线性样本增长不是所有无结构真实世界问题的复杂度定理；未核实公开作者代码。

- [原文](https://arxiv.org/html/2311.02215v1)

---

## Loss of Plasticity in Continual Deep Reinforcement Learning

论文 · 2023 · 作者预印本；长期非平稳实验

### 研究问题

游戏轮换后越来越难学，是旧知识遗忘，还是更新能力自身退化？

### 内容与作用

长时间 Atari 轮换实验测量权重、梯度与激活，发现激活足迹变稀与梯度衰减相关，并研究 CReLU 缓解。

### 阅读与思考

同时看后续学习速度和机制诊断，不仅看切换瞬间回报；对照干扰与可塑性概念。

### 适用范围

CReLU 的有效性限于所评估设置；长达 50 天/20 亿交互的部分实验不是无限生命期保证。

- [原文](https://arxiv.org/abs/2303.07507)

---

## The Ungrounded Alignment Problem

论文 · 2024—2025 · 2024 预印本；ICDL 2025；作者代码

### 研究问题

若不知道目标概念会以何种传感器编码出现，预先指定的反应如何获得指代？

### 内容与作用

固定字符 bigram 关系先验，以无标签图像序列学习感知编码并检测触发词，隔离“知道关系但不知道感知映射”的接地问题。

### 阅读与思考

读损失中冻结知识与可学习编码器的分工，再检查随机重启、字符识别和词检测的不同指标。

### 适用范围

触发词结果依赖关系先验、64 次重启和频率阈值；不等于一般人类价值对齐，也不是流式 RL 系统。

- [原文](https://arxiv.org/html/2408.04242v1)
- [作者代码](https://github.com/EmergenceAI/babybeaver)
- [ICDL 官方日程](https://icdl2025.fel.cvut.cz/wp-content/uploads/2025/09/conference_booklet.pdf)

---

## Joseph Modayil：Finding the Frame 与 RLC 访谈入口

课程与讲座 · 2024—2026 · 主办方录像目录；RLC 2025 短访谈于 2026 发布

### 研究问题

研究者怎样口头表述自主学习与智能体设计的问题？

### 内容与作用

Finding the Frame 2024 目录列出其 invited talk；TalkRL 的 RLC 2025 采访提供机构与研究观点的简短入口。

### 阅读与思考

报告用于定位问题意识，具体算法和实验结论仍返回对应原文；区分活动年份与发布日期。

### 适用范围

本次核实资源入口，未完成全部录像人工观看或转录，不把未听取的发言写为精确引语。

- [主办方录像目录](https://sites.google.com/view/findingtheframe/past-years/2024/recordings)
- [TalkRL 访谈发布页](https://podcasts.apple.com/tw/podcast/joseph-modayil-of-openmind-research-institute-rlc-2025/id1478198107?i=1000743584018&l=en-GB)

---

## E.-Sorina Lupu：发表、控制教学与 Ad Astra 目录

个人主页与发表目录 · 2013—2026 · 作者维护的主页；页面标注更新至 2026-07

### 研究问题

机器人从实验室可运行走向持续学习，需要哪些原有控制与硬件能力？

### 内容与作用

研究轨迹包括航天器模拟器、对接/组装、分散定位、飞行与双足控制、视觉条件自适应控制，近期连接到 Open Ant 与流式 RL。目录还收录 2015 GBT-FPGA 和 2013 微重力材料实验等早期工程背景。

### 阅读与思考

按发表、教学、教程和博客四种文类阅读；讲述历史时保留当时 Caltech/JPL 等机构背景。

### 适用范围

个人目录可能滞后于新论文；早期工程项目不是 RL 成果，作者页列举的共同作者也不自动属于 Openmind。

- [主页](https://lupusorina.github.io/)
- [发表目录](https://lupusorina.github.io/publications/)
- [Ad Astra 博客目录](https://lupusorina.github.io/blog/)

---

## A Six Degree-of-Freedom Spacecraft Dynamics Simulator for Formation Control Research

论文 · 2018 · AAS/AIAA Astrodynamics Specialist Conference；平台论文

### 研究问题

地面实验怎样暴露航天器导航与控制在物理系统上的问题？

### 内容与作用

M-STAR 将气浮提供的五自由度运动与重力方向的运动机构组合，配以推进器、反作用轮、动力学模型和控制分配。

### 阅读与思考

先区分真实自由度、模拟自由度与地面约束，再看控制器接口和实验可重复性。

### 适用范围

地面航天器模拟器不是在轨验证，更不是自己学习出控制律的持续 RL 智能体。

- [Caltech 原始存档](https://authors.library.caltech.edu/records/r1424-xmc64)

---

## Ultra-Soft Electromagnetic Docking with Applications to In-Orbit Assembly

论文 · 2018 · IAC 2018；机械与控制设计

### 研究问题

如何降低反复对接中的推进剂消耗与对准误差敏感性？

### 内容与作用

设计可容忍小偏差、适应多种相对朝向的电磁对接机构，在仿真与地面六自由度航天器模拟平台上测试。

### 阅读与思考

检查接近、吸附与锁定阶段的假设，分开比较机械容错和控制效果。

### 适用范围

标题中的 in-orbit 是应用目标；本文试验不能被改写为在轨组装或在线 RL 已实现。

- [Caltech 原始存档](https://authors.library.caltech.edu/records/97hth-tde87)

---

## Autonomous In-Orbit Satellite Assembly from a Modular Heterogeneous Swarm

论文 · 2020 · Acta Astronautica；2020 版本包含 Lupu

### 研究问题

异质模块怎样分工并生成避免碰撞的组装轨迹？

### 内容与作用

SOCA 结合分布式拍卖的目标分配与 MPC/序列凸优化的轨迹生成，在六自由度仿真和机器人平台上检验。

### 阅读与思考

将“分配谁去哪里”与“怎么到达”分开，再看对接与避障约束的切换。

### 适用范围

不是通过 RL 发现整个规划系统；2016 同题前身作者不同，不能把它也自动列为 Lupu 的论文。

- [Caltech 原始存档](https://authors.library.caltech.edu/records/xe3ng-16k57)

---

## Adaptive Nonlinear Control of Fixed-Wing VTOL with Airflow Vector Sensing

论文 · 2020 · ICRA 2020；作者预印本

### 研究问题

悬停转入前飞时气动力改变，如何在飞行中修正模型？

### 内容与作用

使用三维气流测量与复合自适应控制，在线更新线性力模型，在定制 VTOL 上比较力预测与速度跟踪。

### 阅读与思考

读模型参数更新使用的预测误差和控制误差，再检查稳定性条件与硬件证据。

### 适用范围

在线适应不等于强化学习；受限参数化控制律的保证不能外推到任意深度 RL 更新。

- [原文](https://arxiv.org/abs/2003.07558)

---

## Decentralized Formation Pose Estimation for Spacecraft Swarms

论文 · 2021 · Advances in Space Research；扩展 2019 workshop 工作

### 研究问题

通信与可见邻居不断变化时，如何保持可扩展的群体位姿估计？

### 内容与作用

DPE 在局部可观测邻域联合估计，SRFE 通过一致性估计公共参考坐标系；数值和机器人模拟器实验检验。

### 阅读与思考

重点核对局部邻域大小、传感/通信图及参考系这三项，而不只看总机器人数量。

### 适用范围

相对于群体规模的局部复杂度说法依赖邻域；这是估计与控制支持模块，不是已实现持续多智能体 RL。

- [Caltech 原始存档](https://authors.library.caltech.edu/records/5zy30-t4193)

---

## A Bipedal Walking Robot That Can Fly, Slackline, and Skateboard（LEONARDO）

论文 · 2021 · Science Robotics；平台与控制论文

### 研究问题

为什么把腿与推进器协同使用能扩大单一身体的行动集合？

### 内容与作用

LEONARDO 协调多关节腿和电推进器，实现步行、飞行及要求精细平衡的运动，展示身体设计与控制相互支撑。

### 阅读与思考

从形态与执行器约束读到控制协调，问能力来自硬件、控制设计还是学习。

### 适用范围

复杂演示不是从经验自主发现技能的证据；不应把该平台的动作直接说成持续 RL 学得。

- [Caltech 论文与补充材料入口](https://authors.library.caltech.edu/records/a3y6x-12v73)

---

## MAGIC-VFM: Meta-learning Adaptation for Ground Interaction Control with Visual Foundation Models

论文 · 2024 · 2024 arXiv v2；作者发表目录列 IEEE Transactions on Robotics

### 研究问题

地形外观能否帮助控制器适应难以精确建模的接触动力学？

### 内容与作用

离线元学习残差/作用矩阵模型，视觉基础模型提供地形特征；在线复合自适应控制更新网络最后一层，处理未被离线训练覆盖的相互作用。

### 阅读与思考

对照没有视觉地形特征的适应控制器，分别读稳定性假设、滑移/坡度/执行器退化实验。

### 适用范围

保证属于规定动力学与更新律；不是整个基础模型在线学习，也不是从零流式 RL。期刊标注来自作者目录，未据此补造 DOI。

- [原文 v2](https://arxiv.org/abs/2407.12304v2)

---

## Vision-Based Detection of Uncooperative Targets and Components on Small Satellites

论文 · 2024—2025 · Small Satellite Conference 2024；作者目录亦列 2025 workshop 展示

### 研究问题

资源受限的航天器如何在远近不同尺度检测未知目标？

### 内容与作用

远距离使用 YOLOv8，近距离把视觉基础模型蒸馏进轻量分割网络，兼顾识别、存储与推理成本，并在空间条件数据上测试。

### 阅读与思考

核查不同距离的任务定义、训练来源和蒸馏成本，再区分可支持 onboard training 与实际执行了在线学习。

### 适用范围

不是 RL 控制实验；同一工作在 workshop 再展示不算独立验证，不能由部署愿景推断在轨持续学习已完成。

- [原文](https://arxiv.org/abs/2408.12084)

---

## Lupu：Caltech 控制课程、项目指导与讲义记录

课程与讲座 · 2021—2025 · 作者教学记录；TA、客座讲授与项目指导角色分开

### 研究问题

控制背景的学生如何走向学习系统的实现与诊断？

### 内容与作用

记录 CDS 110 的作业与自适应控制/RL 客座讲授，CDS 245 数据驱动控制助教、CS 163 项目指导，以及机器人设计项目。

### 阅读与思考

先借助控制任务掌握误差反馈与稳定性，再进入 TD/策略更新；按页面标明的角色理解教学贡献。

### 适用范围

不能把 TA 或两次 guest lecture 写成独立主持整门课；资格考试笔记也不是经过同行评审的定理来源。

- [作者教学目录](https://lupusorina.github.io/teaching/)

---

## Lupu 教程：TD Learning 与 Temporal Abstraction

课程与讲座 · 2025—2026 · 作者教学笔记；页面未逐项给出发布日期

### 研究问题

怎样从控制直觉进入 TD 更新和闭环时间抽象？

### 内容与作用

TD 笔记解释相邻预测差与 TD(λ)；options 笔记用倒立摆的起摆和稳定控制说明策略、启动集合与终止条件。

### 阅读与思考

手算一次预测误差与迹更新，再为起摆/平衡定义 option，最后返回笔记引用的原始论文。

### 适用范围

是入门解读，不是新的收敛定理；不能把控制器分段理解为已经学得 option discovery。

- [TD Learning](https://lupusorina.github.io/tutorials/td_learning/)
- [Temporal Abstraction](https://lupusorina.github.io/tutorials/temporal_abstraction/)

---

## Lupu 教程：Time Discretization in RL

课程与讲座 · 2025—2026 · 作者教学笔记；致谢 Kris De Asis 的反馈

### 研究问题

控制频率改了，保留相同 γ 和逐步奖励还表示同一个任务吗？

### 内容与作用

从连续时间积分出发解释 γ_d=exp(−γ_cΔt) 及奖励与步长的关系，为真实机器人不同频率下的比较提供直觉。

### 阅读与思考

按秒而非按步指定时间尺度；辨认奖励是速率还是已经积累的增量，防止重复乘 Δt。

### 适用范围

一致的时间定义不等于算法在任意离散化下完全不变；遇到正式保证应返回 De Asis 与 Sutton 的原论文。

- [作者笔记](https://lupusorina.github.io/tutorials/time_discretization/)

---

## Lupu 教程与代码：Tile Coding、车辆控制、奖励比较

代码 · 2024—2026 · 教学解释与作业 solutions；非论文基准

### 研究问题

怎样把连续控制问题变成可检查的特征、控制输入与奖励实验？

### 内容与作用

tile coding 笔记解释稀疏二值特征；作者作业仓库覆盖 Ackermann 车辆速度/转向控制和 DDPG/PPO 在不同奖励结构下的比较。

### 阅读与思考

先改一个变量并复现实验曲线，再检查步长、状态尺度与奖励密度是否同时变化。

### 适用范围

教学仓库不是统一预算的算法排名；稀疏特征的低单步成本也不消除高维组合带来的困难。

- [Tile Coding 笔记](https://lupusorina.github.io/tutorials/tile_coding/)
- [车辆速度控制](https://github.com/lupusorina/cds_110_hw5/tree/master)
- [车辆转向控制](https://github.com/lupusorina/CDS_110_HW6)
- [奖励比较代码](https://github.com/lupusorina/reward_analysis_ddpg_vs_ppo)
- [完整教程目录](https://lupusorina.github.io/tutorials/)

---

## Perception-Driven Autonomy and Learning Control for Space Robotics

课程与讲座 · 2025 · 2025-02-28 University of Maryland 主办方报告页

### 研究问题

视觉、学习与控制如何组合以服务空间机器人的自主性？

### 内容与作用

报告摘要连接视觉辅助适应和空间机器人应用，适合与 MAGIC-VFM、航天器感知和平台论文一起阅读。

### 阅读与思考

用报告摘要建立问题地图，再返回各论文核实算法与实验。

### 适用范围

本次核实活动和摘要，不宣称已取得/观看完整录像；邀请报告不是一项独立性能验证。

- [主办方活动页](https://robotics.umd.edu/event/19900/future-leaders-in-robotics-and-ai-seminar-sorina-lupu)

---

## Lessons Learnt from the Spacecraft Simulators

Blog · 2025 · 2025-05-19 作者工程反思

### 研究问题

为什么能演示的机器人仍可能不适合长期学习实验？

### 内容与作用

从航天器模拟器维护经验讨论地面平整度、气路、维修和故障等日常条件，提醒评价系统时记录人工维护与停机成本。

### 阅读与思考

把每条经验转换成平台日志字段或故障注入实验，再与 Open Ant 的维护记录比较。

### 适用范围

经验文章提供工程问题，不是故障率的系统统计，也不证明某种学习算法的性能。

- [作者文章](https://lupusorina.github.io/blog/2025/sc-sim/)

---

## Lupu 硬件笔记：Ankle Torques 与 Connectors

Blog · 2024 · 作者工程博客

### 研究问题

执行器设计和接线可靠性如何改变智能体实际能得到的经验？

### 内容与作用

围绕踝关节力矩与连接器的实践经验，补充算法论文通常略写的身体约束和间歇故障。

### 阅读与思考

阅读时区分机械/电气限制与策略错误，把故障处理和更换部件时间纳入真实实验预算。

### 适用范围

是作者经验而非通用硬件规范；不能据此推断其他平台的寿命或安全认证。

- [Ankle Torques](https://lupusorina.github.io/blog/2024/ankle-torques/)
- [Connectors](https://lupusorina.github.io/blog/2024/connectors/)

---

## Lupu 思考笔记：神经系统可变性与未来预测

Blog · 2025 · 作者读书与研究想法；不是原始神经科学证据

### 研究问题

小型神经回路的多样性、未来预测等想法能启发哪些可检验的学习问题？

### 内容与作用

读书笔记讨论 Eve Marder 相关书籍的启发；Predicting the Future 讨论预测与多行动者未来分支。它们可用于提出假设，不直接构成 RL 结果。

### 阅读与思考

把类比改写成明确机制和对照实验，并对神经科学主张查回原始研究。

### 适用范围

作者读书反思不能替代神经科学论文；未来技术想象也不能作为已实现能力。

- [Lessons from the Lobster 读书笔记](https://lupusorina.github.io/blog/2025/books_lobster/)
- [Predicting the Future](https://lupusorina.github.io/blog/2025/predicting_the_future/)
- [Ideas + Resources are Enough](https://lupusorina.github.io/blog/2025/jpl/)
- [8 Forecasts for the Next 25 Years](https://lupusorina.github.io/blog/2026/forecasts/)

---

## Sorina Lupu：Openmind 研究计划

研究计划 · Openmind 公布版本；页面未标独立日期 · 研究 proposal；不是完成结果清单

### 研究问题

怎样把奖励、时间抽象、模型不确定性和低成本身体纳入同一机器人研究路线？

### 内容与作用

计划涉及奖励设计、机器人中的 STOMP、分布式/概率模型与连续行为，以及可负担双足平台。其价值在于给出待检验方向和与控制经验的连接。

### 阅读与思考

把每一方向改写为假设、实验设置、对照与失败条件，再检索后续论文是否实际验证。

### 适用范围

不能用 proposal 证明相应算法已完成；Open Ant 是后续具体平台，不能视作所有研究计划均已实现。

- [Openmind 原始计划](https://www.openmindresearch.org/s/OpenResearchInstitute_Sorina_Lupu-web.pdf)

---

## The Open Ant: A Robot Platform for Reinforcement Learning Research

论文 · 2026 · RLC / Reinforcement Learning Journal 2026；开源硬件与软件

### 研究问题

怎样让常做模拟实验的研究者也能直接在身体上研究 RL？

### 内容与作用

提供可装配、可维护的四足平台与配套模拟；用来回运动奖励形成非回合任务，展示 SARSA(λ) 与 SAC 从真实经验学习，并比较 sim-to-real 排序。

### 阅读与思考

先核对动作抽象、采样周期、回放和人工干预，再看奖励率与物理运行时间；进入代码检查两种算法协议的差别。

### 适用范围

五次 80 分钟试验仍包括缆线解缠暂停；非回合不等于完全无人干预。主要贡献是平台，不是通用持续学习已经解决。

- [原文](https://arxiv.org/html/2607.18488v1)
- [作者硬件与代码](https://github.com/Openmind-Research-Institute/open-ant)

---

## Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots

论文 · 2026 · RLC 2026；真实平台与开源工程

### 研究问题

持续运行的世界、真实动作时延和身体差异会怎样改变 Atari 学习？

### 内容与作用

Robotroller 操作手柄，摄像头读取画面；reactive 调度先行动再学习，报告平台时延、跨身体迁移及长期运行可靠性。

### 阅读与思考

对照 165 ms 平台时延中是否计入推理，再读经验回放/目标网络与累计运行时长；把物理时间成本纳入实验。

### 适用范围

六个游戏重复试验累计约 145 小时不等于一次 145 小时单生命期；使用回放，不能标为严格 streaming RL。

- [作者项目与原文入口](https://keenagi.com/research/physical-atari/)
- [预印本](https://arxiv.org/abs/2606.19357)
- [作者硬件与代码](https://github.com/Keen-Technologies/physical-atari-rlc)

---

## Streaming Deep Reinforcement Learning Finally Works（2026 v3）

论文 · 2024—2026 · 2024 初稿；2026-09-21 v3 新作者/算法版本；预印本

### 研究问题

不用经验回放和批量更新，深度 RL 能否稳定地逐条经验学习？

### 内容与作用

Stream-X 组合归一化、初始化与更新控制；v3 使用按坐标衰减最大值缩放的 StreamingOptimizer，并扩展 StreamRAC、非平稳 Ant 与资源受限硬件证据。

### 阅读与思考

必须按版本读算法；比较共享组件消融、平台预算和非平稳设置，而非只读标题。

### 适用范围

v1/v2 的 ObGD 不能冒充 v3 当前实现。大量基准仍为平稳任务，有限适应实验不证明所有持续学习障碍已消除。

- [v3 原文](https://arxiv.org/abs/2410.14606v3)
- [作者代码](https://github.com/mohmdelsayed/streaming-drl)

---

## Planetary Exploration 3.0: A Roadmap for Software-Defined, Radically Adaptive Space Systems

研究计划 · 2026 · AIAA ASCEND 2026 / 预印本；跨机构路线图

### 研究问题

航程漫长、通信受限的探测任务，能否在途中重新定义软件能力与科学问题？

### 内容与作用

讨论软件定义任务、在板自主性、可重构系统和逐步形成科学假设等方向，并以远距离探测概念组织技术需求。

### 阅读与思考

从在板学习、验证与确认、硬件寿命和通信限制读出持续智能体的约束，再区分任务愿景与已验证部件。

### 适用范围

不是已执行的深空任务，也不是终身 RL 实验；合作团队中的作者不能自动列为 Openmind 成员。

- [原文](https://arxiv.org/abs/2604.20910)
- [Openmind 研究目录的发表状态](https://www.openmindresearch.org/research)

---

## Michael Bowling：从多智能体学习到目标与评测的资料入口

个人主页与发表目录 · 跨时期 · 作者主页、历史发表表与研究组目录

### 研究问题

对手也在学习时，最优响应、均衡、实际收益和适应能力还是同一个评价量吗？

### 内容与作用

Bowling 的研究从多智能体学习及机器人，延伸到扑克中的遗憾最小化、隐藏信息搜索、ALE 评测，以及奖励表示和持续学习的基础问题。沿这条线阅读，可以看到“学得好”如何随着比较对象与信息权限而改变。

### 阅读与思考

历史发表表适合追踪 WoLF、CFR 与 ALE；扑克组目录补充 DeepStack 和 AIVAT。近期的奖励假设与持续 RL 基础论文应回到各自正式记录，保留合作作者与年代。

### 适用范围

个人发表表目前列至 2015 年，扑克组目录也不是 Bowling 的当前完整个人清单。研究组论文、学生学位论文和合作者工作不能仅凭收录关系归于本人；个人历史不统一归属于今天的机构。

- [作者主页](https://bowlingmh.github.io/)
- [历史发表表](https://bowlingmh.github.io/publications/sort_date.html)
- [扑克研究组发表目录](https://poker.cs.ualberta.ca/publications.html)
- [Amii 作者资料](https://www.amii.ca/people/michael-bowling)

---

## WoLF：当其他学习器也在改变时，更新应当多快？

论文 · 2002 · Artificial Intelligence；Bowling 与 Veloso

### 研究问题

对固定对手学到最优响应，为什么仍不保证两个学习器彼此相遇时稳定？

### 内容与作用

Multiagent Learning Using a Variable Learning Rate 分开提出 rationality 与 convergence：前者针对固定对手，后者关心共同学习的策略动力学。WoLF 采用“占优时慢、落后时快”的更新原则，并将受限矩阵博弈的分析与 WoLF-PHC 的随机博弈实验分开。

### 阅读与思考

先在 matching pennies 画出双策略轨迹，再比较固定步长与可变步长。对照理论 WoLF-IGA 和实际 WoLF-PHC，写清“winning”的参照及可用信息。

### 适用范围

主要收敛分析针对两人两动作重复博弈的理想化梯度动力学，不是任意随机博弈、深度网络或非平稳对手的保证。WoLF 的启发式变步长也不等同于对元目标求导的在线元梯度。

- [作者论文页](https://bowlingmh.github.io/publications/b2hd-02aij-wolf.html)
- [作者提供的全文](https://bowlingmh.github.io/papers/02aij-wolf.pdf)

---

## DeepStack：隐藏信息下的 continual re-solving

论文 · 2017 · Science；Moravčík、Schmid 等与 Bowling 合作

### 研究问题

不能直接看到对手私有信息时，怎样只为当前决策计算，而不预存整个博弈的策略？

### 内容与作用

DeepStack 用分解与不断重求解把计算集中到当前局部博弈，以预先通过自我对弈学到的价值近似支持截断搜索。它将信息集推理、决策时计算和离线获得的知识接在一起；人类比赛还需要专门的方差缩减评估。

### 阅读与思考

分开画出价值网络的训练阶段、一次决策的重求解及比赛评价。检查维护的私有信息分布和对手价值约束为何不能替换为单个完全可观测状态。

### 适用范围

continual re-solving 指重复进行决策计算，不表示价值网络在每手真人对局后持续学习。双人零和扑克、已知规则及搜索计算权限不能直接迁移为未知真实世界中的流式 RL 结果。

- [原始论文 v3](https://arxiv.org/abs/1701.01724v3)
- [研究组论文与评测脉络](https://poker.cs.ualberta.ca/publications.html)

---

## The Untold Story of the Atari Benchmark：评测平台怎样改变研究

课程与讲座 · 2024-01-23 · Amii Approximately Correct 原始访谈；Bowling 为受访者

### 研究问题

一种便于快速实验的环境，如何影响整个领域选择研究问题的方式？

### 内容与作用

访谈回顾将老 Atari 游戏变成通用 RL 实验平台的动机与工程障碍。它把基准从静态排行榜还原为研究工具：共同接口让不同方法可比较，也会让某些容易测量的能力占据中心。

### 阅读与思考

把口述研究史与 ALE 原始论文分开。为自己的持续环境列出它加速检验的机制，以及被重置、加速模拟或统一任务集合遮蔽的能力。

### 适用范围

访谈提供问题选择和平台历史，不替代算法效力证据。Atari 成绩、跨游戏泛化、单生命期适应与无干预长期运行是不同主张。

- [Amii 原始发布与音视频入口](https://www.amii.ca/updates-insights/untold-story-atari-benchmark)

---

## Csaba Szepesvári：算法、理论课程与讲座索引

个人主页与发表目录 · 跨时期 · 作者维护的主题化发表、授课与讲座目录

### 研究问题

一个效率保证究竟允许访问什么数据、使用什么函数类，并与谁比较？

### 内容与作用

Szepesvári 的工作贯穿近似动态规划、随机逼近、bandit、探索和 RL 的统计与计算复杂度。个人目录可按问题筛选，而课程与模型失配讲座帮助理解：表示能力、可识别性、覆盖及规划访问方式怎样决定一个结论的范围。

### 阅读与思考

先区分有生成模型的规划、固定数据的离线学习和真正在线交互；再为每篇定理记录目标、比较器、采样方式、函数类条件和资源量。

### 适用范围

理论研究并非一个现成的终身学习器。固定 MDP 或特定线性条件下的保证，不会仅因算法逐步更新就适用于任意环境漂移。作者为 Csaba，不能与同姓的 David Szepesvari 混淆。

- [作者主页](https://sites.ualberta.ca/~szepesva/)
- [主题化发表目录](https://sites.ualberta.ca/~szepesva/pubs.html)
- [研究生课程](https://sites.ualberta.ca/~szepesva/grad.html)
- [讲座与讲义](https://sites.ualberta.ca/~szepesva/talks.html)

---

## RL Theory：从动态规划算法到学习问题的复杂度

课程与讲座 · 2010 教材；2021–2022 公开课程 · 作者短教材与 Alberta 理论课程讲义、录像

### 研究问题

为什么同一个 Bellman 方程，在不同的数据访问条件下会产生不同的学习难度？

### 内容与作用

Algorithms for Reinforcement Learning 用短篇幅串起价值预测、近似、控制与算法边界。RL Theory 课程进一步从价值迭代、策略迭代和规划复杂度，进入批式与在线 RL；公开页面提供笔记、讲课及讨论录像。

### 阅读与思考

先手推小型 MDP 的 Bellman 收缩，再读近似策略迭代的误差传播。最后比较规划 oracle、批式覆盖和在线探索的权限，不要将三者的样本量直接放在同一轴上。

### 适用范围

2010 短教材不是深度持续 RL 的最新综述；课程分析常以固定环境和明确函数类为基础。理解这些前提，才可能研究环境或表示变化时哪里需要新假设。

- [作者短教材主页](https://sites.ualberta.ca/~szepesva/rlbook.html)
- [RL Theory 课程](https://rltheory.github.io/)
- [讲课与讨论录像目录](https://rltheory.github.io/pages/lectures/)

---

## Bandit Algorithms：把探索收益与遗憾比较器说清楚

课程与讲座 · 2020 · Lattimore 与 Szepesvári 教材；作者免费在线版

### 研究问题

累计遗憾小，究竟是在追踪当前最佳动作，还是接近一个固定比较器？

### 内容与作用

教材从概率与集中不等式建立 bandit 分析，比较随机、对抗、上下文和线性问题，再讨论上界、下界与纯探索。它为持续学习提供一个小而精确的实验场：环境可以变化，但可用反馈和比较器仍必须明确。

### 阅读与思考

先用两臂问题比较 explore-then-commit 与 UCB，再读 Exp3 时重写 regret 的定义。在一次均值切换实验中同时画累计收益、固定臂遗憾和逐时最优差距。

### 适用范围

对抗 bandit 的固定比较器保证不等于无条件追踪任意切换的最优臂；bandit 也没有动作改变未来状态的完整控制难点。配套网站是教材与课程的补充入口，不是独立的 CRL 实验。

- [作者提供的完整教材](https://tor-lattimore.com/downloads/book/book.pdf)
- [Szepesvári 的教材入口](https://sites.ualberta.ca/~szepesva/books.html)

---

## UCT：把 bandit 的探索原则放入 Monte Carlo 规划

论文 · 2006 · ECML；Kocsis 与 Szepesvári

### 研究问题

搜索预算有限时，应继续验证看起来最好的分支，还是探索尚不确定的分支？

### 内容与作用

Bandit based Monte-Carlo Planning 在树中的动作选择上应用置信上界思想，把模拟集中到值得区分的决策。原文研究有限时域或折扣 MDP，在其条件下给出一致性和采样误差分析；这条线把探索与决策时规划联系起来。

### 阅读与思考

在已知小树上分别记录访问次数、均值和探索项。比较均匀模拟与 UCT 的根动作错误率；再固定真实交互预算，单独增加模型调用量。

### 适用范围

这里假设可从生成模型模拟后果。模拟采样不是不可重置的真实经验，逐次扩展的搜索树也不自动满足固定内存。原始收敛分析不能直接覆盖任意神经先验、错误模型或现代 MCTS 变体。

- [作者提供的原始全文](https://sites.ualberta.ca/~szepesva/papers/ecml06.pdf)

---

## Dale Schuurmans：从优化与表示到控制和计算边界

个人主页与发表目录 · 跨时期 · 作者论文与课程目录；Google Research 作者索引

### 研究问题

一种表示便于拟合，是否也便于识别动态、规划和可靠优化？

### 内容与作用

Schuurmans 的工作从统计学习、凸优化和结构化表示，延伸到 RL 的价值—策略联系、策略优化、潜变量表示及语言模型计算。可把这些工作组织成一条方法线：先说明可表达什么，再问是否可学习、能否高效求解。

### 阅读与思考

先从数值优化课复习约束与收敛，再对照 PCL 和策略梯度分析。进入 POMDP 表示论文时，单列可识别性、数据访问和规划条件。

### 适用范围

主题跨度大不意味着这些方法已经集成为持续智能体。作者个人历史、Google 合作与 Alberta 合作应逐篇归属；Google 的作者目录也只是其收录范围内的论文，不是完整个人著作表。

- [作者主页](https://webdocs.cs.ualberta.ca/~dale/)
- [作者论文目录](https://webdocs.cs.ualberta.ca/~dale/papers.html)
- [课程目录](https://webdocs.cs.ualberta.ca/~dale/courses.html)
- [Google Research 作者页](https://research.google/people/daleschuurmans/)

---

## CMPUT 670：Numerical Optimization — Theory and Algorithms

课程与讲座 · 2005 秋季 · Schuurmans 原始课程档案；大纲、作业与项目资料

### 研究问题

说一个 RL 更新“在优化”时，目标、约束、数值精度和收敛条件分别是什么？

### 内容与作用

课程从梯度、收敛及 Newton 方法进入线性、二次、半正定和一般凸规划，并关注效率与精度。它是理解策略优化、对偶方法和元学习更新的数学先修，不是一个 RL 算法合集。

### 阅读与思考

先完成一个带约束二次问题，比较梯度法、投影与二阶方法。再将同样的检查表用于 RL：当前方向是真梯度、半梯度，还是含估计误差的代理更新？

### 适用范围

这是历史公开课程，不代表当前开课安排。静态凸优化的保证不能直接应用到数据分布、bootstrap 目标与表示都随学习变化的深度 RL 系统。

- [课程档案与作业](https://webdocs.cs.ualberta.ca/~dale/cmput670/)
- [作者其他课程](https://webdocs.cs.ualberta.ca/~dale/courses.html)

---

## Path Consistency Learning：价值与策略的共同约束

论文 · 2017 · NeurIPS；Nachum、Norouzi、Xu 与 Schuurmans

### 研究问题

一段由其他策略产生的动作序列，能否同时约束价值函数与目标策略？

### 内容与作用

PCL 从熵正则化最优控制中的 soft consistency 出发，将价值、奖励与动作对数概率连接为多步一致性关系，进而在同策略和离策略轨迹上最小化残差。它提供了理解价值法与策略法联系的另一种方式。

### 阅读与思考

先推导确定性一步关系，再沿路径望远镜求和得到多步形式。随后读附录的随机动态推广，区分条件期望关系与单条带噪轨迹上的残差。

### 适用范围

一致性刻画最优解，不保证任意函数逼近及有限样本优化都能找到它。原文使用轨迹与离策略数据，不能因此称为无样本保存的严格流式算法；熵正则化还改变了目标。

- [原始论文 v3](https://arxiv.org/abs/1702.08892v3)
- [NeurIPS 正式全文](https://papers.nips.cc/paper_files/paper/2017/file/facf9f743b083008a894eee7baa16469-Paper.pdf)

---

## Softmax 策略梯度的全局收敛率：熵究竟改变了什么？

论文 · 2020 · ICML；Mei、Xiao、Szepesvári 与 Schuurmans

### 研究问题

加入熵后优化更快，是同一目标更容易求解，还是目标本身也改变了？

### 内容与作用

论文在表格 softmax 参数化及真实梯度条件下分析策略梯度，给出无正则化时的 O(1/t) 率，以及熵正则化目标下的线性收敛率。关键不是一个通用学习率窍门，而是目标几何、初始动作概率与优化路径的联系。

### 阅读与思考

从两动作 bandit 的 logit 梯度开始，观察低概率动作的更新变慢。分开绘制原奖励目标与熵正则化目标的最优差距，并检查常数对初始化的依赖。

### 适用范围

两种速率针对各自的最优目标，不能将 soft-optimal 直接当作原奖励最优。表格、真实梯度及论文覆盖条件不同于采样 actor–critic，也没有给出任意深网持续控制保证。

- [PMLR 正式论文与附录](https://proceedings.mlr.press/v119/mei20b.html)

---

## Language Models and Computation：表达能力与可学性不能互换

课程与讲座 · RLC 2025；2025-08-26 发布 · Schuurmans 报告；Amii 官方录像与摘要

### 研究问题

一个系统能够表达通用计算，距离在有限经验和算力下学会所需行为，还有多远？

### 内容与作用

报告讨论自回归解码的计算能力，以及后训练在引导模型行为时遇到的计算限制。它与持续 RL 是相邻的架构问题：外部 token 序列、内部状态、程序执行和参数学习承担不同角色，不能只用统一的“推理能力”概括。

### 阅读与思考

分别记录报告中的存在性构造、学习算法和经验结果。把“可表达所需计算”改写成三个追问：谁提供程序、需要多少训练或搜索、哪些状态跨任务保留？

### 适用范围

通用计算的构造不代表任意预训练模型会可靠执行任意程序，也不证明后训练解决了长期在线适应。这里是计算与学习边界的补充读物，不替代教材的持续交互协议。

- [Amii 官方报告](https://www.amii.ca/videos/dale-schuurmans-language-models-computation)

---

## Levi Lelis：搜索、程序化策略与可复用语言的原始资料

个人主页与发表目录 · 跨时期 · 作者主页、精选发表、讲座与 Research Notebook

### 研究问题

知识怎样被组织，才能让未来问题的搜索更容易？

### 内容与作用

Lelis 从组合搜索和启发式算法走向学习引导搜索、程序化策略及语言学习。当前主线不是简单地用符号代替神经网络，而是研究可组合表示及其搜索方法：先付出学习表示的代价，再观察跨问题复用是否值得。

### 阅读与思考

把 Policy-Guided Heuristic Search 的搜索预算，与程序库迁移和神经解释器的表示学习成本分账。结合讲座与研究随笔，追踪“指导搜索”如何发展为“学习搜索空间本身”。

### 适用范围

主页提出的持续、高效、安全学习是研究方向，不能当作已经实现的通用保证。发表页明确是精选列表；程序结构可读也不自动意味着环境行为安全或分布外泛化。

- [作者主页](https://levilelis.github.io/)
- [精选发表目录](https://levilelis.github.io/publications.html)
- [原始讲座目录](https://levilelis.github.io/talks.html)
- [Research Notebook](https://levilelis.github.io/notebook/posts/research/)

---

## Learning at Two Timescales：慢速学习表示，快速寻找程序

Blog · 2026-08-07 · 作者 Research Notebook 研究立场与工作串联

### 研究问题

昂贵的表示学习，要在多少个后续问题上复用，才真正节省资源？

### 内容与作用

随笔将语言看作解空间、将学习看作在语言中寻找程序；慢过程塑造语言，快过程组合或搜索解。它用语义程序库、非确定性程序与神经语言解释器串联具体研究，并明确学习本身可能依然昂贵，收益来自成本摊销。

### 阅读与思考

给一个程序库画出累计成本曲线：构建库的投入、每个新任务的搜索和执行成本都计入。改变任务相似性，寻找相对从头学习的盈亏平衡点。

### 适用范围

两时间尺度是组织研究的观点，不等于已经证明持续交互中的长期净收益；也不同于 actor–critic 的随机逼近步长时间尺度。跨任务复用不自动满足单条流、固定内存或无重置条件。

- [作者原文](https://levilelis.github.io/notebook/posts/research/)

---

## Programmatic Representations for Sequential Decision-Making

课程与讲座 · 2024-11 · IPAM 邀请报告；作者提供原始录像入口

### 研究问题

智能体生成的知识，能否成为下一个问题可组合、可修改的组件？

### 内容与作用

这场报告用程序化表示组织序贯决策研究，适合接在策略、子任务与规划之后阅读。它把注意力从单次搜索找到一个好解，转向表示与搜索如何互相塑造，以及程序组件如何为后续问题提供起点。

### 阅读与思考

把报告中的表示选择、搜索算子和评价任务拆成三列，再对照作者的程序语义空间论文。思考程序组件与 option 的对应：是否都有启动条件、内部行为和终止语义？

### 适用范围

这是研究讲座，不是完整开课记录或独立实证。一个函数库也不能不加定义就称为 options 系统；可组合程序不必具有经学习得到的长期后果模型。

- [作者讲座目录与录像](https://levilelis.github.io/talks.html)
- [作者链接的报告视频](https://www.youtube.com/watch?v=UNpg05yxc3o)

---

## Searching for Programmatic Policies in Semantic Spaces

论文 · 2024 · IJCAI；Moraes 与 Lelis，附作者代码

### 研究问题

修改程序语法却不改变行为，会浪费多少策略搜索预算？

### 内容与作用

论文学习具有不同智能体行为的程序库，再以替换局部程序组件的方式定义搜索邻域，近似按行为组织的语义空间。在 MicroRTS 中，先学得的库被用于后续 MDP 的策略搜索，并与按语法修改的搜索进行比较。

### 阅读与思考

检查程序是否不同与执行行为是否不同这两个统计量。复现实验时分别记录程序评估次数、模拟对局数、建库成本和下游收益，而非仅看最终胜率。

### 适用范围

结果支持给定 DSL、任务与模拟评估协议下的搜索及迁移收益。它不是逐转移深度 RL，也未证明程序库可在固定内存、不断变化的真实世界中无限扩展或始终保持有效。

- [作者提供的正式全文](https://levilelis.github.io/papers/2024/moraesL24.pdf)
- [作者代码](https://github.com/rubensolv/Library-Induced-Semantic-Spaces)

---

## Neural Language Interpreter：让程序结构支持组合与测试时搜索

论文 · 2026 · ICLR；Macfarlane、Bonnet、van Hoof 与 Lelis

### 研究问题

快速适应时，应该修改全部网络权重，还是寻找一个新的可组合程序？

### 内容与作用

Gradient-Based Program Synthesis with Neurally Interpreted Languages 学习离散原语及循环式神经解释器，用 Gumbel-Softmax 松弛训练程序结构，并在测试时通过解释器优化程序猜测。程序序列与共享解释器的分离，使组合和长度泛化成为可单独检验的对象。

### 阅读与思考

分别追踪解释器参数、原语库和当前任务程序的更新时间。先看序列操作与 DeepCoder 的训练／测试划分，再比较初始程序预测与额外测试时搜索的收益及成本。

### 适用范围

主要证据来自程序归纳与组合泛化任务，不是长时间环境交互的 CRL 实验。测试时优化需要任务规格中的输入输出示例；程序更长也意味着更多执行计算，不能把可变长度当作固定资源保证。

- [作者提供的 ICLR 正式全文](https://levilelis.github.io/papers/2026/macfarlaneBHL26.pdf)
- [作者发表记录](https://levilelis.github.io/publications.html)

---

## Martin Müller：搜索、知识、模拟与学习的资料地图

个人主页与发表目录 · 跨时期 · 作者主页及研究组论文、软件、讲座和授课目录

### 研究问题

当前决策中花掉的搜索计算，留下了什么可用于下一次决策的知识？

### 内容与作用

Müller 的脉络从计算机围棋、精确求解和启发式搜索，到 Monte Carlo 搜索、TD-search 及知识与搜索的协作；近期还研究组合博弈算法。它为持续智能体提供相邻但重要的视角：搜索树、值函数、证明与持久经验不是同一种记忆。

### 阅读与思考

先从公开教程区分解出一个游戏和打得强，再读 TD-search 如何把模拟反馈变成可泛化的价值更新。用软件目录找实现，用讲座目录追踪具体评测问题。

### 适用范围

论文和讲座目录属于研究组，其中包含学生论文、外部合作者和由学生主讲的报告，应逐条看署名。完美规则模型中的搜索能力也不能直接等同于未知动力学下的终身学习。

- [作者主页](https://webdocs.cs.ualberta.ca/~mmueller/)
- [研究组发表目录](https://webdocs.cs.ualberta.ca/~mmueller/publications.html)
- [讲座目录（标明主讲人）](https://webdocs.cs.ualberta.ca/~mmueller/talks.html)
- [作者授课目录](https://webdocs.cs.ualberta.ca/~mmueller/courses.html)
- [软件目录](https://webdocs.cs.ualberta.ca/~mmueller/software.html)

---

## Temporal-Difference Search：把模拟中的学习与长期知识分开

论文 · 2012 · Machine Learning；Silver、Sutton 与 Müller

### 研究问题

一次搜索为何一定要保存成显式树，而不能更新一个局部适用的价值函数？

### 内容与作用

TD-search 从当前局面产生模拟经验，用 bootstrap 和函数逼近在相关局面间泛化；Dyna-2 再区分来自真实经验的长期记忆与搜索中的短期记忆。短期参数可以在相邻决策间复用，原文还专门检验清空这种复用的损失。

### 阅读与思考

先标出真实转移、模拟转移和两套参数的更新边界，再读 9×9 Go 的消融。并列检查与简单 alpha-beta 成功结合、与所测 MCTS 组合未成功的结果，不只摘最佳曲线。

### 适用范围

这是已知规则下的模拟搜索研究，使用大量模拟和领域表示；不直接证明未知世界中的样本效率或固定总内存。短期／长期划分是功能和更新来源的区分，不意味着所有短期信息必须每步丢弃。

- [作者提供的正式全文](https://webdocs.cs.ualberta.ca/~mmueller/ps/2012/2012-Silver-Temporal-difference-search.pdf)

---

## From Deep Blue to Monte Carlo：搜索方法的共同结构与差异

课程与讲座 · 2014-07-28 · AAAI 全日教程；Akihiro Kishimoto 与 Martin Müller 联合主讲

### 研究问题

证明获胜、估计胜率和选择一个高收益动作，为何需要不同的搜索停止准则？

### 内容与作用

教程并列介绍游戏求解、alpha-beta、proof-number search 与 MCTS，并提供讲义及分类参考文献。Müller 主讲概述、MCTS 与游戏案例，Kishimoto 主讲求解、alpha-beta 及证明数搜索；适合建立规划方法之间的比较语言。

### 阅读与思考

选同一小型游戏，比较精确求解与有限预算 MCTS。分别报告是否得到证明、动作质量、节点扩展和模拟次数，再问学到的价值或策略应接在哪一个接口。

### 适用范围

这是 2014 年的搜索基础教程，不是当前深度 RL 全景。已知规则、任意模拟和回溯权限必须与真实在线学习分账；求解证书与经验胜率也不能混作一种证据。

- [原始教程与六组讲义](https://webdocs.cs.ualberta.ca/~mmueller/courses/2014-AAAI-games-tutorial/index.html)
- [作者后续课程目录](https://webdocs.cs.ualberta.ca/~mmueller/courses.html)

---

## Fuego：可拆解的 Monte Carlo 搜索软件

代码 · 2008–2018 · Alberta 围棋组开源项目；作者软件目录注明已停止开发

### 研究问题

搜索效果来自树策略、模拟策略、领域知识，还是更多的计算？

### 内容与作用

Fuego 提供围棋与其他棋类开发所需的 C++ 库和 MCTS 引擎，把算法与实际运行接口放在一起。它适合用于阅读传统模拟搜索系统怎样组织规则、采样、备份与评估，而非只看一个 UCB 公式。

### 阅读与思考

从项目文档定位规则和搜索模块，固定局面与随机种子，分别改变模拟预算和启发式设置。用节点数、模型调用数及墙钟时间解释胜率差异。

### 适用范围

这是停止开发的历史代码，需要检查构建环境；本资源不声称已在当前系统复现。Fuego 是搜索系统，不因保留树统计就成为持续参数学习器；多人合作开发也不能称为 Müller 独作。

- [原始项目与文档](https://fuego.sourceforge.net/)
- [作者软件目录与维护状态](https://webdocs.cs.ualberta.ca/~mmueller/software.html)

---

## Unifying Task Specification in Reinforcement Learning

论文 · 2017 · Martha White；ICML 2017，PMLR 70:3742–3750

### 研究问题

环境如何变化，与学习者想预测或优化什么，为什么必须分开？

### 内容与作用

用策略、奖励、转移相关折扣和兴趣函数描述任务，使同一个环境承载多种预测与控制问题。折扣可在特定转移上归零，表达终止；兴趣函数则指定学习目标关注哪些状态。

### 阅读与思考

先在同一条传感器轨迹上定义两个不同 GVF，再读广义 Bellman 算子。逐项区分奖励、延续规则和状态权重，特别检查终止的转移语义。

### 适用范围

兴趣不是额外奖励；转移相关折扣也不会自动解决部分可观测性。逼近误差与算子结论要保留原文条件。此条未确认专属作者代码。

- [ICML 正式记录](https://proceedings.mlr.press/v70/white17a.html)
- [补充材料](https://proceedings.mlr.press/v70/white17a/white17a-supp.pdf)

---

## Off-Policy Actor-Critic with Emphatic Weightings

论文 · 2018 前作；2023 完整期刊论文 · Graves、Imani、Kumaraswamy、Martha White；JMLR 24(146):1–63

### 研究问题

离策略 actor 更新究竟在优化哪个状态加权目标？

### 内容与作用

统一离策略目标并推导包含 emphatic weightings 和兴趣函数的策略梯度，形成 ACE。动作重要性比修正采样动作，强调权重处理状态贡献的传播；二者不能互相替代。

### 阅读与思考

先读目标，再读 OffPAC/DPG 的反例，最后比较强调权重的直接估计和近似。复现仓库的理想与近似 ACE 对照，观察偏差与方差的交换。

### 适用范围

一个反例并非证明半梯度方法在所有任务失败；完整梯度也不保证任意深度网络全局收敛。代码采用预生成经验等实验设置，不能直接标为严格流式协议。

- [JMLR 正式论文](https://www.jmlr.org/papers/v24/21-1350.html)
- [2018 前作](https://arxiv.org/abs/1811.09013)
- [作者实验代码](https://github.com/gravesec/actor-critic-with-emphatic-weightings)

---

## General Value Function Networks

论文 · 2018 预印本；2021 JAIR · Schlegel 等；JAIR 70:497–543

### 研究问题

循环网络的内部状态，能否由一组有明确预测含义的问题来约束？

### 内容与作用

GVFN 将未来信号的预测组织为循环状态，而不把所有隐藏单元都留作无语义的自由表示。预测问题与 TD 学习共同约束记忆，从而研究对反向传播截断长度的依赖。

### 阅读与思考

先列出一个 question network 中每个预测的累积信号、折扣与策略，再比较普通 RNN 与预测约束的训练。把“学到预测”与“该预测对下游任务足够”分开检验。

### 适用范围

减轻截断敏感性不等于恢复任意长历史的精确梯度，也不保证预测集合构成充分状态。此条未确认与最终期刊版本对应的专属实验仓库。

- [JAIR DOI](https://doi.org/10.1613/jair.1.12105)
- [作者全文与版本](https://arxiv.org/abs/1807.06763)

---

## Martha White：Better Actor-Critic Algorithms for RL

课程与讲座 · 公开课程讲座 · Deep RL Course 嘉宾课程；讲稿与视频入口

### 研究问题

actor 与 critic 的耦合更新，怎样从近似策略迭代理解？

### 内容与作用

讲座从 actor-critic 的目标和更新结构讨论算法设计，可与 Greedy Actor-Critic 的研究配套。适合用来追问 critic 的估计误差、actor 的改善目标和采样分布是否一致。

### 阅读与思考

先看策略评价与策略改善两部分，再暂停讲稿，写出各自使用的目标和数据分布；最后回到论文核实具体算法条件。

### 适用范围

这是 actor-critic 课程，不是可塑性专项报告；讲座中的直觉不能替代论文证明，也不能据视频标题推断全部算法都适合无 replay 更新。

- [课程原始入口](https://deeprlcourse.github.io/guests/martha_white/)
- [讲稿](https://deeprlcourse.github.io/assets/guests/martha_white.pdf)
- [视频](https://www.youtube.com/watch?v=bmq9jmhWkq0)
- [相关研究](https://arxiv.org/abs/1810.09103)

---

## Empirical Design in Reinforcement Learning

论文 · 2024 · Patterson、Neumann、Martha White、Adam White；JMLR 25(318):1–63

### 研究问题

一条更高的学习曲线，到底支持哪个科学结论？

### 内容与作用

把研究问题、性能度量、随机性、超参数选择和统计比较放进同一实验设计。核心不是套一个显著性检验，而是让观测与问题匹配，避免只展示经过筛选的最好配置。

### 阅读与思考

先写自己要估计的是在线累计收益、最终性能还是稳定性，再选择实验单位、种子与环境变体。特别读调参与比较相互影响的例子，并为失败运行预先定好处理规则。

### 适用范围

方法论指南不是新的基准分数，也不提供对所有 RL 实验通用的单一统计方案。有限任务上的区间不直接描述开放世界的泛化不确定性。

- [JMLR 正式记录](https://www.jmlr.org/papers/v25/23-0183.html)
- [全文](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf)

---

## Adam White：Reinforcement Learning Experiments that Matter!

课程与讲座 · 公开课程讲座 · Deep RL Course 嘉宾课程；作者讲稿与视频

### 研究问题

怎样从“跑通算法”走到“实验真正回答问题”？

### 内容与作用

以 RL 经验研究的设计为入口，连接比较对象、性能指标、参数选择和可信结论。适合作为实验章节的先导课，再用正式方法论论文补充统计与报告细节。

### 阅读与思考

把自己的实验重写成一个可被反驳的问题，列出支持与不支持该判断的观测；再检查所选基线和计算预算是否公平。

### 适用范围

演讲是教学材料，不是已复现的新实验；观看后仍需为具体任务设计独立运行、调参预算与置信区间。

- [课程入口](https://deeprlcourse.github.io/guests/adam_white/)
- [讲稿](https://deeprlcourse.github.io/assets/guests/adam_white.pdf)
- [视频](https://www.youtube.com/watch?v=TjQSaOqj5R8)

---

## University of Alberta：Reinforcement Learning Specialization

课程与讲座 · 公开在线课程系列 · Coursera 官方课程；Martha White 与 Adam White 任教

### 研究问题

进入持续 RL 之前，哪些预测、控制与实验基础必须亲手做过？

### 内容与作用

四门课从 RL 基础、采样式学习和函数逼近走到完整学习系统项目，适合建立 TD、控制、策略梯度与可复现实验的共同语言。

### 阅读与思考

按基础 → 样本更新 → 函数逼近 → 综合项目学习。每做完一项作业，再问原设置是否允许保存数据、重置环境或停止学习，形成向持续 RL 的迁移。

### 适用范围

入门课程的标准任务并不自动满足严格流式或永续学习约束；平台访问条件可能变化，此处不承诺免费证书或持续开放的作业权限。

- [官方四课系列](https://www.coursera.org/specializations/reinforcement-learning)
- [基础课程](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning)
- [函数逼近课程](https://www.coursera.org/learn/prediction-control-function-approximation)

---

## GVFs in the Real World: Making Predictions Online for Water Treatment

论文 · 2023 在线发表；2024 卷期 · Janjua 等；Machine Learning 113:5151–5181

### 研究问题

真实传感器关系会变化时，离线训练好的预测器为何还需要部署中更新？

### 内容与作用

在饮用水处理数据上研究 GVF 的折扣累计预测，结合记忆迹缓解部分可观测性，并用离线数据预训练与选择在线适应参数。比较冻结预测器和部署中继续更新，以及 GVF 与固定步距预测。

### 阅读与思考

先看季节变化、运行模式和传感器变化，再看如何划分预训练与在线评估。注意累计目标与 n 步单点目标不是同一随机变量，因此要理解归一化误差的比较口径。

### 适用范围

这是实际系统数据上的预测研究，不能写成已证明自动控制收益或水处理安全性的论文。离线准备与在线更新并存，不等于从零开始的严格单流协议。

- [期刊正式记录与全文](https://link.springer.com/article/10.1007/s10994-023-06413-x)
- [作者预印本](https://arxiv.org/abs/2301.11476)

---

## Tuning-free Step-size Adaptation（AutoStep）

论文 · 2012 · Mahmood、Sutton、Degris、Pilarski；ICASSP 2012

### 研究问题

每个特征需要不同步长时，如何不把困难全部转移给元步长？

### 内容与作用

AutoStep 在逐特征步长适应中加入归一化与有效更新尺度控制，研究输入尺度、噪声和非平稳目标变化下的跟踪。它是理解后来的 TD 步长适应与 SwiftTD 的重要前史。

### 阅读与思考

先按线性监督学习设定读元更新与归一化，再比较 IDBD、LMS 和 RLS；检查原文哪些稳定机制后来可以迁移到 TD，哪些还需要重新推导。

### 适用范围

标题中的 tuning-free 不能解释为所有深度 RL 都没有超参数；本文的线性学习实验和假设不等同于带 bootstrap 的控制问题。

- [作者论文](https://armahmood.github.io/files/MSDP-Autostep-ICASSP-2012.pdf)
- [作者非平稳实验代码](https://github.com/armahmood/nonstationary-experiments)

---

## Setting up a Reinforcement Learning Task with a Real-World Robot

论文 · 2018 · Mahmood、Korenkevych、Komer、Bergstra；IROS 2018

### 研究问题

把仿真算法接到机器人，为什么任务接口本身就是实验的一部分？

### 内容与作用

研究真实机器人 RL 中动作、观测、奖励与交互周期的设置。SenseAct 提供与机器人硬件交互的基础设施，使时间、系统进程和学习循环的关系变得可检查。

### 阅读与思考

先画传感、行动与更新的时间线，再读环境设置；记录动作周期、重置和任务边界。比较算法前，先确认这些接口条件没有同时发生变化。

### 适用范围

软件接口和有限机器人实验不构成通用安全认证；真实硬件复现需要独立风险评估。时间预算也不能只按环境步数比较。

- [作者论文](https://arxiv.org/abs/1803.07067)
- [SenseAct 原始代码](https://github.com/kindredresearch/SenseAct)

---

## Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers

论文 · 2024 会议；2025 修订预印本 · Vasan 等；NeurIPS 2024；Action Value Gradient（AVG）

### 研究问题

不保存经验、不等一批数据，深度策略梯度还能有效学习吗？

### 内容与作用

提出增量式 AVG 并组合归一化与缩放机制，在每次更新只使用最新样本的条件下研究稳定性；包含机器人仿真、机械臂和移动机器人实验。

### 阅读与思考

先核对 batch=1、无 replay、无 target network 的设置，再逐项读稳定化机制和消融。把样本效率、墙钟时间、内存以及机器人交互时长分开报告。

### 适用范围

有限实验支持增量学习的可行性，不证明无限期稳定或任意任务上的超参数通用性；也不能把全部性能变化归因于单一归一化组件。

- [作者论文与会议声明](https://arxiv.org/abs/2411.15370)
- [作者代码](https://github.com/gauthamvasan/avg)

---

## Streaming Deep Reinforcement Learning Finally Works

论文 · 2024 首稿；2026 v3 · Elsayed、Lupu、Vasan、Mahmood；按 arXiv v3 作者与内容收录；预印本

### 研究问题

深度学习器失去 replay 和批量平均后，如何控制单样本更新的破坏？

### 内容与作用

围绕流式学习的数值稳定与更新尺度组合算法组件，并在离散、连续控制及机器人设置中检验。与 AVG 配套阅读，可以看到“仅删去 replay”与重新设计增量学习系统的差别。

### 阅读与思考

固定阅读版本，先审计经验存储、更新频率和配置共享范围，再读归一化、激活与步长控制的消融。区分数据流协议成立与回报改善两个判断。

### 适用范围

此条按 2026 v3 收录，不能把新增作者与实验反写成 2024 首稿内容；不因预印本标题而宣称所有深度 RL 的流式障碍已经解决。

- [作者版本记录](https://arxiv.org/abs/2410.14606)
- [v3 全文](https://arxiv.org/html/2410.14606v3)
- [作者代码](https://github.com/mohmdelsayed/streaming-drl)

---

## Rupam Mahmood：Reinforcement Learning for Robots

课程与讲座 · 2019 秋季课程及公开讲义 · University of Alberta CMPUT 652 历史课程；非当前招生公告

### 研究问题

从 TD 与策略梯度到真实机器人闭环，还缺哪些工程和学习概念？

### 内容与作用

课程将 RL 基础与机器人控制、实时系统、仿真到现实及实际实验连接。其价值在于把交互周期和系统约束带回算法讨论，而非仅提供一套机器人演示。

### 阅读与思考

先补学习规则，再沿机器人系统与实验设置模块阅读。为自己的装置列出计算时限、观测延迟、允许重置条件和人工干预日志。

### 适用范围

历史课程中的硬件与依赖可能过时；材料不是现成安全部署指南，也不保证所有链接中的作业环境仍可运行。

- [机器人课程主页与日程](https://armahmood.github.io/rl-robots-course/)
- [基础 RL 课程](https://armahmood.github.io/rlcourse/)

---

## Rupam Mahmood：Single-life 与 Streaming RL 讲稿

课程与讲座 · 作者公开讲稿与讲座 · 作者主页提供的原始讲稿；研究观点与实验入口

### 研究问题

若只有一次持续运行的生命，调参、重置和学习成本应怎样计入？

### 内容与作用

将实时交互与单次生命约束作为学习系统设计的出发点。适合对照流式算法论文，区分“学习不存数据”与“实验没有反复重来”的不同要求。

### 阅读与思考

先列出通常被隐去的预训练、调参、重置和失败恢复成本，再看讲稿如何重构问题。随后回到对应论文确认哪些观点已有实验支持。

### 适用范围

single-life 是问题设定和研究方向，不代表所有展示实验都没有离线准备；讲稿不应与已经证明的无限期适应保证混同。

- [Single-life RL 讲稿](https://docs.google.com/presentation/d/16Pw_uT9kq28KeBlnd8zehaYDr4ds6sRy/edit)
- [Streaming RL 讲稿](https://docs.google.com/presentation/d/1LsZnut5V5sCuO0BgjAB7chspzerj_mPUqKU07shqTe0/edit)
- [Streaming RL 视频](https://www.youtube.com/watch?v=QOfkOl9QrZY)
- [作者讲座目录](https://armahmood.github.io/)

---

## Real-Time Recurrent Learning Using Trace Units in Reinforcement Learning

论文 · 2024 · Elelimy 等；NeurIPS 2024；RTU

### 研究问题

保存长期记忆，是否必须保存反向传播所需的历史窗口？

### 内容与作用

用结构化递推设计可高效维护的在线敏感度迹，把循环状态构建与前向梯度递推结合。关键在于针对结构约束的精确递推，而非给任意稠密 RNN 免费提供低成本 RTRL。

### 阅读与思考

先手推一个递归单元的状态和参数敏感度，再比较 RTU 的结构与成本；最后分别读增量预测和控制实验的训练协议。

### 适用范围

表示能力与计算成本存在结构性取舍；多层传播须核对梯度连接和近似。采用 RTU 的 PPO 实验仍可使用 rollout 与批量更新，不能统称严格流式控制。

- [NeurIPS 正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)
- [作者版本](https://arxiv.org/abs/2409.01449)
- [作者代码](https://github.com/esraaelelimy/rtus)

---

## Deep Reinforcement Learning with Gradient Eligibility Traces

论文 · 2025 · Elelimy、Daley、Patterson、Machado、Adam White、Martha White；RLC 2025 / RLJ

### 研究问题

把多步信用加入稳定的梯度 TD，究竟应对哪个带 λ 的目标求导？

### 内容与作用

从带资格迹的广义投影 Bellman 误差出发构造梯度 TD 方法，并区分适合经验回放的前向形式与适合增量学习的后向形式。资格迹既改变目标，也影响可实现的计算。

### 阅读与思考

从目标与投影定义开始，再把前向回报、后向迹和辅助估计逐一对应；最后分开阅读 PPO/MuJoCo 与 Stream Q/MinAtar 的证据。

### 适用范围

不同实验使用不同训练协议；一种后向递推可在线实现，不表示论文所有实验无 replay。理论条件与深度控制经验结果需分别说明。

- [RLJ 正式论文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)
- [作者全文](https://arxiv.org/html/2507.09087v1)
- [作者代码](https://github.com/esraaelelimy/gtd_algos)
- [作者链接的讲解视频](https://youtu.be/fHSk-XPBNKE)

---

## Rethinking the Foundations for Continual Reinforcement Learning

论文 · 2025 · Elelimy、David Szepesvári、Martha White、Michael Bowling；RLC 2025 / RLJ 6:2174–2194

### 研究问题

持续学习的好坏，为什么不能只与一个最终静态策略相比？

### 内容与作用

重新讨论过程、学习目标和评估参照，使持续学习的行为变化进入问题定义。用这一视角审视静态最优策略、预先封闭的任务和阶段结束后的测试分数。

### 阅读与思考

先写一个继续适应优于永久冻结的例子，再读论文对过程与评价的定义；检查所比较的 agent 类、信息和计算条件。

### 适用范围

定义与评价框架不等于一种已解决持续学习的算法。本文使用四作者 RLJ 正式版本；两作者 RLDM 摘要与其作者、篇幅和证据范围不能混用。

- [RLJ 正式记录](https://rlj.cs.umass.edu/2025/papers/Paper243.html)
- [四作者正式全文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)

---

## AGaLiTe: Approximate Gated Linear Transformers for Online Reinforcement Learning

论文 · 2023 预印本；2024 TMLR · Pramanik、Elelimy、Machado、Adam White；TMLR 2024

### 研究问题

注意力记忆能否在历史越来越长时仍保持有限推理成本？

### 内容与作用

把注意力改写为带门控的线性递推，用固定大小的状态汇总历史。它关注长交互中的状态与计算预算，可与 RTU 的在线梯度设计作互补比较。

### 阅读与思考

先区分推理时的 recurrent state 与训练时需要保存的计算图，再读门控和近似注意力。使用作者 PPO 配置时明确 rollout、优化轮数和序列长度。

### 适用范围

固定大小的推理记忆不等于整个训练过程不保存样本或无截断反传；压缩历史也有表达损失，不能当作完整注意力的恒等替换。

- [作者论文](https://arxiv.org/abs/2310.15719)
- [TMLR 记录](https://openreview.net/forum?id=lh6vOAHuvo)
- [作者代码](https://github.com/subho406/agalite)

---

## Forager: A Lightweight Testbed for Continual Learning with Partial Observability in RL

Benchmark · 2026 · Tang 等；作者目录列 RLC 2026；此条链接公开论文 v1

### 研究问题

维护可塑性的方法，遇到记忆不足导致的失败还会有效吗？

### 内容与作用

用轻量、固定环境内存的觅食任务，分别调节视野和变化机制，研究状态构建与持续适应的相互作用。例子说明仅维持可训练性，不一定补足部分观测中丢失的信息。

### 阅读与思考

先比较相同任务的视野，再比较 memory traces、循环结构和可塑性干预。最后读不断产生新学习需求的变体，并核对环境内存与 agent replay 内存的区别。

### 适用范围

固定环境内存不代表 agent 无经验存储；论文也包含 DQN 和 PPO 协议。基准不具备所有开放世界属性，某些静态变体仍允许收敛。未在此条确认独立作者代码入口。

- [论文版本记录](https://arxiv.org/abs/2605.01131)
- [完整实验与协议](https://arxiv.org/html/2605.01131v1)
- [作者发表目录](https://esraaelelimy.github.io/)

---

## Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks

论文 · 2023 · Javed、Haseeb Shah、Sutton、Martha White；JMLR 24(256):1–34

### 研究问题

在线梯度太贵时，可以改变网络本身，而不只近似梯度吗？

### 内容与作用

通过独立模块与分阶段构造的约束，使 RTRL 的成本随参数数目线性增长。主要交换是网络的函数表达能力与高效精确学习，而非给梯度加入新的随机噪声或截断偏差。

### 阅读与思考

先画列之间的依赖，再列出每个参数必须维护哪些敏感度。结合动物学习预测与预训练 Atari 策略评价实验，比较同计算预算下的网络大小与效果。

### 适用范围

精确性针对被约束的架构和阶段，不是任意稠密循环网络。Atari 策略评价不是端到端学习一个持续改善的控制器。

- [JMLR 正式记录](https://www.jmlr.org/papers/v24/23-0367.html)
- [作者代码](https://github.com/kjaved0/columnar-constructive-networks)

---

## Meta-learning Representations for Continual Learning

论文 · 2019 · Javed 与 Martha White；NeurIPS 2019

### 研究问题

一个表示若容易在线更新，是否比只在当前任务准确更有价值？

### 内容与作用

元学习目标关注未来序列中的在线学习表现，学习有利于减少更新干扰的表示。它把表示质量从静态可分性扩展到“后续更新会怎样影响其他知识”。

### 阅读与思考

区分外层元训练和内层在线适应，再观察稀疏表示、干扰和保留表现。画出外层梯度穿过哪些更新，计算它需要保存多少历史。

### 适用范围

外层使用昂贵的跨更新反向传播，不是部署时完全在线的元学习算法；训练任务分布与后续迁移范围仍有限制。

- [NeurIPS 正式论文](https://papers.nips.cc/paper_files/paper/2019/file/f4dd765c12f2ef67f98f3558c282a9cd-Paper.pdf)
- [作者代码](https://github.com/kjaved0/mrcl)

---

## The Big World Hypothesis and Its Ramifications for AI

Blog · 2024 · Javed 与 Sutton；RLC Finding the Frame 立场文章与演讲

### 研究问题

世界始终大于 agent 的计算和表示能力，会怎样改变学习研究的目标？

### 内容与作用

把有限 agent 与复杂世界之间的容量差作为出发点，强调持续跟踪、状态构建与计算效率。由此可以追问：增加离线训练是否真的消除了部署中学习的需求？

### 阅读与思考

先把主张改写为可检验的问题：在固定内存、每步算力和生命长度下，何时继续学习优于冻结？再用 SwiftTD、CCN 等机制论文寻找具体实现。

### 适用范围

这是研究立场，不是证明所有世界必然非平稳或所有离线方法无效的定理；后续创业评论也属于观点，不替代实证比较。

- [作者原文](https://khurramjaved.com/the_big_world_hypothesis.html)
- [作者链接的演讲](https://www.youtube.com/watch?v=Fwkcc9tupCI)
- [后续作者评论](https://khurramjaved.com/not_ready_for_big_worlds.html)

---

## Loss of Plasticity in Deep Continual Learning

论文 · 2024 · Dohare 等；Nature 632:768–774

### 研究问题

为什么一个持续做梯度下降的网络，会逐渐更难学会新东西？

### 内容与作用

用长期持续学习实验区分旧知识遗忘与新知识学习能力下降，并提出 continual backpropagation：伴随常规学习，少量替换成熟且低效用的单元，为学习持续提供新特征。

### 阅读与思考

先看如何保持任务难度并测量后续学习，再读效用、成熟期和替换率。检查与随机替换等对照，避免把所有衰退都直接归因于某一个单元统计量。

### 适用范围

在所测长序列中保持可塑性，不等于任意非平稳过程下无限期学习的数学保证；替换也可能损伤有用知识，须同时测即时损失与后续恢复。

- [Nature 正式论文](https://www.nature.com/articles/s41586-024-07711-7)
- [作者代码](https://github.com/shibhansh/loss-of-plasticity)
- [Nature 作者访谈](https://www.nature.com/articles/d41586-024-02756-0)

---

## Reinitializing Weights vs Units for Maintaining Plasticity in Neural Networks

论文 · 2025 · Hernandez-Garcia、Dohare、Jun Luo、Sutton；CoLLAs 2025

### 研究问题

重新初始化应作用于整个单元，还是更细粒度的权重？

### 内容与作用

Selective Weight Reinitialization 将低效用选择下沉到权重粒度，并与单元替换比较。含 layer normalization 或 attention 的网络使单元作用更耦合，替换粒度因此成为独立设计问题。

### 阅读与思考

先比较两种替换操作会破坏哪些计算，再按网络大小、归一化与注意力结构分层读结果；同时记录替换率与恢复速度。

### 适用范围

优势依赖结构：不能把某些小网络、归一化或注意力设置的收益写成所有架构上都胜过 CBP。作者也报告部分大网络中两类方法接近。

- [CoLLAs 正式条目](https://lifelong-ml.cc/Conferences/2025/acceptedpapersandvideos/conf-2025-17)
- [作者论文](https://arxiv.org/abs/2508.00212)
- [作者代码](https://github.com/JFernando4/collas_2025_swr_paper)

---

## Learning Forever Using Artificial Neural Networks

论文 · 2025–2026 · Alberta 博士论文；作者主页列 2025，公开 PDF 题页为 2026

### 研究问题

怎样把可塑性衰退、持续反向传播与策略坍塌放进统一研究脉络？

### 内容与作用

学位论文汇总并扩展 Dohare 关于可塑性损失、CBP 和 policy collapse 的研究，提供比单篇论文更完整的定义、实验与相关工作讨论。

### 阅读与思考

先用 Nature 论文建立现象，再读论文中的形式化与扩展实验，最后比较预测学习失败和策略坍塌。将重叠的会议/期刊结果与学位论文新增讨论分开记笔记。

### 适用范围

标题中的 forever 是研究目标，不应解读为已证明无限时间内持续学习。年份存在作者目录与公开 PDF 题页差异，因此保留二者而不强行统一。

- [作者公开学位论文](https://shibhansh.github.io/data/Dohare_Shibhansh_202601_PhD.pdf)
- [作者主页与论文说明](https://shibhansh.github.io/)

---

## Marlos Machado：从基础 RL 到 Deep RL 的公开课程

课程与讲座 · 2023—2026 · 作者公开课程与讲义目录

### 研究问题

怎样区分算法的 Bellman 目标、神经网络实现和经验采样协议？

### 内容与作用

CMPUT 365 提供价值、TD、函数逼近和规划的基础线；CMPUT 628 将 DQN、多步、分布价值、辅助任务、架构、经验回放和策略梯度拆开讲授。它适合补足持续 RL 读者的共同语言，而不是把一种深度 RL 配方当作完整智能体。

### 阅读与思考

先完成基础 TD 与函数逼近，再读 W26 深度课程的辅助任务和回放两讲，画出哪些量每步更新、哪些经验会重复使用。

### 适用范围

课程目录不同学期开放程度不同；正在进行的 F26 不能视为全部讲义已发布。课程本身不证明这些组件满足严格流式约束。

- [作者课程页](https://mcmachado.github.io/teaching.html)

---

## Representation-driven Option Discovery in RL：作者客座讲义

课程与讲座 · 2025 · 公开客座讲义

### 研究问题

状态表示为什么可能同时决定技能的方向和未来能够收集的经验？

### 内容与作用

讲义把谱表示、eigenoptions 与表示驱动的技能发现连成一个反馈过程：表示用于定义行为，行为改变访问分布，新经验再影响表示。其教学价值在于解释技能不是预先给定的任务标签。

### 阅读与思考

沿着表示、内在目标、终止与行为采样四个接口阅读；再与价值感知 eigenoptions 的负面在线结果对读，检查正反馈需要什么条件。

### 适用范围

这是研究脉络与方法讲解，不是任意环境中技能发现必然改善探索的定理；离线发现与在线联合学习应分别评估。

- [作者客座讲义 PDF](https://deeprlcourse.github.io/assets/guests/marlos_machado.pdf)
- [课程原始入口](https://deeprlcourse.github.io/)

---

## A Study of Value-Aware Eigenoptions

论文 · 2025 · RLC Inductive Biases in RL Workshop；预印本

### 研究问题

给 eigenoptions 加上价值信息，会改善信用分配，还是让经验分布过早偏向已知区域？

### 内容与作用

研究区分预先指定技能与在线发现技能。前者可以帮助探索和信用分配，但在线生成时技能也可能过强地塑造访问分布，反而妨碍学习。它把表示—行为循环中的失败模式变成可以观察的实验问题。

### 阅读与思考

先比较技能是否固定，再看 option 价值和终止的作用；复现时同时记录外部回报、覆盖范围和 option 使用频率。

### 适用范围

工作坊结果针对所测方法与环境，不能推出所有 eigenoptions 无效；也不能只摘取固定技能的收益而忽略在线联合学习的限制。

- [预印本](https://arxiv.org/abs/2507.09127)
- [工作坊论文 PDF](https://openreview.net/pdf?id=ZHlMV2B49W)

---

## The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis

论文 · 2025 · NeurIPS 2025；作者公开预印本

### 研究问题

当智能体只是世界计算的一小部分，固定的有限任务模型遗漏了什么？

### 内容与作用

论文把有限智能体嵌入更大的计算过程，以此构造持续预测与行为问题；其形式化联系到可数无限 POMDP，而不仅是把有限 MDP 的状态数放大。合成实验用于检验模型与学习方法在这一设定下的表现。

### 阅读与思考

先辨认智能体计算边界和世界生成过程，再比较预测目标、行为目标与实验规模。把它与教材资源账本对读，问环境复杂度增长时哪些状态摘要仍可维护。

### 适用范围

计算嵌入是明确的理论建模选择，不是所有现实环境的经验定律。合成基准与所测试架构的结果不等于一般持续智能已经实现。

- [预印本](https://arxiv.org/abs/2512.23419)
- [NeurIPS 正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/28f6ba36ea9895f41f9cd8a193d72ee5-Abstract-Conference.html)
- [作者研究与发表脉络](https://mcmachado.github.io/research.html)

---

## CMPUT 607：Applied Reinforcement Learning

课程与讲座 · 2018 · 作者公开课程大纲

### 研究问题

怎样从第一周的真实执行器实验开始学习预测与控制，而不是最后才把算法搬到机器人？

### 内容与作用

课程让学生围绕简单机器人学习 GVF、策略梯度和 actor–critic，并接触预测状态、离策略函数逼近与人类奖励。实验记录、同伴批评和自主项目使算法更新与物理接口共同成为学习对象。

### 阅读与思考

把大纲改写为自己的实验协议：先定义时间戳、观测、动作、奖励及安全停止，再开展单一预测问题和简单控制；保留失败记录与设备干预。

### 适用范围

公开页是历史课程大纲，不是可直接运行的完整课程包；部分材料通过课程共享目录分发。机器人课程也不自动满足无限期无重置的持续学习条件。

- [大学托管的课程大纲](https://sites.ualberta.ca/~pilarski/teaching/CMPUT607-W18/index.html)

---

## Creating an Exocerebellum

课程与讲座 · 2024 · CAPNet / CSBBCS 主题讲座与作者幻灯片

### 研究问题

学会大量预测之后，谁选择预测问题，又是谁把预测转化为对人的帮助？

### 内容与作用

讲座将 GVF 在线预测置于人机紧耦合的控制与反馈通路中，讨论假肢和可穿戴原型。它明确区分能够实时学习和使用预测，与尚需人为选择预测单元、设计下游通信的现状。

### 阅读与思考

按 PDF 页码读第 28–29 页的两条通路、第 48 页的振动反馈、第 54–56 页的 Wrystlebot，最后读第 60 页的能力与缺口；为每个预测注明接收者及行为后果。

### 适用范围

exocerebellum 与 digital Purkinje cell 是组织研究的类比，不是神经机制等价证明。讲座不能代替所引临床论文，也未证明预测问题和通信接口已经自主学得。

- [作者讲座入口](https://pilarski.github.io/talk/creating-an-exocerebellum/)
- [作者幻灯片 PDF](https://pilarski.github.io/talk/creating-an-exocerebellum/pilarski-2024-06-25-creating-an-exocerebellum.pdf)
- [作者原型代码](https://github.com/pilarski/Wrystlebot)

---

## Communicative capital: A key resource for human-machine shared agency and collaborative capacity

论文 · 2023 · Neural Computing and Applications；观点与文献综合

### 研究问题

一个人与设备长期磨合后形成的协作能力，应该存放在哪个分析单位中？

### 内容与作用

论文将持续交互中形成的沟通资源称为 communicative capital：价值不只在人的技巧或机器参数中，也在双方逐渐可理解、可协调的关系中。这为人机共同适应提供了超出单个设备准确率的分析视角。

### 阅读与思考

选择一个反馈或动作接口，写出双方分别学到了什么；再比较更换设备、重置机器参数或更换用户后哪些协作能力消失。

### 适用范围

这是概念框架及文献综合，不是一个统一的资本估计器，也未给出跨系统通用的收益保证；不能把增加机器自主性直接等同于改善用户体验。

- [作者论文页](https://pilarski.github.io/publication/mathewson-2023-communicative/)

---

## Joint Action is a Framework for Understanding Partnerships Between Humans and Upper Limb Prostheses

论文 · 2024 · IEEE BioRob 2024；概念分析

### 研究问题

当假肢也会预测和适应，把它看作被动工具还足够吗？

### 内容与作用

作者以 joint action 分析三种肌电假肢控制方式，围绕目标、相互监测、预测、通信和适应比较人机关系。贡献是把控制器性能问题扩展为伙伴如何形成协调的问题，而非简单主张机器应拥有更多控制权。

### 阅读与思考

对照比例肌电加顺序切换、模式识别和自适应切换，找出每种方式中人必须承担的判断与切换负担；将协调平滑和预测使用拆成可测指标。

### 适用范围

论文是对控制方式的概念比较，不是新临床试验。joint action 的解释力不能替代个体化安全、使用负担和长期功能效果的测量。

- [作者论文页](https://pilarski.github.io/publication/dawson-2024-joint/)

---

## FrostHollowVR：人机决策协作的可运行环境

代码 · 2022 研究脉络；开放代码 · 作者开放环境与论文配套资源

### 研究问题

怎样在不直接操作临床设备的情况下，研究机器预测是否帮助人的决策？

### 内容与作用

Frost Hollow 把风险时序预测与行动时机选择分离；作者提供 VR 环境，可用于观察人如何利用或忽略机器信息。它为预测误差到协作结果之间的接口提供了实验载体。

### 阅读与思考

先读论文的情境与提示语义，再检查代码中事件时序、可见信息和交互规则。比较无提示、可靠提示和延迟提示时的行为，而不只比较预测损失。

### 适用范围

代码资源不是独立有效性证据。VR 任务与真实假肢在风险、身体耦合和用户人群上有重要差异，不能直接外推临床收益。

- [作者 VR 代码](https://github.com/pilarski/FrostHollowVR)
- [作者论文页](https://pilarski.github.io/publication/pilarski-2022-frost/)

---

## CMPUT 656：Interactive Machine Learning

课程与讲座 · 2020 · 作者公开课程介绍

### 研究问题

有人参与学习过程时，实验的基本单位还是单个算法吗？

### 内容与作用

课程覆盖从人工标注、人类在环到人机组队的交互式学习，并以设计先导研究为目标。其价值在于把人类研究方法与算法实验放在同一训练路径上。

### 阅读与思考

用课程提出的问题重写一个奖励实验：人能看到什么、何时反馈、反馈成本是多少、机器怎样反过来影响人的行为？先设计可解释的小规模先导研究。

### 适用范围

课程介绍提到录制安排，但不等于当前所有录像均可公开访问；这里提供的是原始教学入口。课程不是单一交互式 RL 算法的性能证据。

- [作者课程入口](https://drmatttaylor.net/teaching/cmput656-f20/)
- [课程网站](https://sites.google.com/ualberta.ca/cmput656/home)

---

## Improving Reinforcement Learning with Human Input

论文 · 2018 · IJCAI Early Career 论文与演讲主题

### 研究问题

演示、评价反馈和课程安排，究竟是在改变学习系统的哪一个接口？

### 内容与作用

Taylor 将非技术参与者可提供的信息区分为行为示范、在线评价和任务课程。这三种帮助分别影响行为数据、更新信号和经验顺序，不能全部压成一个外部奖励通道。

### 阅读与思考

为每种人类输入画出来源、时间延迟和算法消费位置；比较人类投入相同预算时，哪种帮助真正缩短学习过程。

### 适用范围

这是研究脉络概述与开放问题整理，不是所有输入方式的统一比较试验；非专家输入也可能有噪声、偏差和不一致。

- [IJCAI 原始论文入口](https://www.ijcai.org/Proceedings/2018/817)

---

## Human-Interactive Robot Learning: Definition, Challenges, and Recommendations

论文 · 2026 · ACM Transactions on Human-Robot Interaction；定义、挑战与建议

### 研究问题

从人类采集过数据，与人和机器人在学习期间相互影响，是同一种设置吗？

### 内容与作用

文章将 Human-Interactive Robot Learning 的核心放在双向交互：机器人从交互中学习，人的行为影响机器人，机器人也影响人的后续输入。这一界定要求同时研究两方的学习与适应，而不是把人仅视为固定数据源。

### 阅读与思考

从定义检查一个系统是否真的存在在线双向影响，再阅读研究建议。把人类负担、可理解性和交互变化加入算法回报之外的评价。

### 适用范围

这是共同定义与研究议程，不是新的通用机器人学习算法或临床试验。预先收集的人类数据本身不足以满足这一交互定义。

- [共同作者机构原始入口](https://ai.sony/publications/human-interactive-robot-learning-definition-challenges-and-recommendations?hsLang=en)
- [出版 DOI](https://doi.org/10.1145/3779297)
- [共同作者托管的正式论文 PDF](https://people.cs.gmu.edu/~xiao/papers/hirl_vision_paper.pdf)

---

## Transfer Learning for Reinforcement Learning Domains: A Survey

论文 · 2009 · JMLR 10；与 Peter Stone 合作的历史综述

### 研究问题

迁移让新任务更快，是否也让整个学习历程更便宜？

### 内容与作用

综述区分源任务选择、任务映射、被迁移知识及评估目标，并特别区分只计算目标任务成本和计算整个训练历程。它为持续学习中常见的隐藏预训练成本和负迁移问题提供了早期清晰框架。

### 阅读与思考

先读评估部分，分别报告初始性能、累计回报、达到阈值时间以及源任务成本；再判断任务映射是人给定还是系统学得。

### 适用范围

文献覆盖截止于当时，不能当作深度 RL 现状综述。其指标与问题分类可复用，但具体方法结果属于前深度 RL 时期，也不应归为后来的 Alberta 任职成果。

- [JMLR 原文 PDF](https://jmlr.org/papers/volume10/taylor09a/taylor09a.pdf)

---

## Learning to Play Table Tennis From Scratch using Muscular Robots

论文 · 2020—2022 · IEEE Transactions on Robotics 2022；2020 预印本

### 研究问题

高速运动任务中的探索，为什么不能只靠算法中的动作噪声解决？

### 内容与作用

研究把气动人工肌肉驱动的机器人与强化学习结合，利用柔顺、可回驱的身体特性支持高速真实交互中的探索。关键是硬件、状态估计和学习控制共同决定可承受的试错范围。

### 阅读与思考

先读执行器与实验设置，再看学习协议；分别记录击球效果、失误恢复、硬件约束及外部干预，避免只把结果归因于 RL 更新式。

### 适用范围

身体柔顺不等于无条件安全；实验结果限定于具体装置和任务。真实训练也不自动意味着无回放、无重置或可无限期运行。

- [预印本](https://arxiv.org/abs/2006.05935)
- [作者主页与发表记录](https://embodied.ml/)

---

## DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems

论文 · 2022—2023 · ICLR 2023；作者预印本与代码

### 研究问题

在许多执行器共同产生同一运动的身体上，独立动作噪声为什么可能覆盖不了有用状态？

### 内容与作用

DEP-RL 将 differential extrinsic plasticity 的自组织探索与 RL 结合。探索利用传感—动作变化关系生成协调运动，而非仅逐通道注入无结构噪声；研究在过驱动和肌骨系统的到达、运动任务中比较学习效率。

### 阅读与思考

先辨认动作维数、身体约束与实际到达状态的差别，再读探索控制和任务策略的结合方式。比较状态覆盖而不仅是动作方差。

### 适用范围

所报告收益依赖身体与算法设置；它不证明自组织探索总比噪声好，也不是持续分布漂移下的无限期适应保证。

- [预印本与版本记录](https://arxiv.org/abs/2206.00484)
- [作者代码](https://github.com/martius-lab/depRL)
- [ICLR 原始论文入口](https://openreview.net/forum?id=C-xa_D3oTj6)

---

## Hierarchical Reinforcement Learning with Timed Subgoals

论文 · 2021 · NeurIPS 2021；HiTS

### 研究问题

动态世界里的子目标，只规定到哪里而不规定何时到达，是否足够？

### 内容与作用

HiTS 让高层指定子目标状态及到达时间。这个额外接口把动作执行时机纳入层级控制，服务于外部物体持续运动、低层策略又在学习的情境。

### 阅读与思考

比较状态子目标和带时间子目标的信息差异；用移动球的例子推演：到达同一位置但早到或晚到，为什么可能对应完全不同的结果？

### 适用范围

定时子目标解决的是特定的层级接口与动态任务困难，不等于系统已经学得任意抽象技能，也不提供真实设备全生命周期安全保证。

- [NeurIPS 原文 PDF](https://proceedings.neurips.cc/paper/2021/file/b59c21a078fde074a6750e91ed19fb21-Paper.pdf)
- [作者代码](https://github.com/martius-lab/hits)
- [预印本](https://arxiv.org/abs/2112.03100)

---

## Dynamic Real-world Reinforcement Learning through Muscular Robots

课程与讲座 · 2023 · University of Leeds 主办讲座

### 研究问题

能够长期承受试错的身体设计，怎样改变可研究的学习问题？

### 内容与作用

讲座以肌肉驱动机器人解释动态真实 RL 的工程前提：快速运动、状态不确定性和训练耐受性必须一起考虑。它适合与表格式探索讨论对照，理解机器人不是可随意重置的抽象环境。

### 阅读与思考

先读主办方摘要，再观看作者讲解时分别记下学习算法、执行器特性和实验支持系统；不要把机械顺应性带来的收益写成纯算法收益。

### 适用范围

这是 2023 年讲座，其页面中的当时任职信息属于历史。演示与演讲论点须回到原论文实验核对，不表示每种装置都可安全长期训练。

- [主办方讲座入口](https://robotics.leeds.ac.uk/events/seminar-dynamic-real-world-reinforcement-learning-through-muscular-robots-dr-dieter-buchler-from-max-planck-institute-for-intelligent-systems/)
- [主办方链接的视频](https://youtu.be/GQtpSMEpn5A)

---

## General Value Function Networks

论文 · 2018—2021 · JAIR 70；论文与作者实现

### 研究问题

循环状态是否只能通过最终任务损失来间接获得含义？

### 内容与作用

GVFN 用对未来信号的 GVF 预测约束循环网络内部状态，使状态分量具有可检验的预测语义。论文推导学习目标和算法，并研究截断时间反向传播长度变化时的表现。

### 阅读与思考

先写出每个预测问题的 cumulant、终止和策略，再看它如何约束内部状态；对比只有外部任务损失的 RNN 与带预测约束的网络。

### 适用范围

指定预测问题是归纳偏置，不保证得到所有任务充分的状态，也未解决问题集合的自主发现。更短反向传播下的实验鲁棒性不是无条件稳定性定理。

- [JAIR 论文版本](https://arxiv.org/abs/1807.06763)
- [作者代码](https://github.com/mkschleg/GVFN)

---

## Laplacian Representations for Decision-Time Planning

论文 · 2026 · ICML 2026；ALPS，预印本 v2

### 研究问题

长时距规划需要更精细地滚动原始动力学，还是更合适的多尺度距离与子目标？

### 内容与作用

ALPS 利用 Laplacian 表示组织多时间尺度的规划，以局部代价与长期子目标支持决策时搜索，减轻长链预测误差的问题。研究在离线目标条件设置中评估，而不是边部署边持续收集数据的完整智能体。

### 阅读与思考

对照表示学习、局部规划和高层子目标三个模块，检查各自使用的数据与预算；问更好的潜在距离在哪些情形能替代更深的原始状态 rollout。

### 适用范围

OGBench 等离线评估不能直接证明在线持续适应、数据自生成或严格流式性质。应区分离线数据可达性与真实探索能够获得同类覆盖。

- [论文与版本记录](https://arxiv.org/abs/2602.05031)

---

## Operator Learning for Power Systems Simulation

论文 · 2025 · 预印本；NeurIPS Climate Change AI 工作坊报告

### 研究问题

从粗时间步训练的动力学近似，能否用于更细分辨率的物理模拟？

### 内容与作用

Schlegel、Taylor 与 Farrokhabadi 将函数到函数的算子学习用作电力系统时域仿真的代理模型，比较三类方法的零样本超分辨率及稳定、失稳动力学之间的泛化。问题核心是时间步变化下模型语义是否保持。

### 阅读与思考

区分时间步不变性、预测误差与闭环控制性能；先阅读简化测试系统，再看跨时间分辨率和跨动力学状态的两种测试。

### 适用范围

这是简单系统上的概念验证，原文明确尚未纳入高比例可再生能源电网的实际复杂性。它是动力学建模研究，不是持续 RL 控制或电网部署成功的证据。

- [预印本](https://arxiv.org/abs/2510.09704)
- [作者工作坊幻灯片](https://ccai-papers.s3.us-east-1.amazonaws.com/neurips2025/55/slides.pdf)

---

## Matthew Schlegel：预测研究博客与可复现实验工具

Blog · 2018—2022 及后续目录 · 作者博客、教学与代码入口

### 研究问题

预测知识的主张如何落到可读、可追溯的实验实现？

### 内容与作用

早期博客解释作者从物理训练转向预测学习的兴趣；代码目录提供 GVFN、组合 GVF 与实验管理工具。两者合读，可以区分研究动机、技术论文和支撑复现的软件，而不把个人随笔当作实验结果。

### 阅读与思考

先读博客的研究问题，再到代码目录选择对应论文实现；检查配置、随机种子、日志与参数扫描如何被记录。CV 还提供 AI4Good 教学和阅读组经历的历史线索。

### 适用范围

博客是个人研究动机，工具目录不代表每个旧依赖当前都可运行。历史 CV 的任职与研究兴趣不能替代当前机构页面。

- [作者博客](https://mkschleg.github.io/post/2018-11-03-blog-state-of-purpose/)
- [作者代码与教程目录](https://mkschleg.github.io/code/)
- [作者 CV](https://mkschleg.github.io/CV.pdf)