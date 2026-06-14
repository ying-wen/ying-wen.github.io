# 当智能体开始从世界中学习，而不是人类

Author: Ying Wen
Language: zh
Published: 2026-03-15
Canonical URL: https://yingwen.io/zh/blog/when-agents-learn-from-world/
Tags: agentic-rl, environment-scaling, multi-agent, continual-learning
Description: 预训练范式规模化数据，智能体范式规模化环境。关于环境规模化、持续进化与多智能体协作的三个观察。

*温颖 · Claw · 2026 年 3 月*

> 预训练范式的核心洞察是：数据的规模化能涌现出智能。但当我们从"模型"走向"智能体"，一个对偶的洞察正在浮现——**环境的规模化同样能涌现出智能，甚至是更通用的智能**。本文梳理我在这个方向上的三个观察：环境如何规模化，智能体如何持续进化，以及多个智能体如何通过协作突破单体上限。


## 一、环境规模化：智能体时代的苦涩教训

Rich Sutton 在 2019 年写下《苦涩的教训》（The Bitter Lesson）[25](#ref-25)：利用计算的通用方法最终会胜过依赖人类知识的方法。预训练时代的实践完美验证了这一点——堆数据加堆算力，涌现出了我们没有显式编程的能力（Kaplan et al., 2020）。

现在，同样的逻辑正在大模型智能体强化学习中重演，只是规模化的对象从**数据**变成了**环境**。

2025 年，Karpathy 在年度总结中将"基于可验证奖励的强化学习"（RLVR）称为大模型训练流水线的"新主要阶段"[1](#ref-1)。DeepSeek-R1 用纯强化学习激励出推理能力，不依赖人类标注，已在 Nature 上得到系统验证[2](#ref-2)。OpenAI 的 o1/o3 系列同样依赖可验证奖励在大量环境中训练[3](#ref-3)。

但 RLVR 目前的成功集中在**封闭域**——数学证明、代码竞赛、形式化推理——这些领域的奖励信号天然可验证。当大模型智能体强化学习扩展到开放域（网页操作、API 调用、多步决策），**环境的构建本身就成了瓶颈**。

2025 年底的一篇综述系统梳理了这个问题，明确指出"环境和任务的多样性规模化是防止智能体过拟合到特定行动模式的关键"[4](#ref-4)。2026 年 2 月的两篇工作——Agent World Model[5](#ref-5) 和 ScaleEnv[6](#ref-6)——分别探索了用代码自动合成训练环境的路径。AGENTRL[7](#ref-7) 则试图建立大模型智能体强化学习的统一规模化框架。Wing Venture Capital 的分析指出，Anthropic 在强化学习环境上的投入预计 2026 年增长 3-5 倍[8](#ref-8)。

### 环境规模化为什么比数据规模化更难

数据规模化的核心操作是"收集加清洗"。环境规模化则涉及三个更复杂的维度：

**多样性与真实性的张力。** 合成环境容易做到数量大，但难以保证与真实用户需求的分布一致。企业级长流程任务（几十步到上百步）几乎没有好的评测基准，更别说训练环境。我们在 ZSC-Eval[9](#ref-9) 中为多智能体零样本协作建立了系统化的评测框架，但即便在这个相对受限的领域，覆盖真实协作场景的多样性仍然极具挑战。

**环境需要结构。** 简单地堆砌随机环境不够——智能体需要在**有结构的**环境中学习推理。我们在近期的工作中直接研究了这个问题：通过强化学习让模型在结构化的上下文环境中学习推理（Yu, Zhao, Zhang et al., ICLR 2026）[26](#ref-26)。实验表明，当环境具有可被利用的内在结构时，强化学习的训练效率和泛化性都显著优于在非结构化环境中的训练。这提示我们：**环境规模化不只是"量"的问题，更是"结构"的问题。**

**"强化学习友好型"基座模型。** 环境再好，如果基座模型缺乏基本的世界模型能力和自我验证机制，强化学习后训练的天花板也很低。我们在 TS-LLM[10](#ref-10) 中的实验表明，将 AlphaZero 式的树搜索引入大模型的解码和训练过程，可以系统性地提升推理质量——但前提是基座模型本身具备足够的"搜索友好性"。什么样的基座模型最适合在丰富环境中通过强化学习提升？这个问题目前几乎没有系统研究。


## 二、持续进化：当智能体的环境不再外在于自身

去年下半年，大模型能力到达一个拐点——从"玩具"变成了"生产力"。但部署到垂直领域后，"差那么一点"的问题普遍存在。这催生了一个明确需求：**让已部署的智能体在服务过程中持续提升——不是重新训练，而是在线进化。**

但持续进化不只是算法问题。**智能体能否在真实环境中长期稳定运行**，本身就是一个巨大的工程挑战。

### 运行框架：持续进化的基础设施

Anthropic 在 2025 年末发表了一篇重要的工程实践文章《Effective Harnesses for Long-Running Agents》[27](#ref-27)，系统描述了让智能体跨多个上下文窗口持续工作的方案：一个初始化智能体负责搭建环境，一个工作智能体在每个会话中做增量推进，同时留下清晰的中间产物供下一个会话使用。

这篇文章揭示了一个被低估的事实：**当前智能体系统的持续性瓶颈往往不是模型能力，而是运行框架（harness）的鲁棒性。** 网络中断、API 报错、上下文溢出——任何一个都可能让长时间运行的智能体功亏一篑。

这个问题在多智能体场景中被进一步放大。如果单个智能体的长程稳定性都难以保证，多个智能体的协调运行就更加脆弱——一个智能体的崩溃可能导致整个协作链条断裂。**运行框架不仅是单智能体的问题，更是多智能体系统的基础设施问题。**

我们在 DPT（Dual Process Theory）框架[28](#ref-28)中从认知科学的角度切入了这个问题：借鉴人类"快思考"（System 1）和"慢思考"（System 2）的双过程理论，设计了一个能在实时人机协作场景中持续稳定运行的语言智能体框架。快过程负责实时响应和即时决策，慢过程负责深度推理和策略调整。这种架构天然具备更好的鲁棒性——快过程保证了系统不会因为慢过程的延迟而停顿，慢过程保证了系统不会因为快过程的粗糙而退化。

### 记忆到技能的压缩：一条务实的进化路径

在运行框架保证稳定性的基础上，持续进化的核心机制是什么？

我们近期的工作 MemRL[14](#ref-14) 从一个具体角度切入：将智能体的**稳定认知推理**与**动态情景记忆**显式解耦——模型参数不变，通过强化学习优化情景记忆的检索和利用来实现自进化。这回避了直接修改参数带来的灾难性遗忘风险，同时保留了持续学习的能力。

但记忆只是中间态。智能体运行时积累的记忆数据冗杂、昂贵、有噪声。更完整的路径是：

```
记忆（原始数据）→ 原子技能（提炼）→ 组合技能（组合）→ 参数内化（蒸馏）
```

**正向技能蒸馏**：从运行轨迹中提取原子技能，发现频繁共现模式，组合成更高层技能。**反向技能蒸馏**：当技能库膨胀后，把已验证的技能蒸馏回模型参数，释放上下文空间。

这形成了一个闭环：**运行时积累记忆 → 压缩为技能 → 内化到参数 → 释放上下文 → 继续积累新记忆**。

### 长程轨迹中的信用分配

持续进化的另一个关键障碍是：在长程智能体轨迹中如何正确归因？

我们在 POAD[15](#ref-15) 中提出了一种行动分解方法：将智能体的每次行动分解为行动内和行动间两个层次，分别做贝尔曼回溯。实验表明，这种更细粒度的信用分配能显著提升智能体在多步任务中的学习效率。

这和当前 RLVR 的实践相互印证。DeepSeek-R1 和 OpenAI o1 的成功都依赖结果级奖励而非过程级奖励——粗粒度的最终奖励加上模型自身的信用分配能力，比人为设计中间奖励更具可扩展性。POAD 的贡献在于：**在不引入人为中间奖励的前提下，通过行动结构的分解来提升强化学习自身的归因精度。**


## 三、多智能体协作：从分工到探索多样性

当单个模型通过自我博弈或自我改进进化到能力上限后，一个自然的问题是：**引入多个异构智能体能否突破这个上限？**

我的观察是：多智能体协作在大模型场景中的核心价值，**不在于分工，而在于探索多样性**。

每个智能体有自己的模型、上下文和环境配置，能力边界各不相同。一个智能体探索不出来的方向，另一个可能因为不同的模型基座、不同的上下文积累、或不同的环境交互历史而天然覆盖。这和我们在多智能体强化学习中长期积累的洞察一致。

### 从通信协议到推理协作：一条演进线

早在 2017 年，我们提出 BiCNet[16](#ref-16)，首次将双向协调网络引入多智能体系统，证明了智能体间的实时通信可以涌现出人类水平的协调策略——无需人类示范。Foerster et al. (2016)[21](#ref-21) 和 Sukhbaatar et al. (2016)[22](#ref-22) 也分别从可微通信和反向传播通信的角度验证了这一点。

此后，我们沿两条主线持续推进：

**协作的理论保证。** HATRPO/HAPPO[17](#ref-17) 证明了异构智能体可以在保证单调改进的前提下进行去中心化训练。MAT[18](#ref-18) 进一步将多智能体协作重新理解为序列建模问题，用 Transformer 的注意力机制替代了显式的通信协议设计。

**协作的规模化。** MALib[19](#ref-19) 是我们为大规模种群进化训练构建的基础设施，支持多种群并行进化。COLE[11](#ref-11) 在此基础上引入开放式学习，让合作策略在开放环境中自动演化。

**从博弈到推理的跨越。** ReMA[20](#ref-20) 标志着这条线从传统多智能体博弈扩展到了大模型推理领域——多个智能体分别负责推理、反思、验证，通过多智能体强化学习训练实现"元思考"（meta-thinking）。这证明了多智能体协作不仅能提升行动层面的能力，也能提升推理层面的能力。

而 DPT[28](#ref-28) 则从人机协作的角度补充了这条线——当协作的对象从"智能体与智能体"扩展到"智能体与人类"时，实时性和鲁棒性变得至关重要。双过程架构让智能体能在人类的实时交互中保持稳定的协作能力。

### 通信带宽：被低估的瓶颈

但我认为当前多智能体大模型系统有一个被低估的瓶颈：**智能体间通信带宽极低。**

目前主流的多智能体协作模式——MapReduce 式并行、共享上下文、串行流水线——本质上都是**异步的、粗粒度的**。智能体 A 完成任务后把结果传给智能体 B，中间没有实时交互。这和人类团队协作的模式完全不同——人类可以在协作中随时中断、插话、修正方向。

BiCNet[16](#ref-16) 的核心贡献正是证明了：**实时双向通信**能涌现出远超异步协作的协调质量。但在大模型智能体场景中，这些可学习通信方法的迁移几乎是空白。当前大模型智能体间传递的是自然语言或 JSON——信息密度低、冗余高。是否可以设计更结构化、更高效的智能体通信协议？

再叠加上文讨论的运行框架问题——多智能体系统中每个智能体都需要稳定的运行框架，智能体间的协调也需要容错机制。**通信协议的效率和运行框架的鲁棒性，共同构成了多智能体协作的基础设施层。**

### 共享技能：超越参数共享的协作机制

如果多个智能体各自探索后发现了有效的技能，如何高效共享？

直接共享上下文太重，共享蒸馏后的参数太慢。中间态——共享结构化的技能描述符——可能是更实际的方案。这和第二节中记忆到技能的压缩路径有天然联系：如果技能可以被提炼为紧凑的描述符，那么智能体间的技能共享就可以高带宽、低延迟地进行。

Tran et al. (2025)[23](#ref-23) 的综述将大模型多智能体协作机制分为通信、角色分配、记忆共享三个维度。一篇近期预印本明确提出了"受控异构性"的概念——"小而快的模型负责广泛探索，大而慎重的模型负责最终验证"[24](#ref-24)。这和我们在种群级训练中观察到的模式一致：**异构性不是协作的障碍，而是协作的价值来源。**


## 四、三个方向的交叉

这三个问题并非孤立：

**环境规模化 × 多智能体。** 多个异构智能体可以并行探索不同环境，用智能体的异构性弥补环境多样性的不足。我们在 MALib[19](#ref-19) 中已经验证了种群级并行探索的可行性——将其从博弈场景扩展到大模型智能体的环境探索是自然的下一步。

**持续进化 × 环境规模化。** 如果智能体能在部署中持续进化，那它实际接触的用户场景本身就是一种"环境规模化"——这模糊了训练和部署的边界。MemRL[14](#ref-14) 的运行时自进化机制和 DPT[28](#ref-28) 的双过程持续运行架构，分别从算法和系统两个层面支撑了这个方向。

**多智能体 × 持续进化。** 多个智能体的共享技能可以加速持续学习——一个智能体学到的新能力通过技能共享传播给其他智能体，减少重复学习。但同时引入新的非平稳性：其他智能体的策略变化也是环境变化的一部分。这和多智能体强化学习中的经典非平稳性问题有深层联系。


## 结语

更深一层看，这三个方向的交叉实际上指向一个统一的框架。我们在近期的工作中提出了"语言博弈"（Language Games）的视角[29](#ref-29)：通过将人类与智能体的交互建模为开放式的语言博弈，人机协同进化产生无界的数据流，驱动开放式探索。在这个框架下：

- **环境规模化**不再需要人工构建——语言博弈本身就是一个不断生成新环境的过程；
- **持续进化**是博弈动力学的自然结果——每一轮交互都产生新的训练信号；
- **多智能体协作**是博弈的基本结构——人类和多个智能体共同构成博弈的参与者。

这个视角重新定义了数据再生产：不是一个封闭循环，而是一个驱动开放式探索的引擎。当语言博弈的规模从实验室扩展到全球社会技术生态系统时，人机协同进化可能成为通向超人智能的路径。

Sutton 写《苦涩的教训》时说的是：搜索和学习终将胜过人类知识工程。

现在我们可能正在见证这个教训的智能体版本：**让智能体在与世界和人类的博弈中学习——通过环境交互、持续进化、多体协作——终将胜过我们手工设计智能体的行为。**

环境规模化决定了智能体能力的广度，持续进化决定了智能体能力的持久性，多智能体协作决定了智能体能力的上限。语言博弈为这三者提供了统一的理论框架。这条路径的交叉推进，可能是通向通用智能体的关键。



## 参考文献

<div class="text-sm leading-relaxed">


<span id="ref-1">**[1]**</span> A. Karpathy, "2025 LLM Year in Review," karpathy.bearblog.dev, 2025 年 12 月.


<span id="ref-2">**[2]**</span> D. Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," *Nature*, 2025. arXiv:2501.12948.


<span id="ref-3">**[3]**</span> OpenAI, "Learning to Reason with LLMs," openai.com, 2024 年 9 月.


<span id="ref-4">**[4]**</span> "Environment Scaling for Interactive Agentic Experience Collection: A Survey," arXiv:2511.09586, 2025 年 12 月.


<span id="ref-5">**[5]**</span> "Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning," arXiv:2602.10090, 2026 年 2 月.


<span id="ref-6">**[6]**</span> "ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training," arXiv:2602.06820, 2026 年 2 月.


<span id="ref-7">**[7]**</span> "AGENTRL: Scaling Agentic Reinforcement Learning," arXiv:2510.04206, 2025 年 10 月.


<span id="ref-8">**[8]**</span> Wing Venture Capital, "RL Environments for Agentic AI: Who Will Win the Training & Verification Layer by 2030," 2026 年 1 月.


<span id="ref-9">**[9]**</span> S. Zhang et al., "ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination," *NeurIPS*, 2024.


<span id="ref-10">**[10]**</span> X. Feng et al., "Alphazero-like Tree-Search Can Guide Large Language Model Decoding and Training (TS-LLM)," *ICML*, 2024.


<span id="ref-11">**[11]**</span> Y. Li et al., "Cooperative Open-ended Learning Framework for Zero-shot Coordination (COLE)," *ICML*, 2024. arXiv:2302.04831.


<span id="ref-12">**[12]**</span> C. Wolfe, "Continual Learning with RL for LLMs," cameronrwolfe.substack.com, 2026 年 1 月.


<span id="ref-13">**[13]**</span> "Spurious Forgetting in Continual Learning of Language Models," *ICLR*, 2025.


<span id="ref-14">**[14]**</span> S. Zhang et al., "MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory," arXiv:2601.03192, 2026 年 1 月.


<span id="ref-15">**[15]**</span> M. Wen et al., "Reinforcing Language Agents via Policy Optimization with Action Decomposition (POAD)," *NeurIPS*, 2024. arXiv:2405.15821.


<span id="ref-16">**[16]**</span> P. Peng, Y. Wen et al., "Multiagent Bidirectionally-Coordinated Nets (BiCNet)," arXiv:1703.10069, 2017.


<span id="ref-17">**[17]**</span> J. Kuba, M. Wen, Y. Wen et al., "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning (HATRPO/HAPPO)," *ICLR*, 2022. arXiv:2109.11251.


<span id="ref-18">**[18]**</span> M. Wen, J. Kuba, R. Lin, W. Zhang, Y. Wen et al., "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem (MAT)," *NeurIPS*, 2022. arXiv:2205.14953.


<span id="ref-19">**[19]**</span> M. Zhou, Z. Wan, H. Wang, M. Wen, R. Wu, Y. Wen et al., "MALib: A Parallel Framework for Population-based Multi-agent Reinforcement Learning," *JMLR*, 2023. arXiv:2106.07551.


<span id="ref-20">**[20]**</span> Z. Wan et al., "ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning," *NeurIPS*, 2025. arXiv:2503.09501.


<span id="ref-21">**[21]**</span> J. Foerster et al., "Learning to Communicate with Deep Multi-Agent Reinforcement Learning," *NeurIPS*, 2016.


<span id="ref-22">**[22]**</span> S. Sukhbaatar et al., "Learning Multiagent Communication with Backpropagation," *NeurIPS*, 2016.


<span id="ref-23">**[23]**</span> K.-T. Tran et al., "Multi-Agent Collaboration Mechanisms: A Survey of LLMs," arXiv:2501.06322, 2025 年 1 月.


<span id="ref-24">**[24]**</span> "Multi-Agent LLM Systems: From Emergent Collaboration to Structured Collective Intelligence," Preprints.org, 2025 年 11 月.


<span id="ref-25">**[25]**</span> R. Sutton, "The Bitter Lesson," incompleteideas.net, 2019 年 3 月.


<span id="ref-26">**[26]**</span> P. Yu, Z. Zhao, S. Zhang, L. Fu, X. Wang, Y. Wen, "Learning to Reason in Structured In-context Environments with Reinforcement Learning," *ICLR*, 2026. arXiv:2509.23330.


<span id="ref-27">**[27]**</span> Anthropic, "Effective Harnesses for Long-Running Agents," anthropic.com, 2025 年 11 月.


<span id="ref-28">**[28]**</span> S. Zhang, X. Wang et al., "Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration (DPT)," *ACL*, 2025. arXiv:2502.11882.


<span id="ref-29">**[29]**</span> Y. Wen, Z. Wan, S. Zhang, "Language Games as the Pathway to Artificial Superhuman Intelligence," arXiv:2501.18924, 2025 年 1 月.

</div>
