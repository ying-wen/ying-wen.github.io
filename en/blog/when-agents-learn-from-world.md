# When Agents Learn from the World, Not from Us

Author: Ying Wen
Language: en
Published: 2026-03-15
Canonical URL: https://yingwen.io/en/blog/when-agents-learn-from-world/
Tags: agentic-rl, environment-scaling, multi-agent, continual-learning
Description: The pretraining paradigm scaled data. The agent paradigm scales environments. Three observations on environment scaling, continuous evolution, and multi-agent collaboration.

*Ying Wen · March 2026*

> The central insight of the pretraining paradigm is that scaling data can produce emergent intelligence. But as we shift from "models" to "agents," a dual insight is forming—**scaling environments can also produce emergent intelligence, and perhaps a more general form of it**. This post develops three observations: how environments scale, how agents continuously evolve, and how multiple agents push beyond single-agent ceilings through collaboration.


## 1. Environment Scaling: The Bitter Lesson for the Agent Era

In 2019, Rich Sutton wrote *The Bitter Lesson* [25](#ref-25): general methods that leverage computation will ultimately outperform methods that encode human knowledge. The pretraining era bore this out—scaling data and compute yielded capabilities no one explicitly programmed (Kaplan et al., 2020).

The same logic is now replaying in LLM-based agentic RL, except the target of scaling has shifted from **data** to **environments**.

In 2025, Karpathy identified Reinforcement Learning from Verifiable Rewards (RLVR) as the "new major stage" in the LLM training pipeline [1](#ref-1). DeepSeek-R1 used pure RL to elicit reasoning without human-annotated demonstrations, with systematic validation in *Nature* [2](#ref-2). OpenAI's o1/o3 series similarly relies on verifiable rewards across diverse environments [3](#ref-3).

But RLVR's successes remain concentrated in **closed domains**—mathematical proofs, coding competitions, formal reasoning—where reward signals are naturally verifiable. When agentic RL extends to open domains (web navigation, API orchestration, multi-step decision-making), **environment construction itself becomes the bottleneck**.

A late-2025 survey addressed this directly, noting that "scaling the diversity of environments and tasks is crucial for preventing agents from overfitting to specific action patterns" [4](#ref-4). Two February 2026 papers—Agent World Model [5](#ref-5) and ScaleEnv [6](#ref-6)—explored automated environment synthesis via code generation. AGENTRL [7](#ref-7) attempted to establish a unified scaling framework for agentic RL. Wing Venture Capital projects Anthropic's RL environment spending to grow 3–5× in 2026 [8](#ref-8).

### Why Environment Scaling Is Harder Than Data Scaling

The core operation of data scaling is collection and cleaning. Environment scaling involves three more complex dimensions:

**The tension between diversity and fidelity.** Synthetic environments can achieve large numbers easily, but struggle to match the distribution of real user needs. Enterprise-grade long-horizon tasks (tens to hundreds of steps) lack good evaluation benchmarks, let alone training environments. Our work on ZSC-Eval [9](#ref-9) established a systematic evaluation framework for multi-agent zero-shot coordination, yet even in this relatively constrained setting, covering the diversity of real collaborative scenarios remains a major challenge.

**Environments need structure.** Naively accumulating random environments is not enough—agents need to learn reasoning in **structured** environments. We studied this directly in recent work: using RL to train models to reason within structured in-context environments (Yu, Zhao, Zhang et al., ICLR 2026) [26](#ref-26). Experiments showed that when environments have exploitable internal structure, RL training efficiency and generalization significantly outperform unstructured baselines. The implication: **environment scaling is not just about quantity, but about structure.**

**"RL-amenable" base models.** However rich the environments, if the base model lacks basic world-modeling capabilities and self-verification mechanisms, the ceiling for RL post-training remains low. Our experiments with TS-LLM [10](#ref-10) showed that introducing AlphaZero-style tree search into LLM decoding and training can systematically improve reasoning quality—but only when the base model itself is sufficiently "search-amenable." What properties make a base model best suited for RL post-training in rich environments? This question has seen almost no systematic investigation.


## 2. Continuous Evolution: When the Agent's Environment Is No Longer External

In the second half of 2025, LLM capabilities crossed an inflection point—from toy demos to genuine productivity tools. But deployment in vertical domains revealed a pervasive last-mile problem. This created a clear demand: **enable deployed agents to improve continuously in service—not through retraining, but through online evolution.**

Continuous evolution is not purely an algorithmic challenge. **Whether agents can operate stably over long periods in real environments** is itself a formidable engineering problem.

### Harnesses: The Infrastructure of Continuous Evolution

In late 2025, Anthropic published *Effective Harnesses for Long-Running Agents* [27](#ref-27), systematically describing solutions for agents working across multiple context windows: an initializer agent sets up the environment, while a worker agent makes incremental progress per session, leaving clear artifacts for the next session.

The article surfaced an underappreciated reality: **the continuity bottleneck in current agent systems is often not model capability, but harness robustness.** Network interruptions, API errors, context overflow—any of these can cause a long-running agent to lose all progress.

This problem is amplified in multi-agent settings. If long-horizon stability is hard to guarantee for a single agent, coordinated operation of multiple agents becomes even more fragile—one agent's crash can break the entire collaboration chain. **Harnesses are not merely a single-agent concern; they are foundational infrastructure for multi-agent systems.**

We approached this from a cognitive science perspective in DPT (Dual Process Theory) [28](#ref-28): drawing on the human distinction between fast cognition (System 1) and deliberate reasoning (System 2), we designed a language agent framework for real-time human-AI collaboration that runs continuously and stably. The fast process handles real-time responses and immediate decisions; the slow process handles deep reasoning and strategy adjustment. This architecture is inherently more robust—the fast process prevents stalling due to slow-process latency, while the slow process prevents degradation from fast-process approximations.

### Memory-to-Skill Compression: A Pragmatic Path to Evolution

Given a stable harness, what is the core mechanism of continuous evolution?

Our recent work MemRL [14](#ref-14) approaches this from a specific angle: explicitly decoupling the agent's **stable cognitive reasoning** from **dynamic episodic memory**—model parameters remain fixed, while RL optimizes episodic memory retrieval and utilization for self-evolution. This sidesteps catastrophic forgetting risks from direct parameter modification while retaining the capacity for continuous learning.

But memory is only an intermediate state. Data accumulated during agent deployment is noisy, expensive, and redundant. The more complete path is:

```
Memory (raw) → Atomic Skills (extraction) → Composite Skills (composition) → Parameter Internalization (distillation)
```

**Forward skill distillation**: extract atomic skills from execution trajectories, discover frequent co-occurrence patterns, compose higher-level skills. **Backward skill distillation**: as the skill library grows, distill verified skills back into model parameters, freeing context budget.

This forms a closed loop: **accumulate memory at runtime → compress into skills → internalize into parameters → free context → continue accumulating new memory.**

### Credit Assignment in Long-Horizon Trajectories

Another key obstacle to continuous evolution: how to correctly assign credit in long-horizon agent trajectories?

In POAD [15](#ref-15), we proposed an action decomposition method: decomposing each agent action into intra-action and inter-action levels, applying Bellman backup at each granularity. Experiments showed that this finer-grained credit assignment significantly improves learning efficiency in multi-step tasks.

This corroborates current RLVR practice. The success of DeepSeek-R1 and OpenAI o1 relies on outcome-level rather than process-level rewards—coarse final rewards plus the model's own credit assignment capability prove more scalable than hand-designed intermediate rewards. POAD's contribution is: **improving RL's own attribution precision through action-structural decomposition, without introducing hand-designed intermediate rewards.**


## 3. Multi-Agent Collaboration: From Task Decomposition to Exploration Diversity

When a single model reaches its capability ceiling through self-play or self-improvement, a natural question arises: **can introducing multiple heterogeneous agents push beyond this ceiling?**

My observation is that the core value of multi-agent collaboration in the LLM setting **is not about task decomposition, but about diversity of exploration**.

Each agent has its own model, context, and environment configuration, with distinct capability boundaries. Directions one agent cannot explore may be naturally covered by another—owing to a different base model, different accumulated context, or different environment interaction history. This aligns with insights we have long accumulated in multi-agent reinforcement learning.

### From Communication Protocols to Reasoning Collaboration

As early as 2017, we proposed BiCNet [16](#ref-16), introducing bidirectionally-coordinated networks for multi-agent systems and demonstrating that real-time inter-agent communication can produce human-level coordination strategies—without human demonstrations. Foerster et al. (2016) [21](#ref-21) and Sukhbaatar et al. (2016) [22](#ref-22) validated related ideas from differentiable communication and backpropagation-based communication perspectives, respectively.

Since then, we have pursued two main lines:

**Theoretical guarantees for collaboration.** HATRPO/HAPPO [17](#ref-17) proved that heterogeneous agents can perform decentralized training with guaranteed monotonic improvement. MAT [18](#ref-18) further recast multi-agent collaboration as a sequence modeling problem, replacing explicit communication protocol design with Transformer attention mechanisms.

**Scaling collaboration.** MALib [19](#ref-19) is the infrastructure we built for large-scale population-based evolutionary training, supporting multi-population parallel evolution. COLE [11](#ref-11) introduced open-ended learning on top of this, enabling cooperative strategies to evolve automatically in open environments.

**From games to reasoning.** ReMA [20](#ref-20) marked this line's extension from traditional multi-agent games to LLM reasoning—multiple agents handle reasoning, reflection, and verification respectively, trained via multi-agent RL to achieve "meta-thinking." This demonstrated that multi-agent collaboration can improve not just action-level but also reasoning-level capabilities.

DPT [28](#ref-28) complements this from the human-AI collaboration angle—when collaboration extends from agent-to-agent to agent-to-human, real-time responsiveness and robustness become critical. The dual-process architecture enables agents to maintain stable collaborative capabilities during real-time human interaction.

### Communication Bandwidth: An Underestimated Bottleneck

I believe current multi-agent LLM systems have an underestimated bottleneck: **extremely low inter-agent communication bandwidth.**

Mainstream multi-agent collaboration patterns today—MapReduce-style parallelism, shared context, serial pipelines—are fundamentally **asynchronous and coarse-grained**. Agent A completes a task and passes results to Agent B, with no real-time interaction in between. This differs fundamentally from human team collaboration, where participants can interrupt, interject, and redirect at any point.

BiCNet's [16](#ref-16) core contribution was precisely demonstrating that **real-time bidirectional communication** yields coordination quality far exceeding asynchronous approaches. But in the LLM agent setting, adaptation of learnable communication methods remains almost entirely unexplored. Current LLM agents communicate via natural language or JSON—low information density, high redundancy. Can we design more structured, higher-bandwidth inter-agent communication protocols?

Combined with the harness discussion above—every agent in a multi-agent system needs a stable harness, and inter-agent coordination also requires fault-tolerance mechanisms. **Communication protocol efficiency and harness robustness together form the infrastructure layer for multi-agent collaboration.**

### Shared Skills: Collaboration Beyond Parameter Sharing

If multiple agents each discover effective skills through independent exploration, how can they share them efficiently?

Sharing raw context is too heavy; sharing distilled parameters is too slow. An intermediate approach—sharing structured skill descriptors—may be more practical. This connects naturally with the memory-to-skill compression path from Section 2: if skills can be distilled into compact descriptors, inter-agent skill sharing can achieve high bandwidth and low latency.

Tran et al. (2025) [23](#ref-23) categorized LLM multi-agent collaboration mechanisms along three dimensions: communication, role assignment, and memory sharing. A recent preprint explicitly proposed "controlled heterogeneity"—"a small, fast model for broad exploration; a large, careful model for final validation" [24](#ref-24). This mirrors patterns we observed in population-based training: **heterogeneity is not an obstacle to collaboration, but the source of its value.**


## 4. Where the Three Directions Intersect

These three problems are not isolated:

**Environment scaling × multi-agent.** Multiple heterogeneous agents can explore different environments in parallel, using agent heterogeneity to compensate for limited environment diversity. We validated the feasibility of population-level parallel exploration in MALib [19](#ref-19)—extending it from game settings to LLM agent environment exploration is a natural next step.

**Continuous evolution × environment scaling.** If agents can evolve continuously during deployment, the user scenarios they encounter effectively constitute a form of environment scaling—blurring the boundary between training and deployment. MemRL's [14](#ref-14) runtime self-evolution mechanism and DPT's [28](#ref-28) dual-process architecture support this from algorithmic and systems perspectives, respectively.

**Multi-agent × continuous evolution.** Shared skills across multiple agents can accelerate continual learning—a capability learned by one agent propagates to others via skill sharing, reducing redundant learning. But this simultaneously introduces new non-stationarity: other agents' strategy changes are themselves part of the environment shift. This connects deeply with the classic non-stationarity problem in multi-agent reinforcement learning.


## Conclusion

At a deeper level, the intersection of these three directions points toward a unified framework. In recent work, we proposed the perspective of "Language Games" [29](#ref-29): modeling human-agent interaction as open-ended language games, where human-AI co-evolution generates unbounded data streams that drive open-ended exploration. Under this framework:

- **Environment scaling** no longer requires manual construction—language games themselves continuously generate new environments;
- **Continuous evolution** is a natural consequence of game dynamics—each round of interaction produces new training signals;
- **Multi-agent collaboration** is the basic structure of the game—humans and multiple agents together constitute the participants.

This perspective redefines data generation: not as a closed loop, but as an engine for open-ended exploration. When language games scale from laboratories to global sociotechnical ecosystems, human-AI co-evolution may become the pathway to superhuman intelligence.

Sutton's *Bitter Lesson* argued that search and learning will ultimately prevail over human knowledge engineering.

We may now be witnessing the agent-era counterpart: **agents that learn through games with the world and with humans—via environment interaction, continuous evolution, and multi-agent collaboration—will ultimately surpass agents whose behaviors we engineer by hand.**

Environment scaling determines the breadth of agent capabilities, continuous evolution determines their persistence, and multi-agent collaboration determines their ceiling. Language games provide a unified theoretical framework for all three. The joint advancement of these directions may be the critical path toward general-purpose agents.



## References

<div class="text-sm leading-relaxed">

<span id="ref-1">**[1]**</span> A. Karpathy, "2025 LLM Year in Review," karpathy.bearblog.dev, Dec 2025.

<span id="ref-2">**[2]**</span> D. Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," *Nature*, 2025. arXiv:2501.12948.

<span id="ref-3">**[3]**</span> OpenAI, "Learning to Reason with LLMs," openai.com, Sep 2024.

<span id="ref-4">**[4]**</span> "Environment Scaling for Interactive Agentic Experience Collection: A Survey," arXiv:2511.09586, Dec 2025.

<span id="ref-5">**[5]**</span> "Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning," arXiv:2602.10090, Feb 2026.

<span id="ref-6">**[6]**</span> "ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training," arXiv:2602.06820, Feb 2026.

<span id="ref-7">**[7]**</span> "AGENTRL: Scaling Agentic Reinforcement Learning," arXiv:2510.04206, Oct 2025.

<span id="ref-8">**[8]**</span> Wing Venture Capital, "RL Environments for Agentic AI: Who Will Win the Training & Verification Layer by 2030," Jan 2026.

<span id="ref-9">**[9]**</span> S. Zhang et al., "ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination," *NeurIPS*, 2024.

<span id="ref-10">**[10]**</span> X. Feng et al., "Alphazero-like Tree-Search Can Guide Large Language Model Decoding and Training (TS-LLM)," *ICML*, 2024.

<span id="ref-11">**[11]**</span> Y. Li et al., "Cooperative Open-ended Learning Framework for Zero-shot Coordination (COLE)," *ICML*, 2024. arXiv:2302.04831.

<span id="ref-12">**[12]**</span> C. Wolfe, "Continual Learning with RL for LLMs," cameronrwolfe.substack.com, Jan 2026.

<span id="ref-13">**[13]**</span> "Spurious Forgetting in Continual Learning of Language Models," *ICLR*, 2025.

<span id="ref-14">**[14]**</span> S. Zhang et al., "MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory," arXiv:2601.03192, Jan 2026.

<span id="ref-15">**[15]**</span> M. Wen et al., "Reinforcing Language Agents via Policy Optimization with Action Decomposition (POAD)," *NeurIPS*, 2024. arXiv:2405.15821.

<span id="ref-16">**[16]**</span> P. Peng, Y. Wen et al., "Multiagent Bidirectionally-Coordinated Nets (BiCNet)," arXiv:1703.10069, 2017.

<span id="ref-17">**[17]**</span> J. Kuba, M. Wen, Y. Wen et al., "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning (HATRPO/HAPPO)," *ICLR*, 2022. arXiv:2109.11251.

<span id="ref-18">**[18]**</span> M. Wen, J. Kuba, R. Lin, W. Zhang, Y. Wen et al., "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem (MAT)," *NeurIPS*, 2022. arXiv:2205.14953.

<span id="ref-19">**[19]**</span> M. Zhou, Z. Wan, H. Wang, M. Wen, R. Wu, Y. Wen et al., "MALib: A Parallel Framework for Population-based Multi-agent Reinforcement Learning," *JMLR*, 2023. arXiv:2106.07551.

<span id="ref-20">**[20]**</span> Z. Wan et al., "ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning," *NeurIPS*, 2025. arXiv:2503.09501.

<span id="ref-21">**[21]**</span> J. Foerster et al., "Learning to Communicate with Deep Multi-Agent Reinforcement Learning," *NeurIPS*, 2016.

<span id="ref-22">**[22]**</span> S. Sukhbaatar et al., "Learning Multiagent Communication with Backpropagation," *NeurIPS*, 2016.

<span id="ref-23">**[23]**</span> K.-T. Tran et al., "Multi-Agent Collaboration Mechanisms: A Survey of LLMs," arXiv:2501.06322, Jan 2025.

<span id="ref-24">**[24]**</span> "Multi-Agent LLM Systems: From Emergent Collaboration to Structured Collective Intelligence," Preprints.org, Nov 2025.

<span id="ref-25">**[25]**</span> R. Sutton, "The Bitter Lesson," incompleteideas.net, Mar 2019.

<span id="ref-26">**[26]**</span> P. Yu, Z. Zhao, S. Zhang, L. Fu, X. Wang, Y. Wen, "Learning to Reason in Structured In-context Environments with Reinforcement Learning," *ICLR*, 2026. arXiv:2509.23330.

<span id="ref-27">**[27]**</span> Anthropic, "Effective Harnesses for Long-Running Agents," anthropic.com, Nov 2025.

<span id="ref-28">**[28]**</span> S. Zhang, X. Wang et al., "Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration (DPT)," *ACL*, 2025. arXiv:2502.11882.

<span id="ref-29">**[29]**</span> Y. Wen, Z. Wan, S. Zhang, "Language Games as the Pathway to Artificial Superhuman Intelligence," arXiv:2501.18924, Jan 2025.

</div>
