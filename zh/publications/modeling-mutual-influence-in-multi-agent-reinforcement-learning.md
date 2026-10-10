# Modeling mutual influence in multi-agent reinforcement learning

Authors: Ying Wen

Publication: University College London (doctoral thesis) (2020)
Type: Doctoral thesis
Canonical page: https://yingwen.io/zh/publications/modeling-mutual-influence-in-multi-agent-reinforcement-learning/
Publisher / primary source: https://ethos.bl.uk/OrderDetails.do?uin=uk.bl.ethos.825477
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

In multi-agent systems (MAS), agents rarely act in isolation but tend to achieve their goals through interactions with other agents. To be able to achieve their ultimate goals, individual agents should actively evaluate the impacts on themselves of other agents' behaviors before they decide which actions to take. The impacts are reciprocal, and it is of great interest to model the mutual influence of agent's impacts with one another when they are observing the environment or taking actions in the environment. In this thesis, assuming that the agents are aware of each other's existence and their potential impact on themselves, I develop novel multi-agent reinforcement learning (MARL) methods that can measure the mutual influence between agents to shape learning. The first part of this thesis outlines the framework of recursive reasoning in deep multi-agent reinforcement learning. I hypothesize that it is beneficial for each agent to consider how other agents react to their behavior. I start from Probabilistic Recursive Reasoning (PR2) using level-1 reasoning and adopt variational Bayes methods to approximate the opponents' conditional policies. Each agent shapes the individual Q-value by marginalizing the conditional policies in the joint Q-value and finding the best response to improving their policies. I further extend PR2 to Generalized Recursive Reasoning (GR2) with different hierarchical levels of rationality. GR2 enables agents to possess various levels of thinking ability, thereby allowing higher-level agents to best respond to less sophisticated learners. The first part of the thesis shows that eliminating the joint Q-value to an individual Q-value via explicitly recursive reasoning would benefit the learning. In the second part of the thesis, in reverse, I measure the mutual influence by approximating the joint Q-value based on the individual Q-values. I establish Q-DPP, an extension of the Determinantal Point Process (DPP) with partition constraints, and apply it to multi-agent learning as a function approximator for the centralized value function. An attractive property of using Q-DPP is that when it reaches the optimum value, it can offer a natural factorization of the centralized value function, representing both quality (maximizing reward) and diversity (different behaviors). In the third part of the thesis, I depart from the action-level mutual influence and build a policy-space meta-game to analyze agents' relationship between adaptive policies. I present a Multi-Agent Trust Region Learning (MATRL) algorithm that augments single-agent trust region policy optimization with a weak stable fixed point approximated by the policy-space meta-game. The algorithm aims to find a game-theoretic mechanism to adjust the policy optimization steps that force the learning of all agents toward the stable point.

## Citation

```bibtex
@phdthesis{wen2020-modeling-mutual-influence-in-multi-agent-rein,
  title = {{Modeling mutual influence in multi-agent reinforcement learning}},
  author = {Ying Wen},
  year = {2020},
  school = {University College London},
  publisher = {University College London},
  url = {https://ethos.bl.uk/OrderDetails.do?uin=uk.bl.ethos.825477}
}
```

BibTeX download: https://yingwen.io/citations/modeling-mutual-influence-in-multi-agent-reinforcement-learning.bib
