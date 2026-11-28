# Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning

Authors: Jakub Grudzien Kuba; Ruiqing Chen; Muning Wen; Ying Wen; Fanglei Sun; Jun Wang; Yaodong Yang

Publication: ICLR (2022)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/trust-region-policy-optimisation-in-multi-agent-reinforcemen/
Publisher / primary source: https://openreview.net/forum?id=EcGGFkNTxdJ
arXiv: https://arxiv.org/abs/2109.11251
PDF: https://arxiv.org/pdf/2109.11251
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

Trust region methods rigorously enabled reinforcement learning (RL) agents to learn monotonically improving policies, leading to superior performance on a variety of tasks. Unfortunately, when it comes to multi-agent reinforcement learning (MARL), the property of monotonic improvement may not simply apply; this is because agents, even in cooperative games, could have conflicting directions of policy updates. As a result, achieving a guaranteed improvement on the joint policy where each agent acts individually remains an open challenge. In this paper, we extend the theory of trust region learning to MARL. Central to our findings are the multi-agent advantage decomposition lemma and the sequential policy update scheme. Based on these, we develop Heterogeneous-Agent Trust Region Policy Optimisation (HATPRO) and Heterogeneous-Agent Proximal Policy Optimisation (HAPPO) algorithms. Unlike many existing MARL algorithms, HATRPO/HAPPO do not need agents to share parameters, nor do they need any restrictive assumptions on decomposibility of the joint value function. Most importantly, we justify in theory the monotonic improvement property of HATRPO/HAPPO. We evaluate the proposed methods on a series of Multi-Agent MuJoCo and StarCraftII tasks. Results show that HATRPO and HAPPO significantly outperform strong baselines such as IPPO, MAPPO and MADDPG on all tested tasks, therefore establishing a new state of the art.

## Citation

```bibtex
@inproceedings{kuba2022-trust-region-policy-optimisation-in-multi-age,
  title = {{Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning}},
  author = {Jakub Grudzien Kuba and Ruiqing Chen and Muning Wen and Ying Wen and Fanglei Sun and Jun Wang and Yaodong Yang},
  year = {2022},
  booktitle = {International Conference on Learning Representations},
  url = {https://openreview.net/forum?id=EcGGFkNTxdJ}
}
```

BibTeX download: https://yingwen.io/citations/trust-region-policy-optimisation-in-multi-agent-reinforcemen.bib
