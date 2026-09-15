# PMAT: Optimizing Action Generation Order in Multi-Agent Reinforcement Learning

Authors: Kun Hu; Muning Wen; Xihuai Wang; Shao Zhang; Yiwei Shi; Minne Li; Minglong Li; Ying Wen

Publication: AAMAS 2025 (2025)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/pmat-aamas-2025/
Publisher / primary source: https://dl.acm.org/doi/10.5555/3709347.3743619
DOI: https://doi.org/10.5555/3709347.3743619
arXiv: https://arxiv.org/abs/2502.16496
PDF: https://arxiv.org/pdf/2502.16496
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

Multi-agent reinforcement learning (MARL) faces challenges in coordinating agents due to complex interdependencies within multi-agent systems. Most MARL algorithms use the simultaneous decision-making paradigm but ignore the action-level dependencies among agents, which reduces coordination efficiency. In contrast, the sequential decision-making paradigm provides finer-grained supervision for agent decision order, presenting the potential for handling dependencies via better decision order management. However, determining the optimal decision order remains a challenge. In this paper, we introduce Action Generation with Plackett-Luce Sampling (AGPS), a novel mechanism for agent decision order optimization. We model the order determination task as a Plackett-Luce sampling process to address issues such as ranking instability and vanishing gradient during the network training process. AGPS realizes credit-based decision order determination by establishing a bridge between the significance of agents' local observations and their decision credits, thus facilitating order optimization and dependency management. Integrating AGPS with the Multi-Agent Transformer, we propose the Prioritized Multi-Agent Transformer (PMAT), a sequential decision-making MARL algorithm with decision order optimization. Experiments on benchmarks including StarCraft II Multi-Agent Challenge, Google Research Football, and Multi-Agent MuJoCo show that PMAT outperforms state-of-the-art algorithms, greatly enhancing coordination efficiency.

## Citation

```bibtex
@inproceedings{hu2025-pmat-aamas-2025,
  title = {{PMAT: Optimizing Action Generation Order in Multi-Agent Reinforcement Learning}},
  author = {Kun Hu and Muning Wen and Xihuai Wang and Shao Zhang and Yiwei Shi and Minne Li and Minglong Li and Ying Wen},
  year = {2025},
  booktitle = {Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems},
  pages = {997--1005},
  doi = {10.5555/3709347.3743619},
  url = {https://dl.acm.org/doi/10.5555/3709347.3743619}
}
```

BibTeX download: https://yingwen.io/citations/pmat-aamas-2025.bib
