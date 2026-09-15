# Computing ex ante equilibrium in heterogeneous zero-sum team game

Authors: Naming Liu; Mingzhi Wang; Xihuai Wang; Weinan Zhang; Yaodong Yang; Youzhi Zhang; Bo An; Ying Wen

Publication: Frontiers of Computer Science (2026)
Type: Journal article
Canonical page: https://yingwen.io/zh/publications/computing-ex-ante-equilibrium-in-heterogeneous-zero-sum-team-game/
Publisher / primary source: https://doi.org/10.1007/s11704-025-51045-0
DOI: https://doi.org/10.1007/s11704-025-51045-0
arXiv: https://arxiv.org/abs/2410.01575
PDF: https://arxiv.org/pdf/2410.01575
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

The ex ante equilibrium for two-team zero-sum games, where agents within each team collaborate to compete against the opposing team, is known to be the best a team can do for coordination. Many existing works on ex ante equilibrium solutions are aiming to extend the scope of ex ante equilibrium solving to large-scale team games based on Policy Space Response Oracle (PSRO). However, the joint team policy space constructed by the most prominent method, Team PSRO, cannot cover the entire team policy space in heterogeneous team games where teammates play distinct roles. Such insufficient policy expressiveness causes Team PSRO to be trapped into a sub-optimal ex ante equilibrium with significantly higher exploitability and never converges to the global ex ante equilibrium. To find the global ex ante equilibrium without introducing additional computational complexity, we first parameterize heterogeneous policies for teammates, and we prove that optimizing the heterogeneous teammates' policies sequentially can guarantee a monotonic improvement in team rewards. We further propose Heterogeneous-PSRO (H-PSRO), a novel framework for heterogeneous team games, which integrates the sequential correlation mechanism into the PSRO framework and serves as the first PSRO framework for heterogeneous team games. We prove that H-PSRO achieves lower exploitability than Team PSRO in heterogeneous team games. Empirically, H-PSRO achieves convergence in matrix heterogeneous games that are unsolvable by non-heterogeneous baselines. Further experiments reveal that H-PSRO outperforms non-heterogeneous baselines in both heterogeneous team games and homogeneous settings.

## Citation

```bibtex
@article{liu2026-computing-ex-ante-equilibrium-in-heterogeneou,
  title = {{Computing ex ante equilibrium in heterogeneous zero-sum team game}},
  author = {Naming Liu and Mingzhi Wang and Xihuai Wang and Weinan Zhang and Yaodong Yang and Youzhi Zhang and Bo An and Ying Wen},
  year = {2026},
  journal = {Frontiers of Computer Science},
  volume = {20},
  number = {11},
  pages = {2011364},
  publisher = {Springer Science and Business Media LLC},
  doi = {10.1007/s11704-025-51045-0},
  url = {https://doi.org/10.1007/s11704-025-51045-0}
}
```

BibTeX download: https://yingwen.io/citations/computing-ex-ante-equilibrium-in-heterogeneous-zero-sum-team-game.bib
