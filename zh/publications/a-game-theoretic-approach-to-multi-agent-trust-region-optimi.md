# A Game-Theoretic Approach to Multi-agent Trust Region Optimization

Authors: Ying Wen; Hui Chen; Yaodong Yang; Minne Li; Zheng Tian; Xu Chen; Jun Wang

Publication: DAI (2023)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/a-game-theoretic-approach-to-multi-agent-trust-region-optimi/
Publisher / primary source: https://doi.org/10.1007/978-3-031-25549-6_6
DOI: https://doi.org/10.1007/978-3-031-25549-6_6
arXiv: https://arxiv.org/abs/2106.06828
PDF: https://link.springer.com/content/pdf/10.1007/978-3-031-25549-6_6.pdf
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

Trust region methods are widely applied in single-agent reinforcement learning problems due to their monotonic performance-improvement guarantee at every iteration. Nonetheless, when applied in multi-agent settings, the guarantee of trust region methods no longer holds because an agent’s payoff is also affected by other agents’ adaptive behaviors. To tackle this problem, we conduct a game-theoretical analysis in the policy space, and propose a multi-agent trust region learning method (MATRL), which enables trust region optimization for multi-agent learning. Specifically, MATRL finds a stable improvement direction that is guided by the solution concept of Nash equilibrium at the meta-game level. We derive the monotonic improvement guarantee in multi-agent settings and show the local convergence of MATRL to stable fixed points in differential games. To test our method, we evaluate MATRL in both discrete and continuous multiplayer general-sum games including checker and switch grid worlds, multi-agent MuJoCo, and Atari games. Results suggest that MATRL significantly outperforms strong multi-agent reinforcement learning baselines.

## Citation

```bibtex
@inproceedings{wen2023-a-game-theoretic-approach-to-multi-agent-trus,
  title = {{A Game-Theoretic Approach to Multi-agent Trust Region Optimization}},
  author = {Ying Wen and Hui Chen and Yaodong Yang and Minne Li and Zheng Tian and Xu Chen and Jun Wang},
  year = {2023},
  booktitle = {Distributed Artificial Intelligence},
  pages = {74--87},
  publisher = {Springer, Cham},
  doi = {10.1007/978-3-031-25549-6\_6},
  url = {https://doi.org/10.1007/978-3-031-25549-6_6}
}
```

BibTeX download: https://yingwen.io/citations/a-game-theoretic-approach-to-multi-agent-trust-region-optimi.bib
