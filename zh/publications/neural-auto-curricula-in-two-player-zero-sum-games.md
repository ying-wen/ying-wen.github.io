# Neural Auto-Curricula in Two-Player Zero-Sum Games

Authors: Xidong Feng; Oliver Slumbers; Ziyu Wan; Bo Liu; Stephen McAleer; Ying Wen; Jun Wang; Yaodong Yang

Publication: NeurIPS (2021)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/neural-auto-curricula-in-two-player-zero-sum-games/
Publisher / primary source: https://proceedings.neurips.cc/paper/2021/hash/1cd73be1e256a7405516501e94e892ac-Abstract.html
arXiv: https://arxiv.org/abs/2106.02745
PDF: https://proceedings.neurips.cc/paper_files/paper/2021/file/1cd73be1e256a7405516501e94e892ac-Paper.pdf
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

When solving two-player zero-sum games, multi-agent reinforcement learning (MARL) algorithms often create populations of agents where, at each iteration, a new agent is discovered as the best response to a mixture over the opponent population. Within such a process, the update rules of "who to compete with" (i.e., the opponent mixture) and "how to beat them" (i.e., finding best responses) are underpinned by manually developed game theoretical principles such as fictitious play and Double Oracle. In this paper, we introduce a novel framework—Neural Auto-Curricula (NAC)—that leverages meta-gradient descent to automate the discovery of the learning update rule without explicit human design. Specifically, we parameterise the opponent selection module by neural networks and the best-response module by optimisation subroutines, and update their parameters solely via interaction with the game engine, where both players aim to minimise their exploitability. Surprisingly, even without human design, the discovered MARL algorithms achieve competitive or even better performance with the state-of-the-art population-based game solvers (e.g., PSRO) on Games of Skill, differentiable Lotto, non-transitive Mixture Games, Iterated Matching Pennies, and Kuhn Poker. Additionally, we show that NAC is able to generalise from small games to large games, for example training on Kuhn Poker and outperforming PSRO on Leduc Poker. Our work inspires a promising future direction to discover general MARL algorithms solely from data.

## Citation

```bibtex
@inproceedings{NEURIPS2021_1cd73be1,
  title = {{Neural Auto-Curricula in Two-Player Zero-Sum Games}},
  author = {Xidong Feng and Oliver Slumbers and Ziyu Wan and Bo Liu and Stephen McAleer and Ying Wen and Jun Wang and Yaodong Yang},
  year = {2021},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
  pages = {3504--3517},
  publisher = {Curran Associates, Inc.},
  url = {https://proceedings.neurips.cc/paper/2021/hash/1cd73be1e256a7405516501e94e892ac-Abstract.html}
}
```

BibTeX download: https://yingwen.io/citations/neural-auto-curricula-in-two-player-zero-sum-games.bib
