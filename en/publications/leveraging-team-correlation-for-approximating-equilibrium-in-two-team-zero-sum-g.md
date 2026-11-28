# Leveraging Team Correlation for Approximating Equilibrium in Two-Team Zero-Sum Games

Authors: Naming Liu; Mingzhi Wang; Youzhi Zhang; Yaodong Yang; Bo An; Ying Wen

Publication: arXiv (2024)
Type: Preprint
Canonical page: https://yingwen.io/en/publications/leveraging-team-correlation-for-approximating-equilibrium-in-two-team-zero-sum-g/
Publisher / primary source: https://arxiv.org/abs/2403.00255
DOI: https://doi.org/10.48550/arXiv.2403.00255
arXiv: https://arxiv.org/abs/2403.00255
PDF: https://arxiv.org/pdf/2403.00255
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Two-team zero-sum games are one of the most important paradigms in game theory. In this paper, we focus on finding an unexploitable equilibrium in large team games. An unexploitable equilibrium is a worst-case policy, where members in the opponent team cannot increase their team reward by taking any policy, e.g., cooperatively changing to other joint policies. As an optimal unexploitable equilibrium in two-team zero-sum games, correlated-team maxmin equilibrium remains unexploitable even in the worst case where players in the opponent team can achieve arbitrary cooperation through a joint team policy. However, finding such an equilibrium in large games is challenging due to the impracticality of evaluating the exponentially large number of joint policies. To solve this problem, we first introduce a general solution concept called restricted correlated-team maxmin equilibrium, which solves the problem of being impossible to evaluate all joint policy by a sample factor while avoiding an exploitation problem under the incomplete joint policy evaluation. We then develop an efficient sequential correlation mechanism, and based on which we propose an algorithm for approximating the unexploitable equilibrium in large games. We show that our approach achieves lower exploitability than the state-of-the-art baseline when encountering opponent teams with different exploitation ability in large team games including Google Research Football.

## Citation

```bibtex
@misc{liu2024-leveraging-team-correlation-for-approximating,
  title = {{Leveraging Team Correlation for Approximating Equilibrium in Two-Team Zero-Sum Games}},
  author = {Naming Liu and Mingzhi Wang and Youzhi Zhang and Yaodong Yang and Bo An and Ying Wen},
  year = {2024},
  url = {https://arxiv.org/abs/2403.00255},
  eprint = {2403.00255},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2403.00255}
}
```

BibTeX download: https://yingwen.io/citations/leveraging-team-correlation-for-approximating-equilibrium-in-two-team-zero-sum-g.bib
