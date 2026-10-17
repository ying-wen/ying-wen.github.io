# DCAC: Reducing Unnecessary Conservatism in Offline-to-online Reinforcement Learning

Authors: Dongxiang Chen; Ying Wen

Publication: DAI (2023)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/dcac-reducing-unnecessary-conservatism-in-offline-to-online-reinforcement-learning/
Publisher / primary source: https://doi.org/10.1145/3627676.3627677
DOI: https://doi.org/10.1145/3627676.3627677
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/)

## Abstract

Recent advancements in offline reinforcement learning (RL) have facilitated the training of powerful agents using fixed datasets exclusively. Despite this, the quality of a dataset plays a critical role in determining an agent’s performance, and high-quality datasets are often scarce. This scarcity necessitates the enhancement of agents through subsequent environmental interactions. Particularly, the state-action distribution shift may exert a potentially detrimental effect on well-initialized policies, thus impeding the straightforward application of off-policy RL algorithms to policies trained offline. Predominant offline-to-online RL approaches are typically founded on conservatism, a characteristic that may inadvertently confine the asymptotic performance. In response, we propose a method referred to as Dynamically Constrained Actor-Critic (DCAC), grounded in the mathematical form of dynamically constrained policy optimization. This innovative method enables judicious adjustments to the constraints on policy optimization in accordance with a specified rule, thus stabilizing the initial online learning stage and reducing undue conservatism that restricts asymptotic performance. Through comprehensive experimentation across diverse locomotion tasks, we have ascertained that our method successfully improves the policies trained offline with various datasets via subsequent online environmental interactions. The empirical results substantiate that our method mitigates the harmful effects of distribution shift and consistently attains superior asymptotic performance in comparison to prior works.

## Citation

```bibtex
@inproceedings{chen2023-dcac-reducing-unnecessary-conservatism-in-off,
  title = {{DCAC: Reducing Unnecessary Conservatism in Offline-to-online Reinforcement Learning}},
  author = {Dongxiang Chen and Ying Wen},
  year = {2023},
  booktitle = {The Fifth International Conference on Distributed Artificial Intelligence},
  pages = {1--12},
  publisher = {ACM},
  doi = {10.1145/3627676.3627677},
  url = {https://doi.org/10.1145/3627676.3627677}
}
```

BibTeX download: https://yingwen.io/citations/dcac-reducing-unnecessary-conservatism-in-offline-to-online-reinforcement-learning.bib
