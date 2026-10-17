# Fusion-PSRO: Nash Policy Fusion for Policy Space Response Oracles

Authors: Jiesong Lian; Yucong Huang; Chengdong Ma; Mingzhi Wang; Ying Wen; Long Hu; Yixue Hao

Publication: ECAI (2025)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/fusion-psro-nash-policy-fusion-for-policy-space-response-oracles/
Publisher / primary source: https://doi.org/10.3233/faia251106
DOI: https://doi.org/10.3233/faia251106
arXiv: https://arxiv.org/abs/2405.21027
PDF: https://arxiv.org/pdf/2405.21027
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

For solving zero-sum games involving non-transitivity, a useful approach is to maintain a policy population to approximate the Nash Equilibrium (NE). Previous studies have shown that the Policy Space Response Oracles (PSRO) algorithm is an effective framework for solving such games. However, current methods initialize a new policy from scratch or inherit a single historical policy for Best Response (BR), missing the opportunity to leverage past policies to generate a better BR. In this paper, we propose Fusion-PSRO, which employs Nash Policy Fusion to initialize a new policy for BR training. Nash Policy Fusion serves as an implicit guiding policy that starts exploration on the current Meta-NE, thus providing a closer approximation to BR. Moreover, it insightfully captures a weighted moving average of past policies, dynamically adjusting these weights based on the Meta-NE in each iteration. This cumulative process further enhances the policy population. Empirical results on classic benchmarks show that Fusion-PSRO achieves lower exploitability, thereby mitigating the shortcomings of previous research on policy initialization in BR

## Citation

```bibtex
@inproceedings{lian2025-fusion-psro-nash-policy-fusion-for-policy-spa,
  title = {{Fusion-PSRO: Nash Policy Fusion for Policy Space Response Oracles}},
  author = {Jiesong Lian and Yucong Huang and Chengdong Ma and Mingzhi Wang and Ying Wen and Long Hu and Yixue Hao},
  year = {2025},
  booktitle = {Frontiers in Artificial Intelligence and Applications},
  pages = {2562--2569},
  publisher = {IOS Press},
  doi = {10.3233/faia251106},
  url = {https://doi.org/10.3233/faia251106}
}
```

BibTeX download: https://yingwen.io/citations/fusion-psro-nash-policy-fusion-for-policy-space-response-oracles.bib
