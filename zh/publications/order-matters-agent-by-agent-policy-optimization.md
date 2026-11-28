# Order Matters: Agent-by-agent Policy Optimization

Authors: Xihuai Wang; Zheng Tian; Ziyu Wan; Ying Wen; Jun Wang; Weinan Zhang

Publication: ICLR (2023)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/order-matters-agent-by-agent-policy-optimization/
Publisher / primary source: https://openreview.net/forum?id=Q-neeWNVv1
arXiv: https://arxiv.org/abs/2302.06205
PDF: https://arxiv.org/pdf/2302.06205
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

While multi-agent trust region algorithms have achieved great success empirically in solving coordination tasks, most of them, however, suffer from a non-stationarity problem since agents update their policies simultaneously. In contrast, a sequential scheme that updates policies agent-by-agent provides another perspective and shows strong performance. However, sample inefficiency and lack of monotonic improvement guarantees for each agent are still the two significant challenges for the sequential scheme. In this paper, we propose the \textbf{A}gent-by-\textbf{a}gent \textbf{P}olicy \textbf{O}ptimization (A2PO) algorithm to improve the sample efficiency and retain the guarantees of monotonic improvement for each agent during training. We justify the tightness of the monotonic improvement bound compared with other trust region algorithms. From the perspective of sequentially updating agents, we further consider the effect of agent updating order and extend the theory of non-stationarity into the sequential update scheme. To evaluate A2PO, we conduct a comprehensive empirical study on four benchmarks: StarCraftII, Multi-agent MuJoCo, Multi-agent Particle Environment, and Google Research Football full game scenarios. A2PO consistently outperforms strong baselines.

## Citation

```bibtex
@inproceedings{wang2023-order-matters-agent-by-agent-policy-optimizat,
  title = {{Order Matters: Agent-by-agent Policy Optimization}},
  author = {Xihuai Wang and Zheng Tian and Ziyu Wan and Ying Wen and Jun Wang and Weinan Zhang},
  year = {2023},
  booktitle = {International Conference on Learning Representations},
  url = {https://openreview.net/forum?id=Q-neeWNVv1}
}
```

BibTeX download: https://yingwen.io/citations/order-matters-agent-by-agent-policy-optimization.bib
