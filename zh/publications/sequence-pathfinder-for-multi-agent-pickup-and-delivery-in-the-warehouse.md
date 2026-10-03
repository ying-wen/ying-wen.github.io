# Sequence Pathfinder for Multi-Agent Pickup and Delivery in the Warehouse

Authors: Zeyuan Zhao; Chaoran Li; Shao Zhang; Ying Wen

Publication: arXiv (2025)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/sequence-pathfinder-for-multi-agent-pickup-and-delivery-in-the-warehouse/
Publisher / primary source: https://arxiv.org/abs/2509.23778
DOI: https://doi.org/10.48550/arXiv.2509.23778
arXiv: https://arxiv.org/abs/2509.23778
PDF: https://arxiv.org/pdf/2509.23778
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/); [具身智能与机器人学习](https://yingwen.io/zh/research/embodied-ai/)

## Abstract

Multi-Agent Pickup and Delivery (MAPD) is a challenging extension of Multi-Agent Path Finding (MAPF), where agents are required to sequentially complete tasks with fixed-location pickup and delivery demands. Although learning-based methods have made progress in MAPD, they often perform poorly in warehouse-like environments with narrow pathways and long corridors when relying only on local observations for distributed decision-making. Communication learning can alleviate the lack of global information but introduce high computational complexity due to point-to-point communication. To address this challenge, we formulate MAPF as a sequence modeling problem and prove that path-finding policies under sequence modeling possess order-invariant optimality, ensuring its effectiveness in MAPD. Building on this, we propose the Sequential Pathfinder (SePar), which leverages the Transformer paradigm to achieve implicit information exchange, reducing decision-making complexity from exponential to linear while maintaining efficiency and global awareness. Experiments demonstrate that SePar consistently outperforms existing learning-based methods across various MAPF tasks and their variants, and generalizes well to unseen environments. Furthermore, we highlight the necessity of integrating imitation learning in complex maps like warehouses.

## Citation

```bibtex
@misc{zhao2025-sequence-pathfinder-for-multi-agent-pickup-an,
  title = {{Sequence Pathfinder for Multi-Agent Pickup and Delivery in the Warehouse}},
  author = {Zeyuan Zhao and Chaoran Li and Shao Zhang and Ying Wen},
  year = {2025},
  url = {https://arxiv.org/abs/2509.23778},
  eprint = {2509.23778},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2509.23778}
}
```

BibTeX download: https://yingwen.io/citations/sequence-pathfinder-for-multi-agent-pickup-and-delivery-in-the-warehouse.bib
