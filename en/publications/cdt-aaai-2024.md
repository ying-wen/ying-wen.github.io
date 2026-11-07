# Critic-Guided Decision Transformer for Offline Reinforcement Learning

Authors: Yuanfu Wang; Chao Yang; Ying Wen; Yu Liu; Yu Qiao

Publication: AAAI 2024 (2024)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/cdt-aaai-2024/
Publisher / primary source: https://ojs.aaai.org/index.php/AAAI/article/view/29499
DOI: https://doi.org/10.1609/aaai.v38i14.29499
arXiv: https://arxiv.org/abs/2312.13716
PDF: https://arxiv.org/pdf/2312.13716
Code: https://github.com/sharkwyf/critic-guided-decision-transformer
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/)

## Abstract

Recent advancements in offline reinforcement learning (RL) have underscored the capabilities of Return-Conditioned Supervised Learning (RCSL), a paradigm that learns the action distribution based on target returns for each state in a supervised manner. However, prevailing RCSL methods largely focus on deterministic trajectory modeling, disregarding stochastic state transitions and the diversity of future trajectory distributions. A fundamental challenge arises from the inconsistency between the sampled returns within individual trajectories and the expected returns across multiple trajectories. Fortunately, value-based methods offer a solution by leveraging a value function to approximate the expected returns, thereby addressing the inconsistency effectively. Building upon these insights, we propose a novel approach, termed the Critic-Guided Decision Transformer (CGDT), which combines the predictability of long-term returns from value-based methods with the trajectory modeling capability of the Decision Transformer. By incorporating a learned value function, known as the critic, CGDT ensures a direct alignment between the specified target returns and the expected returns of actions. This integration bridges the gap between the deterministic nature of RCSL and the probabilistic characteristics of value-based methods. Empirical evaluations on stochastic environments and D4RL benchmark datasets demonstrate the superiority of CGDT over traditional RCSL methods. These results highlight the potential of CGDT to advance the state of the art in offline RL and extend the applicability of RCSL to a wide range of RL tasks.

## Citation

```bibtex
@inproceedings{wang2024-cdt-aaai-2024,
  title = {{Critic-Guided Decision Transformer for Offline Reinforcement Learning}},
  author = {Yuanfu Wang and Chao Yang and Ying Wen and Yu Liu and Yu Qiao},
  year = {2024},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {38},
  number = {14},
  pages = {15706--15714},
  doi = {10.1609/aaai.v38i14.29499},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/29499}
}
```

BibTeX download: https://yingwen.io/citations/cdt-aaai-2024.bib
