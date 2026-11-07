# Offline Fictitious Self-Play for Competitive Games

Authors: Jingxiao Chen; Weiji Xie; Weinan Zhang; Yong Yu; Ying Wen

Publication: AAAI 2026 (2026)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/offline-fictitious-self-play-for-competitive-games/
Publisher / primary source: https://doi.org/10.1609/aaai.v40i24.39098
DOI: https://doi.org/10.1609/aaai.v40i24.39098
arXiv: https://arxiv.org/abs/2403.00841
PDF: https://arxiv.org/pdf/2403.00841
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/); [Embodied AI & robot learning](https://yingwen.io/en/research/embodied-ai/)

## Abstract

Offline Reinforcement Learning (RL) enables policy improvement from fixed datasets without online interactions, making it highly suitable for real-world applications lacking efficient simulators. Despite its success in the single-agent setting, offline multi-agent RL remains a challenge, especially in competitive games. Firstly, unaware of the game structure, it is impossible to interact with the opponents and conduct a major learning paradigm, self-play, for competitive games. Secondly, real-world datasets cannot cover all the state and action space in the game, resulting in barriers to identifying Nash equilibrium (NE). To address these issues, this paper introduces OFF-FSP, the first practical model-free offline RL algorithm for competitive games. We start by simulating interactions with various opponents by adjusting the weights of the fixed dataset with importance sampling. This technique allows us to learn the best responses to different opponents and employ the Offline Self-Play learning framework. To overcome the challenge of partial coverage, we combine the single-agent offline RL method with Fictitious Self-Play (FSP) to approximate NE by constraining the approximate best responses away from out-of-distribution actions. Experiments on matrix games, extensive-form poker, and board games demonstrate that OFF-FSP achieves significantly lower exploitability than state-of-the-art baselines. Finally, we validate OFF-FSP on a real-world human-robot competitive task, demonstrating its potential for solving complex, hard-to-simulate real-world problems.

## Citation

```bibtex
@inproceedings{chen2026-offline-fictitious-self-play-for-competitive-,
  title = {{Offline Fictitious Self-Play for Competitive Games}},
  author = {Jingxiao Chen and Weiji Xie and Weinan Zhang and Yong Yu and Ying Wen},
  year = {2026},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {40},
  number = {24},
  pages = {20118--20126},
  doi = {10.1609/aaai.v40i24.39098},
  url = {https://doi.org/10.1609/aaai.v40i24.39098}
}
```

BibTeX download: https://yingwen.io/citations/offline-fictitious-self-play-for-competitive-games.bib
