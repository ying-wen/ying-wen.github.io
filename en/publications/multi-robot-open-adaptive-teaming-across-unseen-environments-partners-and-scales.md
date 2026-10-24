# Multi-Robot Open Adaptive Teaming Across Unseen Environments, Partners, and Scales

Authors: Yang Li; Feng Xue; Fan Mo; Yunhao Liu; Jianhong Wang; Ying Wen; Qingrui Zhang; Shaoshuai Mou; Wei Pan

Publication: arXiv (2026)
Type: Preprint
Canonical page: https://yingwen.io/en/publications/multi-robot-open-adaptive-teaming-across-unseen-environments-partners-and-scales/
Publisher / primary source: https://arxiv.org/abs/2607.04972
DOI: https://doi.org/10.48550/arXiv.2607.04972
arXiv: https://arxiv.org/abs/2607.04972
PDF: https://arxiv.org/pdf/2607.04972
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/); [Embodied AI & robot learning](https://yingwen.io/en/research/embodied-ai/)

## Abstract

Deploying robot teams in the real world requires simultaneous adaptation to unseen environments, unknown partners, and varying team sizes, yet existing approaches often address these challenges in isolation under the closed-world assumption of fixed teammates. We formalize this as open adaptive multi-robot teaming and propose a hypergraphic-form game formulation that captures team-level cooperative relationships beyond pairwise interactions, providing a principled foundation for coordination structure inference when team composition changes dynamically within episodes. Unlike graph neural network architectures, this is a game-theoretic construct for modeling strategic interactions and payoff structures among agents. Building on this formulation, we develop the Hypergraphic Open-ended Learning Algorithm (HOLA), which progressively expands partner and environment diversity during training rather than optimizing for fixed configurations. Evaluated on cooperative pursuit with multi-drone and multi-quadruped platforms, HOLA outperforms all baselines across all three adaptability dimensions. Learned policies transfer directly to physical hardware without fine-tuning, with successful deployments on Crazyflie and Zsibot L1 platforms confirming robust real-world coordination in novel environments with unseen teammates.

## Citation

```bibtex
@misc{li2026-multi-robot-open-adaptive-teaming-across-unse,
  title = {{Multi-Robot Open Adaptive Teaming Across Unseen Environments, Partners, and Scales}},
  author = {Yang Li and Feng Xue and Fan Mo and Yunhao Liu and Jianhong Wang and Ying Wen and Qingrui Zhang and Shaoshuai Mou and Wei Pan},
  year = {2026},
  url = {https://arxiv.org/abs/2607.04972},
  eprint = {2607.04972},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2607.04972}
}
```

BibTeX download: https://yingwen.io/citations/multi-robot-open-adaptive-teaming-across-unseen-environments-partners-and-scales.bib
