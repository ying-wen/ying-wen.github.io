# GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models

Authors: Hanjing Wang; Man-Kit Sit; Congjie He; Ying Wen; Weinan Zhang; Jun Wang; Yaodong Yang; Luo Mai

Publication: ICML (2023)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/gear-a-gpu-centric-experience-replay-system-for-large-reinfo/
Publisher / primary source: https://proceedings.mlr.press/v202/wang23aj.html
arXiv: https://arxiv.org/abs/2310.05205
PDF: https://proceedings.mlr.press/v202/wang23aj/wang23aj.pdf
Code: https://github.com/bigrl-team/gear
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/)

## Abstract

This paper introduces a distributed, GPU-centric experience replay system, GEAR, designed to perform scalable reinforcement learning (RL) with large sequence models (such as transformers). With such models, existing systems such as Reverb face considerable bottlenecks in memory, computation, and communication. GEAR, however, optimizes memory efficiency by enabling the memory resources on GPU servers (including host memory and device memory) to manage trajectory data. Furthermore, it facilitates decentralized GPU devices to expedite various trajectory selection strategies, circumventing computational bottlenecks. GEAR is equipped with GPU kernels capable of collecting trajectories using zero-copy access to host memory, along with remote-directed-memory access over InfiniBand, improving communication efficiency. Cluster experiments have shown that GEAR can achieve performance levels up to 6× greater than Reverb when training state-of-the-art large RL models. GEAR is open-sourced at https:// github.com/bigrl-team/gear.

## Citation

```bibtex
@inproceedings{pmlr-v202-wang23aj,
  title = {{GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models}},
  author = {Hanjing Wang and Man-Kit Sit and Congjie He and Ying Wen and Weinan Zhang and Jun Wang and Yaodong Yang and Luo Mai},
  year = {2023},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  volume = {202},
  pages = {36380--36390},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v202/wang23aj.html}
}
```

BibTeX download: https://yingwen.io/citations/gear-a-gpu-centric-experience-replay-system-for-large-reinfo.bib
