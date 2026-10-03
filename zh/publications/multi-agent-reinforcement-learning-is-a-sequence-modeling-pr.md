# Multi-Agent Reinforcement Learning is a Sequence Modeling Problem

Authors: Muning Wen; Jakub Kuba; Runji Lin; Weinan Zhang; Ying Wen; Jun Wang; Yaodong Yang

Publication: NeurIPS (2022)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/multi-agent-reinforcement-learning-is-a-sequence-modeling-pr/
Publisher / primary source: https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html
DOI: https://doi.org/10.52202/068431-1201
arXiv: https://arxiv.org/abs/2205.14953
PDF: https://proceedings.neurips.cc/paper_files/paper/2022/file/69413f87e5a34897cd010ca698097d0a-Paper-Conference.pdf
Code: https://github.com/PKU-MARL/Multi-Agent-Transformer
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

Large sequence models (SM) such as GPT series and BERT have displayed outstanding performance and generalization capabilities in natural language process, vision and recently reinforcement learning. A natural follow-up question is how to abstract multi-agent decision making also as an sequence modeling problem and benefit from the prosperous development of the SMs. In this paper, we introduce a novel architecture named Multi-Agent Transformer (MAT) that effectively casts cooperative multi-agent reinforcement learning (MARL) into SM problems wherein the objective is to map agents'  observation sequences to agents' optimal action sequences. Our goal is to build the bridge between MARL and SMs so that the modeling power of modern sequence models can be unleashed for MARL. Central to our MAT is an encoder-decoder architecture which leverages the multi-agent advantage decomposition theorem to transform the joint policy search problem into a sequential decision making process; this renders only linear time complexity for multi-agent problems and, most importantly, endows MAT with monotonic performance improvement guarantee. Unlike prior arts such as Decision Transformer fit only pre-collected offline data, MAT is trained by online trial and error from the environment in an on-policy fashion. To validate MAT, we conduct extensive experiments on StarCraftII, Multi-Agent MuJoCo, Dexterous Hands Manipulation, and Google Research Football benchmarks. Results demonstrate that MAT achieves superior performance and data efficiency compared to strong baselines including MAPPO and HAPPO. Furthermore, we demonstrate that MAT is an excellent few-short learner on unseen tasks regardless of changes in the number of agents.See our project page at https://sites.google.com/view/multi-agent-transformer.

## Citation

```bibtex
@inproceedings{NEURIPS2022_69413f87,
  title = {{Multi-Agent Reinforcement Learning is a Sequence Modeling Problem}},
  author = {Muning Wen and Jakub Kuba and Runji Lin and Weinan Zhang and Ying Wen and Jun Wang and Yaodong Yang},
  year = {2022},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {35},
  pages = {16509--16521},
  publisher = {Curran Associates, Inc.},
  doi = {10.52202/068431-1201},
  url = {https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html}
}
```

BibTeX download: https://yingwen.io/citations/multi-agent-reinforcement-learning-is-a-sequence-modeling-pr.bib
