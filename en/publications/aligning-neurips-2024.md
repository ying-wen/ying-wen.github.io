# Aligning Individual and Collective Objectives in Multi-Agent Cooperation

Authors: Yang Li; Wenhao Zhang; Jianhong Wang; Shao Zhang; Yali Du; Ying Wen; Wei Pan

Publication: NeurIPS (2024)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/aligning-neurips-2024/
Publisher / primary source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ee3ac2cd119023c79b0d21c4a464dc7-Abstract-Conference.html
DOI: https://doi.org/10.52202/079017-1421
arXiv: https://arxiv.org/abs/2402.12416
PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/4ee3ac2cd119023c79b0d21c4a464dc7-Paper-Conference.pdf
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Among the research topics in multi-agent learning, mixed-motive cooperation is one of the most prominent challenges, primarily due to the mismatch between individual and collective goals. The cutting-edge research is focused on incorporating domain knowledge into rewards and introducing additional mechanisms to incentivize cooperation. However, these approaches often face shortcomings such as the effort on manual design and the absence of theoretical groundings. To close this gap, we model the mixed-motive game as a differentiable game for the ease of illuminating the learning dynamics towards cooperation. More detailed, we introduce a novel optimization method named \textbf{\textit{A}}ltruistic \textbf{\textit{G}}radient \textbf{\textit{A}}djustment (\textbf{\textit{AgA}}) that employs gradient adjustments to progressively align individual and collective objectives. Furthermore, we theoretically prove that AgA effectively attracts gradients to stable fixed points of the collective objective while considering individual interests, and we validate these claims with empirical evidence. We evaluate the effectiveness of our algorithm AgA through benchmark environments for testing mixed-motive collaboration with small-scale agents such as the two-player public good game and the sequential social dilemma games, Cleanup and Harvest, as well as our self-developed large-scale environment in the game StarCraft II.

## Citation

```bibtex
@inproceedings{NEURIPS2024_4ee3ac2c,
  title = {{Aligning Individual and Collective Objectives in Multi-Agent Cooperation}},
  author = {Yang Li and Wenhao Zhang and Jianhong Wang and Shao Zhang and Yali Du and Ying Wen and Wei Pan},
  year = {2024},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  pages = {44735--44760},
  publisher = {Curran Associates, Inc.},
  doi = {10.52202/079017-1421},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ee3ac2cd119023c79b0d21c4a464dc7-Abstract-Conference.html}
}
```

BibTeX download: https://yingwen.io/citations/aligning-neurips-2024.bib
