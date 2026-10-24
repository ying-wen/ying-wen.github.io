# Towards Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games

Authors: Xiangyu Liu; Hangtian Jia; Ying Wen; Yujing Hu; Yingfeng Chen; Changjie Fan; Zhipeng Hu; Yaodong Yang

Publication: NeurIPS (2021)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/towards-unifying-behavioral-and-response-diversity-for-open/
Publisher / primary source: https://proceedings.neurips.cc/paper/2021/hash/07bba581a2dd8d098a3be0f683560643-Abstract.html
arXiv: https://arxiv.org/abs/2106.04958
PDF: https://proceedings.neurips.cc/paper_files/paper/2021/file/07bba581a2dd8d098a3be0f683560643-Paper.pdf
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

Measuring and promoting policy diversity is critical for solving games with strong non-transitive dynamics where strategic cycles exist, and there is no consistent winner (e.g., Rock-Paper-Scissors). With that in mind, maintaining a pool of diverse policies via open-ended learning is an attractive solution, which can generate auto-curricula to avoid being exploited. However, in conventional open-ended learning algorithms, there are no widely accepted definitions for diversity, making it hard to construct and evaluate the diverse policies. In this work, we summarize previous concepts of diversity and work towards offering a unified measure of diversity in multi-agent open-ended learning to include all elements in Markov games, based on both Behavioral Diversity (BD) and Response Diversity (RD). At the trajectory distribution level, we re-define BD in the state-action space as the discrepancies of occupancy measures. For the reward dynamics, we propose RD to characterize diversity through the responses of policies when encountering different opponents. We also show that many current diversity measures fall in one of the categories of BD or RD but not both. With this unified diversity measure, we design the corresponding diversity-promoting objective and population effectivity when seeking the best responses in open-ended learning. We validate our methods in both relatively simple games like matrix game, non-transitive mixture model, and the complex \textit{Google Research Football} environment. The population found by our methods reveals the lowest exploitability, highest population effectivity in matrix game and non-transitive mixture model, as well as the largest goal difference when interacting with opponents of various levels in \textit{Google Research Football}.

## Citation

```bibtex
@inproceedings{NEURIPS2021_07bba581,
  title = {{Towards Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games}},
  author = {Xiangyu Liu and Hangtian Jia and Ying Wen and Yujing Hu and Yingfeng Chen and Changjie Fan and Zhipeng Hu and Yaodong Yang},
  year = {2021},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
  pages = {941--952},
  publisher = {Curran Associates, Inc.},
  url = {https://proceedings.neurips.cc/paper/2021/hash/07bba581a2dd8d098a3be0f683560643-Abstract.html}
}
```

BibTeX download: https://yingwen.io/citations/towards-unifying-behavioral-and-response-diversity-for-open.bib
