# AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training

Authors: Ziyu Wan; Xidong Feng; Muning Wen; Stephen Marcus Mcaleer; Ying Wen; Weinan Zhang; Jun Wang

Publication: ICML (2024)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/ts-llm-icml-2024/
Publisher / primary source: https://proceedings.mlr.press/v235/wan24c.html
arXiv: https://arxiv.org/abs/2309.17179
PDF: https://raw.githubusercontent.com/mlresearch/v235/main/assets/wan24c/wan24c.pdf
Code: https://github.com/waterhorse1/LLM_Tree_Search
Topics: [大模型与推理](https://yingwen.io/zh/research/llms/); [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Recent works like Tree-of-Thought (ToT) and Reasoning via Planning (RAP) aim to augment the multi-step reasoning capabilities of LLMs by using tree-search algorithms. These methods rely on prompting a pre-trained model to serve as a value function and focus on problems with low search depth. As a result, these methods cannot benefit from in-domain training and only rely on pretraining process — they will not work in domains where the pre-trained LLM does not have enough knowledge to serve as an effective value function or in domains that require long-horizon planning. To address these limitations, we present an AlphaZero-like tree-search learning framework for LLMs (termed TS-LLM), systematically illustrating how tree-search with a learned value function can guide LLM decoding. TS-LLM distinguishes itself in two key ways. (1) Leveraging a learned value function and AlphaZero-like algorithms, our approach can be generally adaptable to a wide range of tasks, language models of any size, and tasks of varying search depths. (2) Our approach can guide LLMs during both inference and training, iteratively improving the LLMs. Empirical results across reasoning, planning, alignment, and decision-making tasks show that TS-LLM outperforms existing approaches and can handle trees with a depth of 64.

## Citation

```bibtex
@inproceedings{pmlr-v235-wan24c,
  title = {{AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training}},
  author = {Ziyu Wan and Xidong Feng and Muning Wen and Stephen Marcus Mcaleer and Ying Wen and Weinan Zhang and Jun Wang},
  year = {2024},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  volume = {235},
  pages = {49890--49920},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v235/wan24c.html}
}
```

BibTeX download: https://yingwen.io/citations/ts-llm-icml-2024.bib
