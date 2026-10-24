# Pairwise multi-layer nets for learning distributed representation of multi-field categorical data

Authors: Ying Wen; Tianyao Chen; Jun Wang; Weinan Zhang

Publication: DLP-KDD 2019 (2019)
Type: Workshop paper
Canonical page: https://yingwen.io/zh/publications/pairwise-multi-layer-nets-for-learning-distributed-representation-of-multi-field-categorical-data/
Publisher / primary source: https://doi.org/10.1145/3326937.3341251
DOI: https://doi.org/10.1145/3326937.3341251
PDF: https://dlp-kdd.github.io/dlp-kdd2019/assets/pdf/a2-wen.pdf
Topics: [机器学习与应用](https://yingwen.io/zh/research/machine-learning/)

## Abstract

This paper presents a method of pairwise multi-layer networks for multi-field categorical data, which widely exists with various applications such as web search, recommender systems, social link prediction, and computational advertising. The success of non-linear models, e.g., factorization machines, boosted trees, has proved the potential of exploring the interactions among inter-field discrete categories. Inspired by Word2Vec, the distributed representation for natural language, we propose a PMLN (Pairwise Multi-Layer Nets) model to learn the distributed representation for multi-field categorical data. In PMLN, a low-dimensional continuous vector is automatically learned for each category in each field. The interactions among inter-field categories are explored by different neural gates and the most informative ones are selected by pooling layers. Such combined categories can be further explored by performing more gate interactions with another category and then selected by additional pooling operations. In our experiments, with the exploration of the interactions between pairwise categories over layers, the model outperforms state-of-the-art models in a supervised learning task, i.e., ad click prediction, while capturing the most significant interactions from the data in an unsupervised fashion.

## Citation

```bibtex
@inproceedings{wen2019-pairwise-multi-layer-nets-for-learning-distri,
  title = {{Pairwise multi-layer nets for learning distributed representation of multi-field categorical data}},
  author = {Ying Wen and Tianyao Chen and Jun Wang and Weinan Zhang},
  year = {2019},
  booktitle = {Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data},
  pages = {1--8},
  publisher = {ACM},
  doi = {10.1145/3326937.3341251},
  url = {https://doi.org/10.1145/3326937.3341251},
  note = {Workshop paper}
}
```

BibTeX download: https://yingwen.io/citations/pairwise-multi-layer-nets-for-learning-distributed-representation-of-multi-field-categorical-data.bib
