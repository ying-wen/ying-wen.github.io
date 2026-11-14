# Multi-Agent Determinantal Q-Learning

Authors: Yaodong Yang; Ying Wen; Jun Wang; Liheng Chen; Kun Shao; David Mguni; Weinan Zhang

Publication: ICML (2020)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/multi-agent-determinantal-q-learning/
Publisher / primary source: https://proceedings.mlr.press/v119/yang20i.html
arXiv: https://arxiv.org/abs/2006.01482
PDF: http://proceedings.mlr.press/v119/yang20i/yang20i.pdf
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Centralized training with decentralized execution has become an important paradigm in multi-agent learning. Though practical, current methods rely on restrictive assumptions to decompose the centralized value function across agents for execution. In this paper, we eliminate this restriction by proposing multi-agent determinantal Q-learning. Our method is established on Q-DPP, a novel extension of determinantal point process (DPP) to multi-agent setting. Q-DPP promotes agents to acquire diverse behavioral models; this allows a natural factorization of the joint Q-functions with no need for \emph{a priori} structural constraints on the value function or special network architectures. We demonstrate that Q-DPP generalizes major solutions including VDN, QMIX, and QTRAN on decentralizable cooperative tasks. To efficiently draw samples from Q-DPP, we develop a linear-time sampler with theoretical approximation guarantee. Our sampler also benefits exploration by coordinating agents to cover orthogonal directions in the state space during training. We evaluate our algorithm on multiple cooperative benchmarks; its effectiveness has been demonstrated when compared with the state-of-the-art.

## Citation

```bibtex
@inproceedings{pmlr-v119-yang20i,
  title = {{Multi-Agent Determinantal Q-Learning}},
  author = {Yaodong Yang and Ying Wen and Jun Wang and Liheng Chen and Kun Shao and David Mguni and Weinan Zhang},
  year = {2020},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  volume = {119},
  pages = {10757--10766},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v119/yang20i.html}
}
```

BibTeX download: https://yingwen.io/citations/multi-agent-determinantal-q-learning.bib
