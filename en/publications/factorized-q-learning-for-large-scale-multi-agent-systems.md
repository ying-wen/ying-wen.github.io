# Factorized Q-learning for large-scale multi-agent systems

Authors: Ming Zhou; Yong Chen; Ying Wen; Yaodong Yang; Yufeng Su; Weinan Zhang; Dell Zhang; Jun Wang

Publication: DAI (2019)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/factorized-q-learning-for-large-scale-multi-agent-systems/
Publisher / primary source: https://doi.org/10.1145/3356464.3357707
DOI: https://doi.org/10.1145/3356464.3357707
arXiv: https://arxiv.org/abs/1809.03738
PDF: https://arxiv.org/pdf/1809.03738
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Deep Q-learning has achieved significant success in single-agent decision making tasks. However, it is challenging to extend Q-learning to large-scale multi-agent scenarios, due to the explosion of action space resulting from the complex dynamics between the environment and the agents. In this paper, we propose to make the computation of multi-agent Q-learning tractable by treating the Q-function (w.r.t. state and joint-action) as a high-order high-dimensional tensor and then approximate it with factorized pairwise interactions. Furthermore, we utilize a composite deep neural network architecture for computing the factorized Q-function, share the model parameters among all the agents within the same group, and estimate the agents' optimal joint actions through a coordinate descent type algorithm. All these simplifications greatly reduce the model complexity and accelerate the learning process. Extensive experiments on two different multi-agent problems demonstrate the performance gain of our proposed approach in comparison with strong baselines, particularly when there are a large number of agents.

## Citation

```bibtex
@inproceedings{zhou2019-factorized-q-learning-for-large-scale-multi-a,
  title = {{Factorized Q-learning for large-scale multi-agent systems}},
  author = {Ming Zhou and Yong Chen and Ying Wen and Yaodong Yang and Yufeng Su and Weinan Zhang and Dell Zhang and Jun Wang},
  year = {2019},
  booktitle = {Proceedings of the First International Conference on Distributed Artificial Intelligence},
  pages = {1--7},
  publisher = {ACM},
  doi = {10.1145/3356464.3357707},
  url = {https://doi.org/10.1145/3356464.3357707}
}
```

BibTeX download: https://yingwen.io/citations/factorized-q-learning-for-large-scale-multi-agent-systems.bib
