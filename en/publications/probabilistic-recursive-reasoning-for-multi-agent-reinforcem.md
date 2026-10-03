# Probabilistic Recursive Reasoning for Multi-Agent Reinforcement Learning

Authors: Ying Wen; Yaodong Yang; Rui Luo; Jun Wang; Wei Pan

Publication: ICLR (2019)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/probabilistic-recursive-reasoning-for-multi-agent-reinforcem/
Publisher / primary source: https://openreview.net/forum?id=rkl6As0cF7
arXiv: https://arxiv.org/abs/1901.09207
PDF: https://arxiv.org/pdf/1901.09207
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Humans are capable of attributing latent mental contents such as beliefs or intentions to others. The social skill is critical in daily life for reasoning about the potential consequences of others' behaviors so as to plan ahead. It is known that humans use such reasoning ability recursively by considering what others believe about their own beliefs. In this paper, we start from level-$1$ recursion and introduce a probabilistic recursive reasoning (PR2) framework for multi-agent reinforcement learning. Our hypothesis is that it is beneficial for each agent to account for how the opponents would react to its future behaviors. Under the PR2 framework, we adopt variational Bayes methods to approximate the opponents' conditional policies, to which each agent finds the best response and then improve their own policies. We develop decentralized-training-decentralized-execution algorithms, namely PR2-Q and PR2-Actor-Critic, that are proved to converge in the self-play scenarios when there exists one Nash equilibrium. Our methods are tested on both the matrix game and the differential game, which have a non-trivial equilibrium where common gradient-based methods fail to converge. Our experiments show that it is critical to reason about how the opponents believe about what the agent believes. We expect our work to contribute a new idea of modeling the opponents to the multi-agent reinforcement learning community.

## Citation

```bibtex
@inproceedings{wen2019-probabilistic-recursive-reasoning-for-multi-a,
  title = {{Probabilistic Recursive Reasoning for Multi-Agent Reinforcement Learning}},
  author = {Ying Wen and Yaodong Yang and Rui Luo and Jun Wang and Wei Pan},
  year = {2019},
  booktitle = {International Conference on Learning Representations},
  url = {https://openreview.net/forum?id=rkl6As0cF7}
}
```

BibTeX download: https://yingwen.io/citations/probabilistic-recursive-reasoning-for-multi-agent-reinforcem.bib
