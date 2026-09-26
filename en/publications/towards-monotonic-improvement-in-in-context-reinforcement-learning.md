# Towards Monotonic Improvement in In-Context Reinforcement Learning

Authors: Wenhao Zhang; Shao Zhang; Xihuai Wang; Yang Li; Ying Wen

Publication: arXiv (2025)
Type: Preprint
Canonical page: https://yingwen.io/en/publications/towards-monotonic-improvement-in-in-context-reinforcement-learning/
Publisher / primary source: https://arxiv.org/abs/2509.23209
DOI: https://doi.org/10.48550/arXiv.2509.23209
arXiv: https://arxiv.org/abs/2509.23209
PDF: https://arxiv.org/pdf/2509.23209
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/)

## Abstract

In-Context Reinforcement Learning (ICRL) has emerged as a promising paradigm for developing agents that can rapidly adapt to new tasks by leveraging past experiences as context, without updating their parameters. Recent approaches train large sequence models on monotonic policy improvement data from online RL, aiming to a continue improved testing time performance. However, our experimental analysis reveals a critical flaw: these models cannot show a continue improvement like the training data during testing time. Theoretically, we identify this phenomenon as Contextual Ambiguity, where the model's own stochastic actions can generate an interaction history that misleadingly resembles that of a sub-optimal policy from the training data, initiating a vicious cycle of poor action selection. To resolve the Contextual Ambiguity, we introduce Context Value into training phase and propose Context Value Informed ICRL (CV-ICRL). CV-ICRL use Context Value as an explicit signal representing the ideal performance theoretically achievable by a policy given the current context. As the context expands, Context Value could include more task-relevant information, and therefore the ideal performance should be non-decreasing. We prove that the Context Value tightens the lower bound on the performance gap relative to an ideal, monotonically improving policy. We fruther propose two methods for estimating Context Value at both training and testing time. Experiments conducted on the Dark Room and Minigrid testbeds demonstrate that CV-ICRL effectively mitigates performance degradation and improves overall ICRL abilities across various tasks and environments. The source code and data of this paper are available at this https URL .

## Citation

```bibtex
@misc{zhang2025-towards-monotonic-improvement-in-in-context-r,
  title = {{Towards Monotonic Improvement in In-Context Reinforcement Learning}},
  author = {Wenhao Zhang and Shao Zhang and Xihuai Wang and Yang Li and Ying Wen},
  year = {2025},
  url = {https://arxiv.org/abs/2509.23209},
  eprint = {2509.23209},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2509.23209}
}
```

BibTeX download: https://yingwen.io/citations/towards-monotonic-improvement-in-in-context-reinforcement-learning.bib
