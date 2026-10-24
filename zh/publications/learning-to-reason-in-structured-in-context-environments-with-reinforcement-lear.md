# Structured In-context Environment Scaling for Large Language Model Reasoning

Authors: Peng Yu; Zeyuan Zhao; Shao Zhang; Luoyi Fu; Xinbing Wang; Ying Wen

Publication: ICLR 2026 (2026)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/learning-to-reason-in-structured-in-context-environments-with-reinforcement-lear/
Publisher / primary source: https://openreview.net/forum?id=CicK2lJMUy
arXiv: https://arxiv.org/abs/2509.23330
PDF: https://openreview.net/pdf?id=CicK2lJMUy
Topics: [大模型与推理](https://yingwen.io/zh/research/llms/); [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Large language models (LLMs) have achieved significant advancements in reasoning capabilities through reinforcement learning (RL) via environmental exploration. As the intrinsic properties of the environment determine the abilities that LLMs can learn, the environment plays a important role in the RL finetuning process. An ideal LLM reasoning environment should possess three core characteristics: scalability, generalizable reasoning, and verifiability. However, existing mathematical and coding environments are difficult to scale due to heavy reliance on expert annotation, while the skills learned in game-based environments are too specialized to generalize. To bridge this gap, we introduce the \textbf{S}tructured \textbf{I}n-context \textbf{E}nvironment (SIE) framework. SIE achieves scalability by automatically constructing reasoning environments from large-scale structured data, where the rich compositional patterns naturally support generalizable reasoning. Moreover, the explicit schemas and reasoning chains in structured data provide a foundation for rule-based verifiability. Experimental results show that SIE framework not only achieves substantial improvements in in-domain structured reasoning, but also enables the learned compositional reasoning skills to generalize effectively to out-of-domain mathematical and logical reasoning tasks. We further explored learning in information-limited partial SIEs and found that LLMs can infer the missing information through exploring the environment, leading to robust reasoning improvements and generalization performance.

## Citation

```bibtex
@inproceedings{yu2026-learning-to-reason-in-structured-in-context-e,
  title = {{Structured In-context Environment Scaling for Large Language Model Reasoning}},
  author = {Peng Yu and Zeyuan Zhao and Shao Zhang and Luoyi Fu and Xinbing Wang and Ying Wen},
  year = {2026},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  url = {https://openreview.net/forum?id=CicK2lJMUy}
}
```

BibTeX download: https://yingwen.io/citations/learning-to-reason-in-structured-in-context-environments-with-reinforcement-lear.bib
