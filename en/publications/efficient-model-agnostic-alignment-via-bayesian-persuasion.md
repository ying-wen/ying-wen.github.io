# Efficient Model-agnostic Alignment via Bayesian Persuasion

Authors: Fengshuo Bai; Mingzhi Wang; Zhaowei Zhang; Boyuan Chen; Yinda Xu; Ying Wen; Yaodong Yang

Publication: arXiv (2024)
Type: Preprint
Canonical page: https://yingwen.io/en/publications/efficient-model-agnostic-alignment-via-bayesian-persuasion/
Publisher / primary source: https://arxiv.org/abs/2405.18718
DOI: https://doi.org/10.48550/arXiv.2405.18718
arXiv: https://arxiv.org/abs/2405.18718
PDF: https://arxiv.org/pdf/2405.18718
Topics: [Large language models & reasoning](https://yingwen.io/en/research/llms/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

With recent advancements in large language models (LLMs), alignment has emerged as an effective technique for keeping LLMs consensus with human intent. Current methods primarily involve direct training through Supervised Fine-tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF), both of which require substantial computational resources and extensive ground truth data. This paper explores an efficient method for aligning black-box large models using smaller models, introducing a model-agnostic and lightweight Bayesian Persuasion Alignment framework. We formalize this problem as an optimization of the signaling strategy from the small model's perspective. In the persuasion process, the small model (Advisor) observes the information item (i.e., state) and persuades large models (Receiver) to elicit improved responses. The Receiver then generates a response based on the input, the signal from the Advisor, and its updated belief about the information item. Through training using our framework, we demonstrate that the Advisor can significantly enhance the performance of various Receivers across a range of tasks. We theoretically analyze our persuasion framework and provide an upper bound on the Advisor's regret, confirming its effectiveness in learning the optimal signaling strategy. Our Empirical results demonstrates that GPT-2 can significantly improve the performance of various models, achieving an average enhancement of 16.1% in mathematical reasoning ability and 13.7% in code generation. We hope our work can provide an initial step toward rethinking the alignment framework from the Bayesian Persuasion perspective.

## Citation

```bibtex
@misc{bai2024-efficient-model-agnostic-alignment-via-bayesi,
  title = {{Efficient Model-agnostic Alignment via Bayesian Persuasion}},
  author = {Fengshuo Bai and Mingzhi Wang and Zhaowei Zhang and Boyuan Chen and Yinda Xu and Ying Wen and Yaodong Yang},
  year = {2024},
  url = {https://arxiv.org/abs/2405.18718},
  eprint = {2405.18718},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2405.18718}
}
```

BibTeX download: https://yingwen.io/citations/efficient-model-agnostic-alignment-via-bayesian-persuasion.bib
