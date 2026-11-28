# ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination

Authors: Xihuai Wang; Shao Zhang; Wenhao Zhang; Wentao Dong; Jingxiao Chen; Ying Wen; Weinan Zhang

Publication: NeurIPS (2024)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/zsc-eval-neurips-2024/
Publisher / primary source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/54a7139c548c88e288aa0fcd2bcbeceb-Abstract-Datasets_and_Benchmarks_Track.html
DOI: https://doi.org/10.52202/079017-1501
arXiv: https://arxiv.org/abs/2310.05208
PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/54a7139c548c88e288aa0fcd2bcbeceb-Paper-Datasets_and_Benchmarks_Track.pdf
Code: https://github.com/sjtu-marl/ZSC-Eval
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

Zero-shot coordination (ZSC) is a new cooperative multi-agent reinforcement learning (MARL) challenge that aims to train an ego agent to work with diverse, unseen partners during deployment. The significant difference between the deployment-time partners' distribution and the training partners' distribution determined by the training algorithm makes ZSC a unique out-of-distribution (OOD) generalization challenge. The potential distribution gap between evaluation and deployment-time partners leads to inadequate evaluation, which is exacerbated by the lack of appropriate evaluation metrics. In this paper, we present ZSC-Eval , the first evaluation toolkit and benchmark for ZSC algorithms.  ZSC-Eval consists of: 1) Generation of evaluation partner candidates through behavior-preferring rewards to approximate deployment-time partners' distribution; 2) Selection of evaluation partners by Best-Response Diversity (BR-Div); 3) Measurement of generalization performance with various evaluation partners via the Best-Response Proximity (BR-Prox) metric. We use ZSC-Eval to benchmark ZSC algorithms in Overcooked and Google Research Football environments and get novel empirical findings.  We also conduct a human experiment of current ZSC algorithms to verify the ZSC-Eval's consistency with human evaluation. ZSC-Eval is now available at https://github.com/sjtu-marl/ZSC-Eval.

## Citation

```bibtex
@inproceedings{NEURIPS2024_54a7139c,
  title = {{ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination}},
  author = {Xihuai Wang and Shao Zhang and Wenhao Zhang and Wentao Dong and Jingxiao Chen and Ying Wen and Weinan Zhang},
  year = {2024},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  pages = {47344--47377},
  publisher = {Curran Associates, Inc.},
  doi = {10.52202/079017-1501},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/54a7139c548c88e288aa0fcd2bcbeceb-Abstract-Datasets_and_Benchmarks_Track.html}
}
```

BibTeX download: https://yingwen.io/citations/zsc-eval-neurips-2024.bib
