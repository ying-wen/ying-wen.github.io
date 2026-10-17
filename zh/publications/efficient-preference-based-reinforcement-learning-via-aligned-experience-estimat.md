# Efficient Preference-based Reinforcement Learning via Aligned Experience Estimation

Authors: Fengshuo Bai; Rui Zhao; Hongming Zhang; Sijia Cui; Ying Wen; Yaodong Yang; Bo Xu; Lei Han

Publication: arXiv (2024)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/efficient-preference-based-reinforcement-learning-via-aligned-experience-estimat/
Publisher / primary source: https://arxiv.org/abs/2405.18688
DOI: https://doi.org/10.48550/arXiv.2405.18688
arXiv: https://arxiv.org/abs/2405.18688
PDF: https://arxiv.org/pdf/2405.18688
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Preference-based reinforcement learning (PbRL) has shown impressive capabilities in training agents without reward engineering. However, a notable limitation of PbRL is its dependency on substantial human feedback. This dependency stems from the learning loop, which entails accurate reward learning compounded with value/policy learning, necessitating a considerable number of samples. To boost the learning loop, we propose SEER, an efficient PbRL method that integrates label smoothing and policy regularization techniques. Label smoothing reduces overfitting of the reward model by smoothing human preference labels. Additionally, we bootstrap a conservative estimate $\widehat{Q}$ using well-supported state-action pairs from the current replay memory to mitigate overestimation bias and utilize it for policy learning regularization. Our experimental results across a variety of complex tasks, both in online and offline settings, demonstrate that our approach improves feedback efficiency, outperforming state-of-the-art methods by a large margin. Ablation studies further reveal that SEER achieves a more accurate Q-function compared to prior work.

## Citation

```bibtex
@misc{bai2024-efficient-preference-based-reinforcement-lear,
  title = {{Efficient Preference-based Reinforcement Learning via Aligned Experience Estimation}},
  author = {Fengshuo Bai and Rui Zhao and Hongming Zhang and Sijia Cui and Ying Wen and Yaodong Yang and Bo Xu and Lei Han},
  year = {2024},
  url = {https://arxiv.org/abs/2405.18688},
  eprint = {2405.18688},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2405.18688}
}
```

BibTeX download: https://yingwen.io/citations/efficient-preference-based-reinforcement-learning-via-aligned-experience-estimat.bib
