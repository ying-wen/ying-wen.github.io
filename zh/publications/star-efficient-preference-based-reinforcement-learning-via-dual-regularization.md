# STAR: Efficient Preference-based Reinforcement Learning via Dual Regularization

Authors: Fengshuo Bai; Rui Zhao; Hongming Zhang; Sijia Cui; Shao Zhang; Bo Xu; Lei Han; Ying Wen; Yaodong Yang

Publication: NeurIPS (2025)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/star-efficient-preference-based-reinforcement-learning-via-dual-regularization/
Publisher / primary source: https://proceedings.neurips.cc/paper_files/paper/2025/hash/f5e40176a0a905b9fcba6b21d840cb1e-Abstract-Conference.html
DOI: https://doi.org/10.52202/085713-5609
PDF: https://proceedings.neurips.cc/paper_files/paper/2025/file/f5e40176a0a905b9fcba6b21d840cb1e-Paper-Conference.pdf
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Preference-based reinforcement learning (PbRL) bypasses complex reward engineering by learning from human feedback. However, due to the high cost of obtaining feedback, PbRL typically relies on a limited set of preference-labeled samples. This data scarcity introduces two key inefficiencies: (1) the reward model overfits to the limited feedback, leading to poor generalization to unseen samples, and (2) the agent exploits the learned reward model, exacerbating overestimation of action values in temporal difference (TD) learning. To address these issues, we propose STAR, an efficient PbRL method that integrates preference margin regularization and policy regularization. Preference margin regularization mitigates overfitting by introducing a bounded margin in reward optimization, preventing excessive bias toward specific feedback. Policy regularization bootstraps a conservative estimate $\widehat{Q}$ from well-supported state-action pairs in the replay memory, reducing overestimation during policy learning. Experimental results show that STAR improves feedback efficiency, achieving 34.8\% higher performance in online settings and 29.7\% in offline settings compared to state-of-the-art methods. Ablation studies confirm that STAR facilitates more robust reward and value function learning. The videos of this project are released at https://sites.google.com/view/pbrl-star.

## Citation

```bibtex
@inproceedings{NEURIPS2025_f5e40176,
  title = {{STAR: Efficient Preference-based Reinforcement Learning via Dual Regularization}},
  author = {Fengshuo Bai and Rui Zhao and Hongming Zhang and Sijia Cui and Shao Zhang and Bo Xu and Lei Han and Ying Wen and Yaodong Yang},
  year = {2025},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {38, Main Conference},
  pages = {168265--168300},
  publisher = {Curran Associates, Inc.},
  doi = {10.52202/085713-5609},
  url = {https://proceedings.neurips.cc/paper_files/paper/2025/hash/f5e40176a0a905b9fcba6b21d840cb1e-Abstract-Conference.html}
}
```

BibTeX download: https://yingwen.io/citations/star-efficient-preference-based-reinforcement-learning-via-dual-regularization.bib
