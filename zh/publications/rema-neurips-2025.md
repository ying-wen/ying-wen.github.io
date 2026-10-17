# ReMA: Learning to Meta-Think for LLMs with Multi-agent Reinforcement Learning

Authors: Ziyu Wan; Yunxiang Li; Xiaoyu Wen; Yan Song; Hanjing Wang; Linyi Yang; Mark Schmidt; Jun Wang; Weinan Zhang; Shuyue Hu; Ying Wen

Publication: NeurIPS (2025)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/rema-neurips-2025/
Publisher / primary source: https://proceedings.neurips.cc/paper_files/paper/2025/hash/b83514e453ff03741a5fb3e5e49f709b-Abstract-Conference.html
DOI: https://doi.org/10.52202/085713-4225
arXiv: https://arxiv.org/abs/2503.09501
PDF: https://proceedings.neurips.cc/paper_files/paper/2025/file/b83514e453ff03741a5fb3e5e49f709b-Paper-Conference.pdf
Topics: [大模型与推理](https://yingwen.io/zh/research/llms/); [大模型智能体](https://yingwen.io/zh/research/llm-agents/); [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

Recent research on Reasoning of Large Language Models (LLMs) has sought to further enhance their performance by integrating meta-thinking—enabling models to monitor, evaluate, and control their reasoning processes for more adaptive and effective problem-solving. However, current single-agent work lacks a specialized design for acquiring meta-thinking, resulting in low efficacy. To address this challenge, we introduce Reinforced Meta-thinking Agents (ReMA), a novel framework that leverages Multi-Agent Reinforcement Learning (MARL) to elicit meta-thinking behaviors, encouraging LLMs to think about thinking. ReMA decouples the reasoning process into two hierarchical agents: a high-level meta-thinking agent responsible for generating strategic oversight and plans, and a low-level reasoning agent for detailed executions. Through iterative reinforcement learning with aligned objectives, these agents explore and learn collaboration, leading to improved generalization and robustness. Empirical results from single-turn experiments demonstrate that ReMA outperforms single-agent RL baselines on complex reasoning tasks, including competitive-level mathematical benchmarks and LLM-as-a-Judge benchmarks. Additionally, we further extend ReMA to multi-turn interaction settings, leveraging turn-level ratio and parameter sharing to improve efficiency. Comprehensive ablation studies further illustrate the evolving dynamics of each distinct agent, providing valuable insights into how the meta-thinking reasoning process enhances the reasoning capabilities of LLMs.

## Citation

```bibtex
@inproceedings{NEURIPS2025_b83514e4,
  title = {{ReMA: Learning to Meta-Think for LLMs with Multi-agent Reinforcement Learning}},
  author = {Ziyu Wan and Yunxiang Li and Xiaoyu Wen and Yan Song and Hanjing Wang and Linyi Yang and Mark Schmidt and Jun Wang and Weinan Zhang and Shuyue Hu and Ying Wen},
  year = {2025},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {38, Main Conference},
  pages = {126621--126667},
  publisher = {Curran Associates, Inc.},
  doi = {10.52202/085713-4225},
  url = {https://proceedings.neurips.cc/paper_files/paper/2025/hash/b83514e453ff03741a5fb3e5e49f709b-Abstract-Conference.html}
}
```

BibTeX download: https://yingwen.io/citations/rema-neurips-2025.bib
