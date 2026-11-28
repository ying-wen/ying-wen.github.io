# MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety

Authors: Xiaoyu Wen; Zhida He; Han Qi; Ziyu Wan; Zhongtian Ma; Ying Wen; Tianhang Zheng; Xingcheng Xu; Chaochao Lu; Qiaosheng Zhang

Publication: arXiv (2026)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/magic-a-co-evolving-attacker-defender-adversarial-game-for-robust-llm-safety/
Publisher / primary source: https://arxiv.org/abs/2602.01539
DOI: https://doi.org/10.48550/arXiv.2602.01539
arXiv: https://arxiv.org/abs/2602.01539
PDF: https://arxiv.org/pdf/2602.01539
Topics: [大模型与推理](https://yingwen.io/zh/research/llms/); [大模型智能体](https://yingwen.io/zh/research/llm-agents/); [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

Ensuring robust safety alignment is crucial for Large Language Models (LLMs), yet existing defenses often lag behind evolving adversarial attacks due to their \textbf{reliance on static, pre-collected data distributions}. In this paper, we introduce \textbf{MAGIC}, a novel multi-turn multi-agent reinforcement learning framework that formulates LLM safety alignment as an adversarial asymmetric game. Specifically, an attacker agent learns to iteratively rewrite original queries into deceptive prompts, while a defender agent simultaneously optimizes its policy to recognize and refuse such inputs. This dynamic process triggers a \textbf{co-evolution}, where the attacker's ever-changing strategies continuously uncover long-tail vulnerabilities, driving the defender to generalize to unseen attack patterns. Remarkably, we observe that the attacker, endowed with initial reasoning ability, evolves \textbf{novel, previously unseen combinatorial strategies} through iterative RL training, underscoring our method's substantial potential. Theoretically, we provide insights into a more robust game equilibrium and derive safety guarantees. Extensive experiments validate our framework's effectiveness, demonstrating superior defense success rates without compromising the helpfulness of the model. Our code is available at this https URL .

## Citation

```bibtex
@misc{wen2026-magic-a-co-evolving-attacker-defender-adversa,
  title = {{MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety}},
  author = {Xiaoyu Wen and Zhida He and Han Qi and Ziyu Wan and Zhongtian Ma and Ying Wen and Tianhang Zheng and Xingcheng Xu and Chaochao Lu and Qiaosheng Zhang},
  year = {2026},
  url = {https://arxiv.org/abs/2602.01539},
  eprint = {2602.01539},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2602.01539}
}
```

BibTeX download: https://yingwen.io/citations/magic-a-co-evolving-attacker-defender-adversarial-game-for-robust-llm-safety.bib
