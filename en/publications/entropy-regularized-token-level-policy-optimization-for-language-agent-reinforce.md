# Entropy-Regularized Token-Level Policy Optimization for Language Agent Reinforcement

Authors: Muning Wen; Junwei Liao; Cheng Deng; Jun Wang; Weinan Zhang; Ying Wen

Publication: arXiv (2024)
Type: Preprint
Canonical page: https://yingwen.io/en/publications/entropy-regularized-token-level-policy-optimization-for-language-agent-reinforce/
Publisher / primary source: https://arxiv.org/abs/2402.06700
DOI: https://doi.org/10.48550/arXiv.2402.06700
arXiv: https://arxiv.org/abs/2402.06700
PDF: https://arxiv.org/pdf/2402.06700
Topics: [LLM agents](https://yingwen.io/en/research/llm-agents/); [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/)

## Abstract

Large Language Models (LLMs) have shown promise as intelligent agents in interactive decision-making tasks. Traditional approaches often depend on meticulously designed prompts, high-quality examples, or additional reward models for in-context learning, supervised fine-tuning, or RLHF. Reinforcement learning (RL) presents a dynamic alternative for LLMs to overcome these dependencies by engaging directly with task-specific environments. Nonetheless, it faces significant hurdles: 1) instability stemming from the exponentially vast action space requiring exploration; 2) challenges in assigning token-level credit based on action-level reward signals, resulting in discord between maximizing rewards and accurately modeling corpus data. In response to these challenges, we introduce Entropy-Regularized Token-level Policy Optimization (ETPO), an entropy-augmented RL method tailored for optimizing LLMs at the token level. At the heart of ETPO is our novel per-token soft Bellman update, designed to harmonize the RL process with the principles of language modeling. This methodology decomposes the Q-function update from a coarse action-level view to a more granular token-level perspective, backed by theoretical proof of optimization consistency. Crucially, this decomposition renders linear time complexity in action exploration. We assess the effectiveness of ETPO within a simulated environment that models data science code generation as a series of multi-step interactive tasks; results underline ETPO's potential as a robust method for refining the interactive decision-making capabilities of language agents. For a more detailed preliminary work describing our motivation for token-level decomposition and applying it in PPO methods, please refer to arXiv:2405.15821 .

## Citation

```bibtex
@misc{wen2024-entropy-regularized-token-level-policy-optimi,
  title = {{Entropy-Regularized Token-Level Policy Optimization for Language Agent Reinforcement}},
  author = {Muning Wen and Junwei Liao and Cheng Deng and Jun Wang and Weinan Zhang and Ying Wen},
  year = {2024},
  url = {https://arxiv.org/abs/2402.06700},
  eprint = {2402.06700},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2402.06700}
}
```

BibTeX download: https://yingwen.io/citations/entropy-regularized-token-level-policy-optimization-for-language-agent-reinforce.bib
