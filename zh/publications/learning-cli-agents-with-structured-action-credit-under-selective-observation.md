# Learning CLI Agents with Structured Action Credit under Selective Observation

Authors: Haoyang Su; Ying Wen

Publication: arXiv (2026)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/learning-cli-agents-with-structured-action-credit-under-selective-observation/
Publisher / primary source: https://arxiv.org/abs/2605.08013
DOI: https://doi.org/10.48550/arXiv.2605.08013
arXiv: https://arxiv.org/abs/2605.08013
PDF: https://arxiv.org/pdf/2605.08013
Topics: [大模型智能体](https://yingwen.io/zh/research/llm-agents/); [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Command line interface (CLI) agents are emerging as a practical paradigm for agent-computer interaction over evolving filesystems, executable command line programs, and online execution feedback. Recent work has used reinforcement learning (RL) to learn these interaction abilities from verifiable task feedback, yet few methods exploit the native structured attributes of CLI actions as learning signals. Beyond this underused action structure, CLI learning also couples two bottlenecks for coding agents. First, the agent must identify task-relevant evidence in a large codebase from partial observations. Second, sparse terminal rewards must be assigned to the actions that shape a long multi-turn trajectory. We study these bottlenecks through shell-driven information extraction and file editing tasks. For selective observation, we introduce $\sigma$-Reveal, an inference-time mechanism that selects token-budgeted context for the same CLI. For credit assignment, we propose Action Advantage Assignment ($\mathrm{A}^3$), a native agentic RL method that preserves the algorithmic complexity of standard agentic RL. $\mathrm{A}^3$ constructs turn-level advantages from episode-level relative feedback, abstract syntax tree (AST) based action sub-chain residuals, and tree-level trajectory margins. To further evaluate this problem setting, we construct ShellOps, a verifiable dataset suite covering CLI tasks in repository environments.

## Citation

```bibtex
@misc{su2026-learning-cli-agents-with-structured-action-cr,
  title = {{Learning CLI Agents with Structured Action Credit under Selective Observation}},
  author = {Haoyang Su and Ying Wen},
  year = {2026},
  url = {https://arxiv.org/abs/2605.08013},
  eprint = {2605.08013},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2605.08013}
}
```

BibTeX download: https://yingwen.io/citations/learning-cli-agents-with-structured-action-credit-under-selective-observation.bib
