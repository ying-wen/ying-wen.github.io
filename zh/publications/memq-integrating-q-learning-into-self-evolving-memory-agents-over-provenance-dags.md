# MemQ: Integrating Q-Learning into Self-Evolving Memory Agents over Provenance DAGs

Authors: Junwei Liao; Haoting Shi; Ruiwen Zhou; Jiaqian Wang; Shengtao Zhang; Wei Zhang; Ying Wen; Zhiyu Li; Feiyu Xiong; Bo Tang; Weinan Zhang; Muning Wen

Publication: arXiv (2026)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/memq-integrating-q-learning-into-self-evolving-memory-agents-over-provenance-dags/
Publisher / primary source: https://arxiv.org/abs/2605.08374
DOI: https://doi.org/10.48550/arXiv.2605.08374
arXiv: https://arxiv.org/abs/2605.08374
PDF: https://arxiv.org/pdf/2605.08374
Topics: [大模型智能体](https://yingwen.io/zh/research/llm-agents/); [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Episodic memory allows LLM agents to accumulate and retrieve experience, but current methods treat each memory independently, i.e., evaluating retrieval quality in isolation without accounting for the dependency chains through which memories enable the creation of future memories. We introduce MemQ, which applies TD($\lambda$) eligibility traces to memory Q-values, propagating credit backward through a provenance DAG that records which memories were retrieved when each new memory was created. Credit weight decays as $(\gamma\lambda)^d$ with DAG depth $d$, replacing temporal distance with structural proximity. We formalize the setting as an Exogenous-Context MDP, whose factored transition decouples the exogenous task stream from the endogenous memory store. Across six benchmarks, spanning OS interaction, function calling, code generation, multimodal reasoning, embodied reasoning, and expert-level QA, MemQ achieves the highest success rate on all six in generalization evaluation and runtime learning, with gains largest on multi-step tasks that produce deep and relevant provenance chains (up to +5.7~pp) and smallest on single-step classification (+0.77~pp) where single-step updates already suffice. We further study how $\gamma$ and $\lambda$ interact with the EC-MDP structure, providing principled guidance for parameter selection and future research. Code is available at this https URL .

## Citation

```bibtex
@misc{liao2026-memq-integrating-q-learning-into-self-evolvin,
  title = {{MemQ: Integrating Q-Learning into Self-Evolving Memory Agents over Provenance DAGs}},
  author = {Junwei Liao and Haoting Shi and Ruiwen Zhou and Jiaqian Wang and Shengtao Zhang and Wei Zhang and Ying Wen and Zhiyu Li and Feiyu Xiong and Bo Tang and Weinan Zhang and Muning Wen},
  year = {2026},
  url = {https://arxiv.org/abs/2605.08374},
  eprint = {2605.08374},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2605.08374}
}
```

BibTeX download: https://yingwen.io/citations/memq-integrating-q-learning-into-self-evolving-memory-agents-over-provenance-dags.bib
