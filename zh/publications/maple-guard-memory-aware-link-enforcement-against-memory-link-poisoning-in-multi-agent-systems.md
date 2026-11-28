# MAPLE-Guard: Memory-Aware Link Enforcement Against Memory-Link Poisoning in Multi-Agent Systems

Authors: Wenjun Xiong; Yijin Zhou; Jiaqian Wang; Shangding Gu; Bo Tang; Zhiyu Li; Feiyu Xiong; Ying Wen; Muning Wen

Publication: arXiv (2026)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/maple-guard-memory-aware-link-enforcement-against-memory-link-poisoning-in-multi-agent-systems/
Publisher / primary source: https://arxiv.org/abs/2608.00426
DOI: https://doi.org/10.48550/arXiv.2608.00426
arXiv: https://arxiv.org/abs/2608.00426
PDF: https://arxiv.org/pdf/2608.00426
Topics: [大模型智能体](https://yingwen.io/zh/research/llm-agents/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

LLM-based multi-agent systems (MAS) increasingly rely on persistent private and shared memories for long-horizon coordination. This memory layer improves continuity, but it also gives attackers a durable channel: a poisoned memory can be written once, continuously retrieved in later tasks, promoted into shared memory, and reused by other agents. A single poisoned write can therefore steer many later decisions and contaminate agents that never saw the original attack, all while no malicious message crosses a visible communication edge at the moment of harm. Further, because existing safeguards mainly inspect prompts, actions, or communication edges, they can miss attacks whose content appears benign at write time but becomes harmful after retrieval. We introduce Memory-Aware Propagation and Link Enforcement Guard, MAPLE-Guard, a memory-link guard for memory-enabled MAS. MAPLE-Guard monitors the memory lifecycle and places gates at write, retrieval, promotion, and cross-agent reuse, so risky memories can be quarantined, unsafe retrievals filtered, and poisoned private memories blocked before they enter shared memory. In the main evaluation, MAPLE-Guard lowers attack success rate (ASR) from 38.2% to 0.9% on LongMemEval and from 34.7% to 0.2% on AppWorld; it also raises multi-agent defense success rate (MDSR) from 54.0% to 74.3% and from 42.5% to 99.8% on the same benchmarks. These results suggest that memory-aware link enforcement covers a gap left by prompt-level and topology-level defenses. Code is available at the link: this https URL .

## Citation

```bibtex
@misc{xiong2026-maple-guard-memory-aware-link-enforcement-aga,
  title = {{MAPLE-Guard: Memory-Aware Link Enforcement Against Memory-Link Poisoning in Multi-Agent Systems}},
  author = {Wenjun Xiong and Yijin Zhou and Jiaqian Wang and Shangding Gu and Bo Tang and Zhiyu Li and Feiyu Xiong and Ying Wen and Muning Wen},
  year = {2026},
  url = {https://arxiv.org/abs/2608.00426},
  eprint = {2608.00426},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2608.00426}
}
```

BibTeX download: https://yingwen.io/citations/maple-guard-memory-aware-link-enforcement-against-memory-link-poisoning-in-multi-agent-systems.bib
