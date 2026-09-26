# Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis

Authors: Yujie Zheng; Zhuo Li; Shengtao Zhang; Hanjing Wang; Junjie Sheng; Jiaqian Wang; Junchi Yan; Weinan Zhang; Ying Wen; Bo Tang; Muning Wen

Publication: arXiv (2026)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/towards-cold-start-drafting-and-continual-refining-a-value-driven-memory-approach-with-application-to-npu-kernel-synthesis/
Publisher / primary source: https://arxiv.org/abs/2603.10846
DOI: https://doi.org/10.48550/arXiv.2603.10846
arXiv: https://arxiv.org/abs/2603.10846
PDF: https://arxiv.org/pdf/2603.10846
Topics: [大模型智能体](https://yingwen.io/zh/research/llm-agents/); [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Deploying Large Language Models to data-scarce programming domains poses significant challenges, particularly for kernel synthesis on emerging Domain-Specific Architectures where a "Data Wall" limits available training data. While models excel on data-rich platforms like CUDA, they suffer catastrophic performance drops on data-scarce ecosystems such as NPU programming. To overcome this cold-start barrier without expensive fine-tuning, we introduce EvoKernel, a self-evolving agentic framework that automates the lifecycle of kernel synthesis from initial drafting to continual refining. EvoKernel addresses this by formulating the synthesis process as a memory-based reinforcement learning task. Through a novel value-driven retrieval mechanism, it learns stage-specific Q-values that prioritize experiences based on their contribution to the current objective, whether bootstrapping a feasible draft or iteratively refining latency. Furthermore, by enabling cross-task memory sharing, the agent generalizes insights from simple to complex operators. By building an NPU variant of KernelBench and evaluating on it, EvoKernel improves frontier models' correctness from 11.0% to 83.0% and achieves a median speedup of 3.60x over initial drafts through iterative refinement. This demonstrates that value-guided experience accumulation allows general-purpose models to master the kernel synthesis task on niche hardware ecosystems. Our official page is available at this https URL .

## Citation

```bibtex
@misc{zheng2026-towards-cold-start-drafting-and-continual-ref,
  title = {{Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis}},
  author = {Yujie Zheng and Zhuo Li and Shengtao Zhang and Hanjing Wang and Junjie Sheng and Jiaqian Wang and Junchi Yan and Weinan Zhang and Ying Wen and Bo Tang and Muning Wen},
  year = {2026},
  url = {https://arxiv.org/abs/2603.10846},
  eprint = {2603.10846},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2603.10846}
}
```

BibTeX download: https://yingwen.io/citations/towards-cold-start-drafting-and-continual-refining-a-value-driven-memory-approach-with-application-to-npu-kernel-synthesis.bib
