# Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity

Authors: Yingxuan Yang; Chengrui Qu; Muning Wen; Laixi Shi; Ying Wen; Weinan Zhang; Adam Wierman; Shangding Gu

Publication: arXiv (2026)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/understanding-agent-scaling-in-llm-based-multi-agent-systems-via-diversity/
Publisher / primary source: https://arxiv.org/abs/2602.03794
DOI: https://doi.org/10.48550/arXiv.2602.03794
arXiv: https://arxiv.org/abs/2602.03794
PDF: https://arxiv.org/pdf/2602.03794
Topics: [大模型智能体](https://yingwen.io/zh/research/llm-agents/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/)

## Abstract

LLM-based multi-agent systems (MAS) have emerged as a promising approach to tackle complex tasks that are difficult for individual LLMs. A natural strategy is to scale performance by increasing the number of agents; however, we find that such scaling exhibits strong diminishing returns in homogeneous settings, while introducing heterogeneity (e.g., different models, prompts, or tools) continues to yield substantial gains. This raises a fundamental question: what limits scaling, and why does diversity help? We present an information-theoretic framework showing that MAS performance is bounded by the intrinsic task uncertainty, not by agent count. We derive architecture-agnostic bounds demonstrating that improvements depend on how many effective channels the system accesses. Homogeneous agents saturate early because their outputs are strongly correlated, whereas heterogeneous agents contribute complementary evidence. We further introduce $K^*$, an effective channel count that quantifies the number of effective channels without ground-truth labels. Empirically, we show that heterogeneous configurations consistently outperform homogeneous scaling: 2 diverse agents can match or exceed the performance of 16 homogeneous agents. Our results provide principled guidelines for building efficient and robust MAS through diversity-aware design. Code and Dataset are available at the link: this https URL .

## Citation

```bibtex
@misc{yang2026-understanding-agent-scaling-in-llm-based-mult,
  title = {{Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity}},
  author = {Yingxuan Yang and Chengrui Qu and Muning Wen and Laixi Shi and Ying Wen and Weinan Zhang and Adam Wierman and Shangding Gu},
  year = {2026},
  url = {https://arxiv.org/abs/2602.03794},
  eprint = {2602.03794},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2602.03794}
}
```

BibTeX download: https://yingwen.io/citations/understanding-agent-scaling-in-llm-based-multi-agent-systems-via-diversity.bib
