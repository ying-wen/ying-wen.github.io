# Accelerating Multi‑turn LLM Agent Workflows via Context Templating and Opportunistic Prefill

Authors: Hanjing Wang; Ying Wen; Weinan Zhang

Publication: ICIC (2025)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/accelerating-multi-turn-llm-agent-workflows-via-context-templating-and-opportunistic-prefill/
Publisher / primary source: https://doi.org/10.1007/978-981-96-9901-8_14
DOI: https://doi.org/10.1007/978-981-96-9901-8_14
PDF: https://link.springer.com/content/pdf/10.1007/978-981-96-9901-8_14.pdf
Topics: [Large language models & reasoning](https://yingwen.io/en/research/llms/); [LLM agents](https://yingwen.io/en/research/llm-agents/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Large Language Model (LLM) based agents have become essential interfaces for delivering AI-assisted services in daily workflows. However, these agentic workflows typically involve complex in-context dependencies, long context histories and small batch sizes, which lead to under-utilization of GPU resources and elevated end-to-end generation latency. To tackle this challenge, we propose a context templating mechanism that decouples agents’ context as a combination of static templates and dynamic string variables, allowing partial prompt construction and runtime updates. Built on this abstraction, our execution engine resolves dependencies, identifies satisfiable contexts and issues generation calls as soon as their dependencies are met, without waiting for unrelated segments. Known context portions are opportunistically prefilled during idle GPU cycles, priming the key-value (KV) cache to reduce decoding overhead. This strategy not only enhances GPU utilization via asynchronous prefill but also eliminates unnecessary serialization between generation steps, thus maximizing opportunities for intra-conversation parallelism. We implement these techniques in a modular orchestration framework supporting multi-agent, multi-turn workflows. In a simulated multi-agent debate task, our method reduces up to 20% end-to-end generation latency in long-context, low-batch settings compared to a standard Python-threaded baseline. The benefits diminish at higher batch sizes, where traditional batching is sufficient. Our results highlight that leveraging the structural regularities of agent conversations can meaningfully accelerate LLM-backed services.

## Citation

```bibtex
@inproceedings{wang2025-accelerating-multi-turn-llm-agent-workflows-v,
  title = {{Accelerating Multi‑turn LLM Agent Workflows via Context Templating and Opportunistic Prefill}},
  author = {Hanjing Wang and Ying Wen and Weinan Zhang},
  year = {2025},
  booktitle = {Advanced Intelligent Computing Technology and Applications},
  pages = {169--179},
  publisher = {Springer, Singapore},
  doi = {10.1007/978-981-96-9901-8\_14},
  url = {https://doi.org/10.1007/978-981-96-9901-8_14}
}
```

BibTeX download: https://yingwen.io/citations/accelerating-multi-turn-llm-agent-workflows-via-context-templating-and-opportunistic-prefill.bib
