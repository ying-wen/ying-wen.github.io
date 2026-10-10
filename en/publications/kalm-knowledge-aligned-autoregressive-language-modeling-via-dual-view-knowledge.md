# KaLM: Knowledge-aligned Autoregressive Language Modeling via Dual-view Knowledge Graph Contrastive Learning

Authors: Peng Yu; Cheng Deng; Beiya Dai; Xinbing Wang; Ying Wen

Publication: arXiv (2024)
Type: Preprint
Canonical page: https://yingwen.io/en/publications/kalm-knowledge-aligned-autoregressive-language-modeling-via-dual-view-knowledge/
Publisher / primary source: https://arxiv.org/abs/2412.04948
DOI: https://doi.org/10.48550/arXiv.2412.04948
arXiv: https://arxiv.org/abs/2412.04948
PDF: https://arxiv.org/pdf/2412.04948
Topics: [Large language models & reasoning](https://yingwen.io/en/research/llms/)

## Abstract

Autoregressive large language models (LLMs) pre-trained by next token prediction are inherently proficient in generative tasks. However, their performance on knowledge-driven tasks such as factual knowledge querying remains unsatisfactory. Knowledge graphs (KGs), as high-quality structured knowledge bases, can provide reliable knowledge for LLMs, potentially compensating for their knowledge deficiencies. Aligning LLMs with explicit, structured knowledge from KGs has been a challenge; previous attempts either failed to effectively align knowledge representations or compromised the generative capabilities of LLMs, leading to less-than-optimal outcomes. This paper proposes \textbf{KaLM}, a \textit{Knowledge-aligned Language Modeling} approach, which fine-tunes autoregressive LLMs to align with KG knowledge via the joint objective of explicit knowledge alignment and implicit knowledge alignment. The explicit knowledge alignment objective aims to directly optimize the knowledge representation of LLMs through dual-view knowledge graph contrastive learning. The implicit knowledge alignment objective focuses on incorporating textual patterns of knowledge into LLMs through triple completion language modeling. Notably, our method achieves a significant performance boost in evaluations of knowledge-driven tasks, specifically embedding-based knowledge graph completion and generation-based knowledge graph question answering.

## Citation

```bibtex
@misc{yu2024-kalm-knowledge-aligned-autoregressive-languag,
  title = {{KaLM: Knowledge-aligned Autoregressive Language Modeling via Dual-view Knowledge Graph Contrastive Learning}},
  author = {Peng Yu and Cheng Deng and Beiya Dai and Xinbing Wang and Ying Wen},
  year = {2024},
  url = {https://arxiv.org/abs/2412.04948},
  eprint = {2412.04948},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2412.04948}
}
```

BibTeX download: https://yingwen.io/citations/kalm-knowledge-aligned-autoregressive-language-modeling-via-dual-view-knowledge.bib
