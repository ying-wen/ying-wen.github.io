# Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning

Authors: Jiachen Zhu; Congmin Zheng; Jianghao Lin; Kounianhua Du; Ying Wen; Yong Yu; Jun Wang; Weinan Zhang

Publication: Findings of ACL 2025 (2025)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/retrieval-augmented-process-reward-model-for-generalizable-mathematical-reasonin/
Publisher / primary source: https://aclanthology.org/2025.findings-acl.444/
DOI: https://doi.org/10.18653/v1/2025.findings-acl.444
arXiv: https://arxiv.org/abs/2502.14361
PDF: https://aclanthology.org/2025.findings-acl.444.pdf
Topics: [大模型与推理](https://yingwen.io/zh/research/llms/)

## Abstract

While large language models (LLMs) have significantly advanced mathematical reasoning, Process Reward Models (PRMs) have been developed to evaluate the logical validity of reasoning steps. However, PRMs still struggle with out-of-distribution (OOD) challenges. This paper identifies the OOD issues including step OOD, arising from differences in reasoning patterns across model types and sizes, and question OOD, due to dataset shifts between training and real-world problems. To address these issues, we introduce Retrieval-Augmented Process Reward Model (RetrievalPRM), a novel framework designed to tackle these OOD issues. By utilizing a two-stage retrieval-enhanced mechanism, RetrievalPRM retrieves semantically similar questions and steps for PRM as a warmup to stimulate its potential to judge target steps, improving generalization and reasoning consistency across different models and problem types. Our extensive experiments demonstrate that RetrievalPRM outperforms existing baselines across multiple real-world datasets. Our open-source contributions include a retrieval-enhanced dataset, a tuning framework for PRM training, and the RetreivalPRM model, establishing a new standard for PRM performance.

## Citation

```bibtex
@inproceedings{zhu-etal-2025-retrieval,
  title = {{Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning}},
  author = {Jiachen Zhu and Congmin Zheng and Jianghao Lin and Kounianhua Du and Ying Wen and Yong Yu and Jun Wang and Weinan Zhang},
  year = {2025},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages = {8453--8468},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2025.findings-acl.444},
  url = {https://aclanthology.org/2025.findings-acl.444/}
}
```

BibTeX download: https://yingwen.io/citations/retrieval-augmented-process-reward-model-for-generalizable-mathematical-reasonin.bib
