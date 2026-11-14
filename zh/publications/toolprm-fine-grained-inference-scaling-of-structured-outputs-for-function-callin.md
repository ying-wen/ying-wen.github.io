# ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling

Authors: Jianghao Lin; Yuanyuan Shi; Xin Peng; Renjie Ding; Hairui Wang; Yuxuan Peng; Bizhe Bai; Weixi Song; Fengshuo Bai; Huacan Chai; Weinan Zhang; Fei Huang; Ying Wen

Publication: ACL (2026)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/toolprm-fine-grained-inference-scaling-of-structured-outputs-for-function-callin/
Publisher / primary source: https://aclanthology.org/2026.acl-long.855/
DOI: https://doi.org/10.18653/v1/2026.acl-long.855
arXiv: https://arxiv.org/abs/2510.14703
PDF: https://aclanthology.org/2026.acl-long.855.pdf
Topics: [大模型与推理](https://yingwen.io/zh/research/llms/); [大模型智能体](https://yingwen.io/zh/research/llm-agents/)

## Abstract

Large language models (LLMs) excel at function calling, but inference scaling has been explored mainly for unstructured generation. We propose an inference-scaling framework for structured outputs that combines fine-grained beam search with ToolPRM, a process reward model scoring each intra-call decision (function name and argument filling). We build the first fine-grained intra-call supervision dataset via function masking, rollout collection, and step-level annotation. ToolPRM outperforms outcome and coarse-grained reward models in predictive accuracy and yields consistent test-time gains on multiple function-calling benchmarks. We further show that structured generation follows “explore more but retain less”, since early JSON errors are unrecoverable.

## Citation

```bibtex
@inproceedings{lin-etal-2026-toolprm,
  title = {{ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling}},
  author = {Jianghao Lin and Yuanyuan Shi and Xin Peng and Renjie Ding and Hairui Wang and Yuxuan Peng and Bizhe Bai and Weixi Song and Fengshuo Bai and Huacan Chai and Weinan Zhang and Fei Huang and Ying Wen},
  year = {2026},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {18792--18804},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2026.acl-long.855},
  url = {https://aclanthology.org/2026.acl-long.855/}
}
```

BibTeX download: https://yingwen.io/citations/toolprm-fine-grained-inference-scaling-of-structured-outputs-for-function-callin.bib
