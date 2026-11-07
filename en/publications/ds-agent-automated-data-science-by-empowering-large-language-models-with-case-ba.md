# DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning

Authors: Siyuan Guo; Cheng Deng; Ying Wen; Hechang Chen; Yi Chang; Jun Wang

Publication: ICML (2024)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/ds-agent-automated-data-science-by-empowering-large-language-models-with-case-ba/
Publisher / primary source: https://proceedings.mlr.press/v235/guo24b.html
arXiv: https://arxiv.org/abs/2402.17453
PDF: https://raw.githubusercontent.com/mlresearch/v235/main/assets/guo24b/guo24b.pdf
Topics: [LLM agents](https://yingwen.io/en/research/llm-agents/)

## Abstract

In this work, we investigate the potential of large language models (LLMs) based agents to automate data science tasks, with the goal of comprehending task requirements, then building and training the best-fit machine learning models. Despite their widespread success, existing LLM agents are hindered by generating unreasonable experiment plans within this scenario. To this end, we present DS-Agent, a novel automatic framework that harnesses LLM agent and case-based reasoning (CBR). In the development stage, DS-Agent follows the CBR framework to structure an automatic iteration pipeline, which can flexibly capitalize on the expert knowledge from Kaggle, and facilitate consistent performance improvement through the feedback mechanism. Moreover, DS-Agent implements a low-resource deployment stage with a simplified CBR paradigm to adapt past successful solutions from the development stage for direct code generation, significantly reducing the demand on foundational capabilities of LLMs. Empirically, DS-Agent with GPT-4 achieves 100% success rate in the development stage, while attaining 36% improvement on average one pass rate across alternative LLMs in the deployment stage. In both stages, DS-Agent achieves the best rank in performance, costing $1.60 and \$0.13 per run with GPT-4, respectively. Our data and code are open-sourced at https://github.com/guosyjlu/DS-Agent.

## Citation

```bibtex
@inproceedings{pmlr-v235-guo24b,
  title = {{DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning}},
  author = {Siyuan Guo and Cheng Deng and Ying Wen and Hechang Chen and Yi Chang and Jun Wang},
  year = {2024},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  volume = {235},
  pages = {16813--16848},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v235/guo24b.html}
}
```

BibTeX download: https://yingwen.io/citations/ds-agent-automated-data-science-by-empowering-large-language-models-with-case-ba.bib
