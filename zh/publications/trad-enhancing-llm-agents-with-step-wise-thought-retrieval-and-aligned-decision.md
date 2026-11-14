# TRAD: Enhancing LLM Agents with Step-Wise Thought Retrieval and Aligned Decision

Authors: Ruiwen Zhou; Yingxuan Yang; Muning Wen; Ying Wen; Wenhao Wang; Chunling Xi; Guoqiang Xu; Yong Yu; Weinan Zhang

Publication: SIGIR (2024)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/trad-enhancing-llm-agents-with-step-wise-thought-retrieval-and-aligned-decision/
Publisher / primary source: https://doi.org/10.1145/3626772.3657788
DOI: https://doi.org/10.1145/3626772.3657788
arXiv: https://arxiv.org/abs/2403.06221
PDF: https://arxiv.org/pdf/2403.06221
Topics: [大模型智能体](https://yingwen.io/zh/research/llm-agents/)

## Abstract

Numerous large language model (LLM) agents have been built for different tasks like web navigation and online shopping due to LLM's wide knowledge and text-understanding ability. Among these works, many of them utilize in-context examples to achieve generalization without the need for fine-tuning, while few of them have considered the problem of how to select and effectively utilize these examples. Recently, methods based on trajectory-level retrieval with task meta-data and using trajectories as in-context examples have been proposed to improve the agent's overall performance in some sequential decision making tasks. However, these methods can be problematic due to plausible examples retrieved without task-specific state transition dynamics and long input with plenty of irrelevant context. In this paper, we propose a novel framework (TRAD) to address these issues. TRAD first conducts Thought Retrieval, achieving step-level demonstration selection via thought matching, leading to more helpful demonstrations and less irrelevant input noise. Then, TRAD introduces Aligned Decision, complementing retrieved demonstration steps with their previous or subsequent steps, which enables tolerance for imperfect thought and provides a choice for balance between more context and less noise. Extensive experiments on ALFWorld and Mind2Web benchmarks show that TRAD not only outperforms state-of-the-art models but also effectively helps in reducing noise and promoting generalization. Furthermore, TRAD has been deployed in real-world scenarios of a global business insurance company and improves the success rate of robotic process automation.

## Citation

```bibtex
@inproceedings{zhou2024-trad-enhancing-llm-agents-with-step-wise-thou,
  title = {{TRAD: Enhancing LLM Agents with Step-Wise Thought Retrieval and Aligned Decision}},
  author = {Ruiwen Zhou and Yingxuan Yang and Muning Wen and Ying Wen and Wenhao Wang and Chunling Xi and Guoqiang Xu and Yong Yu and Weinan Zhang},
  year = {2024},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages = {3--13},
  publisher = {ACM},
  doi = {10.1145/3626772.3657788},
  url = {https://doi.org/10.1145/3626772.3657788}
}
```

BibTeX download: https://yingwen.io/citations/trad-enhancing-llm-agents-with-step-wise-thought-retrieval-and-aligned-decision.bib
