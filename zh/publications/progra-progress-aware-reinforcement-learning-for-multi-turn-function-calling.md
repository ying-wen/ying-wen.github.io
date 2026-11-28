# Progra: Progress-Aware Reinforcement Learning for Multi-Turn Function Calling

Authors: Huacan Chai; Zijie Cao; Maolin Ran; Yingxuan Yang; Jianghao Lin; Xin Peng; Hairui Wang; Renjie Ding; Ziyu Wan; Muning Wen; Weiwen Liu; Weinan Zhang; Fei Huang; Ying Wen

Publication: Findings of ACL 2026 (2026)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/progra-progress-aware-reinforcement-learning-for-multi-turn-function-calling/
Publisher / primary source: https://aclanthology.org/2026.findings-acl.325/
DOI: https://doi.org/10.18653/v1/2026.findings-acl.325
PDF: https://aclanthology.org/2026.findings-acl.325.pdf
Code: https://github.com/FatCatCHC/Progra
Topics: [大模型智能体](https://yingwen.io/zh/research/llm-agents/); [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Large language models (LLMs) have achieved impressive success in single-turn function calling, yet real-world applications such as travel planning or multi-stage data analysis typically unfold across multi-turn conversations. In these settings, LLMs must not only issue accurate function calls at each step but also maintain progress awareness, the ability to summarize past interactions and plan future actions to ensure coherent, long-horizon task execution. Existing approaches, however, either reduce multi-turn training to isolated single-turn samples, which neglects task-level planning, or employ end-to-end reinforcement learning (RL) that struggles with redundancy and lacks explicit integration of progress awareness. To overcome these limitations, we introduce Progra, a framework that explicitly incorporates progress awareness into LLM training for multi-turn function calling. Progra combines (i) a Progress Awareness Generation (PAG) pipeline, which automatically constructs datasets coupling conversation summaries with future task planning, and (ii) a Progress Awareness-Guided Reinforcement Learning (PAG-RL) algorithm, which integrates progress awareness into RL training to reduce contextual redundancy and improve alignment between local actions and global task completion. Empirical results on two public benchmarks demonstrate that Progra significantly outperforms existing methods, highlighting the effectiveness of progress awareness in enabling robust and efficient multi-turn function calling. Our code is available at https://github.com/FatCatCHC/Progra .

## Citation

```bibtex
@inproceedings{chai-etal-2026-progra,
  title = {{Progra: Progress-Aware Reinforcement Learning for Multi-Turn Function Calling}},
  author = {Huacan Chai and Zijie Cao and Maolin Ran and Yingxuan Yang and Jianghao Lin and Xin Peng and Hairui Wang and Renjie Ding and Ziyu Wan and Muning Wen and Weiwen Liu and Weinan Zhang and Fei Huang and Ying Wen},
  year = {2026},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages = {6519--6535},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2026.findings-acl.325},
  url = {https://aclanthology.org/2026.findings-acl.325/}
}
```

BibTeX download: https://yingwen.io/citations/progra-progress-aware-reinforcement-learning-for-multi-turn-function-calling.bib
