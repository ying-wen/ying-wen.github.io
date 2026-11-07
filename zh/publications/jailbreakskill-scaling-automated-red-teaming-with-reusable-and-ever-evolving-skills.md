# JailbreakSkill: Scaling Automated Red-Teaming with Reusable and Ever-Evolving Skills

Authors: Xiaoyu Wen; Jiajia Li; Zhida He; Peng Yu; Chenxu Wang; Han Qi; Ziyuan Zhou; Cheng Jin; Ying Wen; Xingcheng Xu; Shuyue Hu; Tianhang Zheng; Chaochao Lu; Qiaosheng Zhang

Publication: arXiv (2026)
Type: Preprint
Canonical page: https://yingwen.io/zh/publications/jailbreakskill-scaling-automated-red-teaming-with-reusable-and-ever-evolving-skills/
Publisher / primary source: https://arxiv.org/abs/2608.16465
DOI: https://doi.org/10.48550/arXiv.2608.16465
arXiv: https://arxiv.org/abs/2608.16465
PDF: https://arxiv.org/pdf/2608.16465
Topics: [大模型与推理](https://yingwen.io/zh/research/llms/); [大模型智能体](https://yingwen.io/zh/research/llm-agents/)

## Abstract

Automated red-teaming has produced a growing collection of attack strategies, yet they typically remain scattered across prompts and workflows, making them difficult to systematically integrate, reuse, and improve at scale. We introduce \textsc{JailbreakSkill}, a skill-centric framework for scaling automated red-teaming through reusable and continuously evolving attack capabilities. \textsc{JailbreakSkill} packages existing attack strategies into modular, agent-ready skills that can be directly reused and adaptively selected across tasks and target models. Beyond reuse, it closes the loop between attacking and learning: attack experience is used to diagnose, refine, combine, and discover new skills, which are added back to an ever-growing skill library. This evolution lifts macro-average ASR by 17.5 percentage points on AdvBench and 13.4 points on HarmBench, including a 48.6-point gain against GPT-5.4 on AdvBench, while yielding novel attack strategies such as reframing a direct request as an unfinished document-completion task. Several evolved skills also generalize to unseen prompts and target models without further adaptation. Our code is available at this https URL .

## Citation

```bibtex
@misc{wen2026-jailbreakskill-scaling-automated-red-teaming-,
  title = {{JailbreakSkill: Scaling Automated Red-Teaming with Reusable and Ever-Evolving Skills}},
  author = {Xiaoyu Wen and Jiajia Li and Zhida He and Peng Yu and Chenxu Wang and Han Qi and Ziyuan Zhou and Cheng Jin and Ying Wen and Xingcheng Xu and Shuyue Hu and Tianhang Zheng and Chaochao Lu and Qiaosheng Zhang},
  year = {2026},
  url = {https://arxiv.org/abs/2608.16465},
  eprint = {2608.16465},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2608.16465}
}
```

BibTeX download: https://yingwen.io/citations/jailbreakskill-scaling-automated-red-teaming-with-reusable-and-ever-evolving-skills.bib
