# Offline Pre-trained Multi-agent Decision Transformer

Authors: Linghui Meng; Muning Wen; Chenyang Le; Xiyun Li; Dengpeng Xing; Weinan Zhang; Ying Wen; Haifeng Zhang; Jun Wang; Yaodong Yang; Bo Xu

Publication: Machine Intelligence Research (2023)
Type: Journal article
Canonical page: https://yingwen.io/en/publications/offline-pre-trained-multi-agent-decision-transformer/
Publisher / primary source: https://doi.org/10.1007/s11633-022-1383-7
DOI: https://doi.org/10.1007/s11633-022-1383-7
arXiv: https://arxiv.org/abs/2112.02845
PDF: https://arxiv.org/pdf/2112.02845
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Offline reinforcement learning leverages previously collected offline datasets to learn optimal policies with no necessity to access the real environment. Such a paradigm is also desirable for multi-agent reinforcement learning (MARL) tasks, given the combinatorially increased interactions among agents and with the environment. However, in MARL, the paradigm of offline pre-training with online fine-tuning has not been studied, nor even datasets or benchmarks for offline MARL research are available. In this paper, we facilitate the research by providing large-scale datasets and using them to examine the usage of the decision transformer in the context of MARL. We investigate the generalization of MARL offline pre-training in the following three aspects: 1) between single agents and multiple agents, 2) from offline pretraining to online fine tuning, and 3) to that of multiple downstream tasks with few-shot and zero-shot capabilities. We start by introducing the first offline MARL dataset with diverse quality levels based on the StarCraftII environment, and then propose the novel architecture of multi-agent decision transformer (MADT) for effective offline learning. MADT leverages the transformer’s modelling ability for sequence modelling and integrates it seamlessly with both offline and online MARL tasks. A significant benefit of MADT is that it learns generalizable policies that can transfer between different types of agents under different task scenarios. On the StarCraft II offline dataset, MADT outperforms the state-of-the-art offline reinforcement learning (RL) baselines, including BCQ and CQL. When applied to online tasks, the pre-trained MADT significantly improves sample efficiency and enjoys strong performance in both few-short and zero-shot cases. To the best of our knowledge, this is the first work that studies and demonstrates the effectiveness of offline pre-trained models in terms of sample efficiency and generalizability enhancements for MARL

## Citation

```bibtex
@article{meng2023-offline-pre-trained-multi-agent-decision-tran,
  title = {{Offline Pre-trained Multi-agent Decision Transformer}},
  author = {Linghui Meng and Muning Wen and Chenyang Le and Xiyun Li and Dengpeng Xing and Weinan Zhang and Ying Wen and Haifeng Zhang and Jun Wang and Yaodong Yang and Bo Xu},
  year = {2023},
  journal = {Machine Intelligence Research},
  volume = {20},
  number = {2},
  pages = {233--248},
  publisher = {Springer Science and Business Media LLC},
  doi = {10.1007/s11633-022-1383-7},
  url = {https://doi.org/10.1007/s11633-022-1383-7}
}
```

BibTeX download: https://yingwen.io/citations/offline-pre-trained-multi-agent-decision-transformer.bib
