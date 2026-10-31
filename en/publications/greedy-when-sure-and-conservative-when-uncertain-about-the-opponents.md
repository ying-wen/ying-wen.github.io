# Greedy when Sure and Conservative when Uncertain about the Opponents

Authors: Haobo Fu; Ye Tian; Hongxiang Yu; Weiming Liu; Shuang Wu; Jiechao Xiong; Ying Wen; Kai Li; Junliang Xing; Qiang Fu; Wei Yang

Publication: ICML (2022)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/greedy-when-sure-and-conservative-when-uncertain-about-the-opponents/
Publisher / primary source: https://proceedings.mlr.press/v162/fu22b.html
PDF: https://proceedings.mlr.press/v162/fu22b/fu22b.pdf
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

We develop a new approach, named Greedy when Sure and Conservative when Uncertain (GSCU), to competing online against unknown and nonstationary opponents. GSCU improves in four aspects: 1) introduces a novel way of learning opponent policy embeddings offline; 2) trains offline a single best response (conditional additionally on our opponent policy embedding) instead of a finite set of separate best responses against any opponent; 3) computes online a posterior of the current opponent policy embedding, without making the discrete and ineffective decision which type the current opponent belongs to; and 4) selects online between a real-time greedy policy and a fixed conservative policy via an adversarial bandit algorithm, gaining a theoretically better regret than adhering to either. Experimental studies on popular benchmarks demonstrate GSCU’s superiority over the state-of-the-art methods. The code is available online at https://github.com/YeTianJHU/GSCU.

## Citation

```bibtex
@inproceedings{pmlr-v162-fu22b,
  title = {{Greedy when Sure and Conservative when Uncertain about the Opponents}},
  author = {Haobo Fu and Ye Tian and Hongxiang Yu and Weiming Liu and Shuang Wu and Jiechao Xiong and Ying Wen and Kai Li and Junliang Xing and Qiang Fu and Wei Yang},
  year = {2022},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  volume = {162},
  pages = {6829--6848},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v162/fu22b.html}
}
```

BibTeX download: https://yingwen.io/citations/greedy-when-sure-and-conservative-when-uncertain-about-the-opponents.bib
