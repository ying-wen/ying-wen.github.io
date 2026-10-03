# Modelling Behavioural Diversity for Learning in Open-Ended Games

Authors: Nicolas Perez-Nieves; Yaodong Yang; Oliver Slumbers; David H Mguni; Ying Wen; Jun Wang

Publication: ICML (2021)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/modelling-behavioural-diversity-for-learning-in-open-ended-games/
Publisher / primary source: https://proceedings.mlr.press/v139/perez-nieves21a.html
arXiv: https://arxiv.org/abs/2103.07927
PDF: http://proceedings.mlr.press/v139/perez-nieves21a/perez-nieves21a.pdf
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Promoting behavioural diversity is critical for solving games with non-transitive dynamics where strategic cycles exist, and there is no consistent winner (e.g., Rock-Paper-Scissors). Yet, there is a lack of rigorous treatment for defining diversity and constructing diversity-aware learning dynamics. In this work, we offer a geometric interpretation of behavioural diversity in games and introduce a novel diversity metric based on \emph{determinantal point processes} (DPP). By incorporating the diversity metric into best-response dynamics, we develop \emph{diverse fictitious play} and \emph{diverse policy-space response oracle} for solving normal-form games and open-ended games. We prove the uniqueness of the diverse best response and the convergence of our algorithms on two-player games. Importantly, we show that maximising the DPP-based diversity metric guarantees to enlarge the \emph{gamescape} – convex polytopes spanned by agents’ mixtures of strategies. To validate our diversity-aware solvers, we test on tens of games that show strong non-transitivity. Results suggest that our methods achieve at least the same, and in most games, lower exploitability than PSRO solvers by finding effective and diverse strategies.

## Citation

```bibtex
@inproceedings{pmlr-v139-perez-nieves21a,
  title = {{Modelling Behavioural Diversity for Learning in Open-Ended Games}},
  author = {Nicolas Perez-Nieves and Yaodong Yang and Oliver Slumbers and David H Mguni and Ying Wen and Jun Wang},
  year = {2021},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  volume = {139},
  pages = {8514--8524},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v139/perez-nieves21a.html}
}
```

BibTeX download: https://yingwen.io/citations/modelling-behavioural-diversity-for-learning-in-open-ended-games.bib
