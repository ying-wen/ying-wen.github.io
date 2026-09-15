# Efficient Policy Space Response Oracles

Authors: Ming Zhou; Jingxiao Chen; Ying Wen; Weinan Zhang; Yaodong Yang; Yong Yu; Jun Wang

Publication: arXiv (2022)
Type: Preprint
Canonical page: https://yingwen.io/en/publications/efficient-policy-space-response-oracles/
Publisher / primary source: https://arxiv.org/abs/2202.00633
DOI: https://doi.org/10.48550/arXiv.2202.00633
arXiv: https://arxiv.org/abs/2202.00633
PDF: https://arxiv.org/pdf/2202.00633
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/)

## Abstract

Policy Space Response Oracle methods (PSRO) provide a general solution to learn Nash equilibrium in two-player zero-sum games but suffer from two drawbacks: (1) the computation inefficiency due to the need for consistent meta-game evaluation via simulations, and (2) the exploration inefficiency due to finding the best response against a fixed meta-strategy at every epoch. In this work, we propose Efficient PSRO (EPSRO) that largely improves the efficiency of the above two steps. Central to our development is the newly-introduced subroutine of no-regret optimization on the unrestricted-restricted (URR) game. By solving URR at each epoch, one can evaluate the current game and compute the best response in one forward pass without the need for meta-game simulations. Theoretically, we prove that the solution procedures of EPSRO offer a monotonic improvement on the exploitability, which none of existing PSRO methods possess. Furthermore, we prove that the no-regret optimization has a regret bound of $\mathcal{O}(\sqrt{T\log{[(k^2+k)/2]}})$, where $k$ is the size of restricted policy set. Most importantly, a desirable property of EPSRO is that it is parallelizable, this allows for highly efficient exploration in the policy space that induces behavioral diversity. We test EPSRO on three classes of games, and report a 50x speedup in wall-time and 10x data efficiency while maintaining similar exploitability as existing PSRO methods on Kuhn and Leduc Poker games.

## Citation

```bibtex
@misc{zhou2022-efficient-policy-space-response-oracles,
  title = {{Efficient Policy Space Response Oracles}},
  author = {Ming Zhou and Jingxiao Chen and Ying Wen and Weinan Zhang and Yaodong Yang and Yong Yu and Jun Wang},
  year = {2022},
  url = {https://arxiv.org/abs/2202.00633},
  eprint = {2202.00633},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2202.00633}
}
```

BibTeX download: https://yingwen.io/citations/efficient-policy-space-response-oracles.bib
