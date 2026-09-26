# DSR: optimization of performance lower bound for hierarchical policy with dynamical skill refinement

Authors: Dongxiang Chen; Ying Wen

Publication: Frontiers of Computer Science (2026)
Type: Journal article
Canonical page: https://yingwen.io/zh/publications/dsr-optimization-of-performance-lower-bound-for-hierarchical-policy-with-dynamical-skill-refinement/
Publisher / primary source: https://doi.org/10.1007/s11704-025-50561-3
DOI: https://doi.org/10.1007/s11704-025-50561-3
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Introduction (excerpt)

To address these challenges, we propose an on-policy skill-based RL method that jointly optimizes high-level policy and skills to maximize a performance lower bound of the hierarchical policy in original MDP. By synchronously updating both policy levels only at each epoch’s end, our method inherently avoids temporal abstraction shift. Crucially, we unify the optimization objectives for high-level policy and skills, thereby guaranteeing performance improvement in TA-MDP. We establish theoretical equivalence between optimizing performance in TA-MDP and optimizing the hierarchical policy’s performance lower bound in original MDP, given an appropriate choice of TA-MDP’s discount factor. This fills the theoretical gap of fine-tuning entire hierarchical policy. We further introduce a Dynamical Skill Refinement (DSR) mechanism—the namesake of our method—that incorporates skill refinement via a residual low-level policy. DSR dynamically weights the action increment predicted by this residual policy. This mechanism addresses a critical challenge: since all skills are embedded to the same parametric low-level policy’s latent space, refining one skill may inadvertently introduce detrimental behavioral changes to others, which may destroy well-initialized skills. We name this potential phenomenon skill space collapse. DSR tries to preserve each skill’s initial behavior until it obtains sufficient refinements which offset detrimental changes.

## Citation

```bibtex
@article{chen2026-dsr-optimization-of-performance-lower-bound-f,
  title = {{DSR: optimization of performance lower bound for hierarchical policy with dynamical skill refinement}},
  author = {Dongxiang Chen and Ying Wen},
  year = {2026},
  journal = {Frontiers of Computer Science},
  volume = {20},
  number = {6},
  pages = {2006347},
  publisher = {Springer Science and Business Media LLC},
  doi = {10.1007/s11704-025-50561-3},
  url = {https://doi.org/10.1007/s11704-025-50561-3}
}
```

BibTeX download: https://yingwen.io/citations/dsr-optimization-of-performance-lower-bound-for-hierarchical-policy-with-dynamical-skill-refinement.bib
