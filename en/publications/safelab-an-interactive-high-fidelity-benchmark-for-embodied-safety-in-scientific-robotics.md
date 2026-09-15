# SafeLab: An Interactive High-Fidelity Benchmark for Embodied Safety in Scientific Robotics

Authors: Fengshuo Bai; Yufeng Li; Ruihai Wu; Peishuo Wang; Yuhan Wang; Bernie Hao Zhu; Yuanfei Wang; Tawei Chou; Jing Gao; Runchuan Zhu; Ying Wen; Yaodong Yang; Yuanpei Chen

Publication: ICML 2026 (2026)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/safelab-an-interactive-high-fidelity-benchmark-for-embodied-safety-in-scientific-robotics/
Publisher / primary source: https://openreview.net/forum?id=ojwpa0wXo9
PDF: https://openreview.net/pdf?id=ojwpa0wXo9
Topics: [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/); [Embodied AI & robot learning](https://yingwen.io/en/research/embodied-ai/)

## Abstract

Scientific embodied agents could automate laboratory workflows, but laboratory success is trajectory-level: an agent must remain safe throughout execution, not merely reach a final goal. In benchtop settings a small pose, force, or tilt error can cause irreversible spillage or equipment damage, yet most robot benchmarks evaluate reversible, high-tolerance manipulation and imitation-trained policies receive no recovery signal for execution drift. We introduce SafeLab, a generative simulation benchmark that couples a verified generative engine, an automated expert for teleoperation-free demonstrations, and a safety-aware RL interface, with 63 calibrated laboratory assets, 64 tasks across 9 manipulation categories, and 6,400 expert trajectories. Across state-of-the-art policies, it reveals unsafe-but-successful trajectories—large gaps between task success and Safe Success Rate—most acutely in liquid handling, force-limited actuation, and bimanual glassware rearrangement. Bounded residual RL then learns execution-level corrections that raise the simulated Safe Success Rate by up to 43.0 percentage points without retraining the base policy. Finally, 50 open-loop physical replays show 86% agreement between simulated and observed safety outcomes, supporting SafeLab as a scalable platform for screening and training safer laboratory agents.

## Citation

```bibtex
@inproceedings{bai2026-safelab-an-interactive-high-fidelity-benchmar,
  title = {{SafeLab: An Interactive High-Fidelity Benchmark for Embodied Safety in Scientific Robotics}},
  author = {Fengshuo Bai and Yufeng Li and Ruihai Wu and Peishuo Wang and Yuhan Wang and Bernie Hao Zhu and Yuanfei Wang and Tawei Chou and Jing Gao and Runchuan Zhu and Ying Wen and Yaodong Yang and Yuanpei Chen},
  year = {2026},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  volume = {306},
  publisher = {PMLR},
  url = {https://openreview.net/forum?id=ojwpa0wXo9}
}
```

BibTeX download: https://yingwen.io/citations/safelab-an-interactive-high-fidelity-benchmark-for-embodied-safety-in-scientific-robotics.bib
