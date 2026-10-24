# RAT: Adversarial Attacks on Deep Reinforcement Agents for Targeted Behaviors

Authors: Fengshuo Bai; Runze Liu; Yali Du; Ying Wen; Yaodong Yang

Publication: AAAI 2025 (2025)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/rat-aaai-2025/
Publisher / primary source: https://doi.org/10.1609/aaai.v39i15.33696
DOI: https://doi.org/10.1609/aaai.v39i15.33696
arXiv: https://arxiv.org/abs/2412.10713
PDF: https://arxiv.org/pdf/2412.10713
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/)

## Abstract

Evaluating deep reinforcement learning (DRL) agents against targeted behavior attacks is critical for assessing their robustness. These attacks aim to manipulate the victim into specific behaviors that align with the attacker’s objectives, often bypassing traditional reward-based defenses. Prior methods have primarily focused on reducing cumulative rewards; however, rewards are typically too generic to capture complex safety requirements effectively. As a result, focusing solely on reward reduction can lead to suboptimal attack strategies, particularly in safety-critical scenarios where more precise behavior manipulation is needed. To address these challenges, we propose RAT, a method designed for universal, targeted behavior attacks. RAT trains an intention policy that is explicitly aligned with human preferences, serving as a precise behavioral target for the adversary. Concurrently, an adversary manipulates the victim's policy to follow this target behavior. To enhance the effectiveness of these attacks, RAT dynamically adjusts the state occupancy measure within the replay buffer, allowing for more controlled and effective behavior manipulation. Our empirical results on robotic simulation tasks demonstrate that RAT outperforms existing adversarial attack algorithms in inducing specific behaviors. Additionally, RAT shows promise in improving agent robustness, leading to more resilient policies. We further validate RAT by guiding Decision Transformer agents to adopt behaviors aligned with human preferences in various MuJoCo tasks, demonstrating its effectiveness across diverse tasks

## Citation

```bibtex
@inproceedings{bai2025-rat-aaai-2025,
  title = {{RAT: Adversarial Attacks on Deep Reinforcement Agents for Targeted Behaviors}},
  author = {Fengshuo Bai and Runze Liu and Yali Du and Ying Wen and Yaodong Yang},
  year = {2025},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {39},
  number = {15},
  pages = {15453--15461},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  doi = {10.1609/aaai.v39i15.33696},
  url = {https://doi.org/10.1609/aaai.v39i15.33696}
}
```

BibTeX download: https://yingwen.io/citations/rat-aaai-2025.bib
