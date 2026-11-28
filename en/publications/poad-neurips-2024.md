# Reinforcing LLM Agents via Policy Optimization with Action Decomposition

Authors: Muning Wen; Ziyu Wan; Jun Wang; Weinan Zhang; Ying Wen

Publication: NeurIPS (2024)
Type: Conference paper
Canonical page: https://yingwen.io/en/publications/poad-neurips-2024/
Publisher / primary source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc09efb501c801ed92e181e26a885c2d-Abstract-Conference.html
DOI: https://doi.org/10.52202/079017-3297
arXiv: https://arxiv.org/abs/2405.15821
PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/bc09efb501c801ed92e181e26a885c2d-Paper-Conference.pdf
Topics: [LLM agents](https://yingwen.io/en/research/llm-agents/); [Reinforcement learning](https://yingwen.io/en/research/reinforcement-learning/)

## Abstract

Language models as intelligent agents push the boundaries of sequential decision-making agents but struggle with limited knowledge of environmental dynamics and exponentially huge action space. Recent efforts like GLAM and TWOSOME manually constrain the action space to a restricted subset and employ reinforcement learning to align agents' knowledge with specific environments. However, they overlook fine-grained credit assignments for intra-action tokens, which is essential for efficient language agent optimization, and rely on human's prior knowledge to restrict action space. This paper proposes decomposing language agent optimization from the action level to the token level, offering finer supervision for each intra-action token and manageable optimization complexity in environments with unrestricted action spaces. Beginning with the simplification of flattening all actions, we theoretically explore the discrepancies between action-level optimization and this naive token-level optimization. We then derive the Bellman backup with Action Decomposition (BAD) to integrate credit assignments for both intra-action and inter-action tokens, effectively eliminating the discrepancies. Implementing BAD within the PPO algorithm, we introduce Policy Optimization with Action Decomposition (POAD). POAD benefits from a finer-grained credit assignment process and lower optimization complexity, leading to enhanced learning efficiency and generalization abilities in aligning language agents with interactive environments. We validate POAD across diverse testbeds, with results affirming the advantages of our approach and the correctness of our theoretical analysis. The source code can be accessed directly with this link: https://github.com/morning9393/ADRL.

## Citation

```bibtex
@inproceedings{NEURIPS2024_bc09efb5,
  title = {{Reinforcing LLM Agents via Policy Optimization with Action Decomposition}},
  author = {Muning Wen and Ziyu Wan and Jun Wang and Weinan Zhang and Ying Wen},
  year = {2024},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  pages = {103774--103805},
  publisher = {Curran Associates, Inc.},
  doi = {10.52202/079017-3297},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc09efb501c801ed92e181e26a885c2d-Abstract-Conference.html}
}
```

BibTeX download: https://yingwen.io/citations/poad-neurips-2024.bib
