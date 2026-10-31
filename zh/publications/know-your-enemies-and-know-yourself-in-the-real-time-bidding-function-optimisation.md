# Know Your Enemies and Know Yourself in the Real-Time Bidding Function Optimisation

Authors: Manxing Du; Alexander I. Cowen-Rivers; Ying Wen; Phu Sakulwongtana; Jun Wang; Mats Brorsson; Radu State

Publication: ICDM Workshops (2019)
Type: Conference paper
Canonical page: https://yingwen.io/zh/publications/know-your-enemies-and-know-yourself-in-the-real-time-bidding-function-optimisation/
Publisher / primary source: https://doi.org/10.1109/icdmw.2019.00141
DOI: https://doi.org/10.1109/icdmw.2019.00141
Topics: [强化学习](https://yingwen.io/zh/research/reinforcement-learning/); [多智能体博弈与开放式学习](https://yingwen.io/zh/research/multi-agent/); [机器学习与应用](https://yingwen.io/zh/research/machine-learning/)

## Abstract

Real-time bidding (RTB) is a popular method to sell online ad space inventory using real-time auctions to determine which advertiser gets to make the ad impression. Advertisers can take user information into account when making their bids and get more control over the process. The goal of an optimal bidding function is to maximise the overall effectiveness of the ad campaigns defined by the advertisers under a certain budget constraint. A straightforward solution would be to model the bidding function in an explicit form. However, such functional solutions lack generality in practice and are insensitive to the stochastic behaviour of other bidders in the environment. In this paper, we propose to formulate the online auctions into a general mean field multi-agent framework, in which the agents compete with each other and each agent’s best response strategy depends on its opponents’ actions. We firstly introduce a novel Deep Attentive Survival Analysis (DASA) model to estimate the opponent’s action distribution on the ad impression level which outperforms state-of-the-art survival analysis. Furthermore, we introduce the DASA model as the opponent model into the Mean Field Deep Deterministic Policy Gradients (DDPG) algorithm for each agent to learn the optimal bidding strategy and converge to the mean field equilibrium. The experiments have shown that with the inference of the market, the market converges to the equilibrium faster while playing against both fixed strategy agents and dynamic learning agents.

## Citation

```bibtex
@inproceedings{du2019-know-your-enemies-and-know-yourself-in-the-re,
  title = {{Know Your Enemies and Know Yourself in the Real-Time Bidding Function Optimisation}},
  author = {Manxing Du and Alexander I. Cowen-Rivers and Ying Wen and Phu Sakulwongtana and Jun Wang and Mats Brorsson and Radu State},
  year = {2019},
  booktitle = {2019 International Conference on Data Mining Workshops (ICDMW)},
  pages = {973--981},
  publisher = {IEEE},
  doi = {10.1109/icdmw.2019.00141},
  url = {https://doi.org/10.1109/icdmw.2019.00141}
}
```

BibTeX download: https://yingwen.io/citations/know-your-enemies-and-know-yourself-in-the-real-time-bidding-function-optimisation.bib
