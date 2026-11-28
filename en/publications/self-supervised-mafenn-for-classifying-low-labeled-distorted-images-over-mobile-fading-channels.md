# Self-Supervised MAFENN for Classifying Low-Labeled Distorted Images Over Mobile Fading Channels

Authors: Yang Li; Fanglei Sun; Jingchen Hu; Chang Liu; Fan Wu; Kai Li; Ying Wen; Zheng Tian; Yaodong Yang; Jiangcheng Zhu; Zhifeng Chen; Jun Wang; Yang Yang

Publication: IEEE Transactions on Mobile Computing (2024)
Type: Journal article
Canonical page: https://yingwen.io/en/publications/self-supervised-mafenn-for-classifying-low-labeled-distorted-images-over-mobile-fading-channels/
Publisher / primary source: https://doi.org/10.1109/tmc.2023.3343939
DOI: https://doi.org/10.1109/tmc.2023.3343939
Topics: [Multi-agent games & open-ended learning](https://yingwen.io/en/research/multi-agent/); [Machine learning & applications](https://yingwen.io/en/research/machine-learning/)

## Abstract

Image distortion during wireless transmission presents a significant challenge for real-world artificial intelligence (AI) applications. Recent methods have attempted to address this issue by integrating neural networks into the wireless transmission system. However, these approaches often require a large volume of labeled training data, which can be expensive and time-consuming to collect. To address this issue, we propose a novel approach, Self-Supervised Multi-Agent Feedback Enabled Neural Networks (S2MAFENN). S2MAFENN is designed to improve the efficiency of labeled data in wireless image transmission. It incorporates a Feedbacker agent that emulates the error correction mechanisms observed in primate brains and employs self-supervised contrastive learning to extract representations from unlabeled distorted images independently. From a theoretical perspective, we model the training process of S2MAFENN as a three-player Stackelberg game and provide evidence that S2MAFENN can achieve exponential convergence rates. We then empirically validate our approach by assessing the representations learned through S2MAFENN. We use varied labeled CIFAR10 and CIFAR100 data to simulate real image transmissions over the Rayleigh fading and 5G channels. Our results show that S2MAFENN matches or even surpasses the performance of state-of-the-art self-supervised training methods, even when only 50% of labels are used. Moreover, S2MAFENN yields average accuracy gains of 5.11%, 5.8%, and 4.58% with only 0.1, 0.2, and 0.5 of the labels transmitted over the 5G channel, respectively. For the downstream task of semantic segmentation over the 5G channel, S2MAFENN exhibits significant advancements on the ADE20K dataset. It achieves enhancements of approximately 7% and 8.7% in Mean IoU and DICE metrics, respectively, surpassing the performance of current state-of-the-art methods.

## Citation

```bibtex
@article{li2024-self-supervised-mafenn-for-classifying-low-la,
  title = {{Self-Supervised MAFENN for Classifying Low-Labeled Distorted Images Over Mobile Fading Channels}},
  author = {Yang Li and Fanglei Sun and Jingchen Hu and Chang Liu and Fan Wu and Kai Li and Ying Wen and Zheng Tian and Yaodong Yang and Jiangcheng Zhu and Zhifeng Chen and Jun Wang and Yang Yang},
  year = {2024},
  journal = {IEEE Transactions on Mobile Computing},
  volume = {23},
  number = {8},
  pages = {8077--8091},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  doi = {10.1109/tmc.2023.3343939},
  url = {https://doi.org/10.1109/tmc.2023.3343939}
}
```

BibTeX download: https://yingwen.io/citations/self-supervised-mafenn-for-classifying-low-labeled-distorted-images-over-mobile-fading-channels.bib
