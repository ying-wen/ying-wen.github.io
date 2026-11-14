# Cross-Utterance Conditioned VAE for Speech Generation

Authors: Yang Li; Cheng Yu; Guangzhi Sun; Weiqin Zu; Zheng Tian; Ying Wen; Wei Pan; Chao Zhang; Jun Wang; Yang Yang; Fanglei Sun

Publication: IEEE/ACM Transactions on Audio, Speech, and Language Processing (2024)
Type: Journal article
Canonical page: https://yingwen.io/en/publications/cross-utterance-conditioned-vae-for-speech-generation/
Publisher / primary source: https://doi.org/10.1109/taslp.2024.3453598
DOI: https://doi.org/10.1109/taslp.2024.3453598
arXiv: https://arxiv.org/abs/2309.04156
PDF: https://arxiv.org/pdf/2309.04156
Topics: [Machine learning & applications](https://yingwen.io/en/research/machine-learning/)

## Abstract

Speech synthesis systems powered by neural networks hold promise for multimedia production, but frequently face issues with producing expressive speech and seamless editing. In response, we present the Cross-Utterance Conditioned Variational Autoencoder speech synthesis (CUC-VAE S2) framework to enhance prosody and ensure natural speech generation. This framework leverages the powerful representational capabilities of pre-trained language models and the re-expression abilities of variational autoencoders (VAEs). The core component of the CUC-VAE S2 framework is the cross-utterance CVAE, which extracts acoustic, speaker, and textual features from surrounding sentences to generate context-sensitive prosodic features, more accurately emulating human prosody generation. We further propose two practical algorithms tailored for distinct speech synthesis applications: CUC-VAE TTS for text-to-speech and CUC-VAE SE for speech editing. The CUC-VAE TTS is a direct application of the framework, designed to generate audio with contextual prosody derived from surrounding texts. On the other hand, the CUC-VAE SE algorithm leverages real mel spectrogram sampling conditioned on contextual information, producing audio that closely mirrors real sound and thereby facilitating flexible speech editing based on text such as deletion, insertion, and replacement. Experimental results on the LibriTTS datasets demonstrate that our proposed models significantly enhance speech synthesis and editing, producing more natural and expressive speech.

## Citation

```bibtex
@article{li2024-cross-utterance-conditioned-vae-for-speech-ge,
  title = {{Cross-Utterance Conditioned VAE for Speech Generation}},
  author = {Yang Li and Cheng Yu and Guangzhi Sun and Weiqin Zu and Zheng Tian and Ying Wen and Wei Pan and Chao Zhang and Jun Wang and Yang Yang and Fanglei Sun},
  year = {2024},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume = {32},
  pages = {4263--4276},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  doi = {10.1109/taslp.2024.3453598},
  url = {https://doi.org/10.1109/taslp.2024.3453598}
}
```

BibTeX download: https://yingwen.io/citations/cross-utterance-conditioned-vae-for-speech-generation.bib
