# Utilizing Large Language Models for Geoscience Literature Information Extraction

Authors: Peng Yu; Cheng Deng; Huawei Ji; Ying Wen

Publication: EGU General Assembly 2024 (abstract) (2024)
Type: Conference abstract
Canonical page: https://yingwen.io/en/publications/utilizing-large-language-models-for-geoscience-literature-information-extraction/
Publisher / primary source: https://meetingorganizer.copernicus.org/EGU24/EGU24-16101.html
DOI: https://doi.org/10.5194/egusphere-egu24-16101
Topics: [Large language models & reasoning](https://yingwen.io/en/research/llms/); [Machine learning & applications](https://yingwen.io/en/research/machine-learning/)

## Abstract

Extracting information from unstructured and semi-structured geoscience literature is a crucial step in conducting geological research. The traditional machine learning extraction paradigm requires a substantial amount of high-quality manually annotated data for model training, which is time-consuming, labor-intensive, and not easily transferable to new fields. Recently, large language models (LLMs) (e.g., ChatGPT, GPT-4, and LLaMA), have shown great performance in various natural language processing (NLP) tasks, such as question answering, machine translation, and text generation. A substantial body of work has demonstrated that LLMs possess strong in-context learning (ICL) and even zero-shot learning capabilities to solve downstream tasks without specifically designed supervised fine-tuning. In this paper, we propose utilizing LLMs for geoscience literature information extraction. Specifically, we design a hierarchical PDF parsing pipeline and an automated knowledge extraction process, which can significantly reduce the need for manual data annotation, assisting geoscientists in literature data mining. For the hierarchical PDF parsing pipeline, firstly, a document layout detection model fine-tuned on geoscience literature is employed for layout detection, obtaining layout detection information for the document. Secondly, based on the document layout information, an optical character content parsing model is used for content parsing, obtaining the text structure and plain text corresponding to the content. Finally, the text structure and plain text are combined and reconstructed to ultimately obtain the parsed structured data. For the automated knowledge extraction process, firstly, the parsed long text is segmented into paragraphs to adapt to the input length limit of LLMs. Subsequently, a few-shot prompting method is employed for structured knowledge extraction, encompassing two tasks: attribute value extraction and triplet extraction. In attribute value extraction, prompts are generated automatically by the LLMs based on the subdomain and attribute names, facilitating the location and extraction of values related to subdomain attribute names in the text. For triplet extraction, the LLMs employ a procedural approach to entity extraction, entity type extraction, and relation extraction, following the knowledge graph structure pattern. Finally, the extracted structured knowledge is stored in the form of knowledge graphs, facilitating further analysis and integration of various types of knowledge from the literature. Our proposed approach turns out to be simple, flexible, and highly effective in geoscience literature information extraction. Demonstrations of information extraction in subdomains such as radiolarian fossils and fluvial facies have yielded satisfactory results. The extraction efficiency has significantly improved, and feedback from domain experts indicates a relatively high level of accuracy in the extraction process. The extracted results can be used to construct a foundational knowledge graph for geoscience literature, supporting the comprehensive construction and efficient application of a geoscience knowledge graph.

## Citation

```bibtex
@inproceedings{yu2024-utilizing-large-language-models-for-geoscienc,
  title = {{Utilizing Large Language Models for Geoscience Literature Information Extraction}},
  author = {Peng Yu and Cheng Deng and Huawei Ji and Ying Wen},
  year = {2024},
  booktitle = {EGU General Assembly 2024},
  publisher = {Copernicus Meetings},
  doi = {10.5194/egusphere-egu24-16101},
  url = {https://meetingorganizer.copernicus.org/EGU24/EGU24-16101.html},
  note = {Conference abstract}
}
```

BibTeX download: https://yingwen.io/citations/utilizing-large-language-models-for-geoscience-literature-information-extraction.bib
