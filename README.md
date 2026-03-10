# Density2R
This is the official implementation of [Density2R: Efficient Document Re-ranking via Embedding Density over Parametric Knowledge of Large Language Models](https://ieeexplore.ieee.org/document/11401587). (IEEE BigData 2025)

## ⭐ Citation
If you find this work helpful in your research, please consider citing our work.
```
@INPROCEEDINGS {11401587,
author = { Zaoad, Md Shahir and Zawad, Niamat and Khan, Latifur and Ranade, Priyanka and Krogman, Richard },
booktitle = { 2025 IEEE International Conference on Big Data (BigData) },
title = {{ Density2R: Efficient Document Re-Ranking via Embedding Density Over Parametric Knowledge of Large Language Models }},
year = {2025},
volume = {},
ISSN = {},
pages = {3142-3151},
abstract = { The advent of transformers has substantially advanced the task of document re-ranking within the field of Information Retrieval (IR). While pre-trained transformers have enabled the development of numerous supervised re-rankers, the unsupervised domain has largely been dominated by Large Language Model (LLM)-based approaches. However, despite their effectiveness, these models suffer from several critical limitations. Supervised re-rankers rely on large datasets of relevance judgments, an expensive and scarce resource, which further leads to poor generalizability. At the same time, unsupervised, zero-shot LLM-based methods suffer from limited context length, high token cost, and significant inference latency. To address these, we propose Density2R, a lightweight zero-shot re-ranker that leverages parametric knowledge from LLM to rank documents via density estimators. Unlike contemporary unsupervised methods, it only requires a handful of tokens to represent the query-dependent parametric information. The proposed approach further incorporates a selective re-ranking strategy that operates only on relevant candidates from the initial retriever, thereby reducing latency. Additionally, we extend the Density2R into a multi-stage pipeline, enabling it to boost the efficiency of re-rankers proposed in contemporary literature. Extensive evaluation on the TREC DL19, DL20, and the BEIR benchmark demonstrates the effectiveness of the proposed approach while maintaining a low-latency footprint. },
keywords = {Uncertainty;Costs;Filtering;Large language models;Computational modeling;Pipelines;Estimation;Transformers;Information retrieval;Low latency communication},
doi = {10.1109/BigData66926.2025.11401587},
url = {https://doi.ieeecomputersociety.org/10.1109/BigData66926.2025.11401587},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =Dec}
```
