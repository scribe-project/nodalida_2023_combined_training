# Code for the paper "Improving Generalization of Norwegian ASR with Limited Linguistic Resources"

This repository contains the code for the paper "Improving Generalization of Norwegian ASR with Limited Linguistic Resources", presented at NoDaLiDa 2023. To cite the paper, please use:
```
@InProceedings{SolbergEtAlNoDaLiDa2023,
  author = {Per Erik Solberg and Pablo Ortiz and Phoebe Parsons and Torbjørn Svendsen and Giampiero Salvi},	 
  title = {Improving Generalization of Norwegian ASR with Limited Linguistic Resources},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics},
  year = 	 {2023},
  month = 	 {May},
  address = 	 {Tórshavn, Faroe Islands},
}
```

* [analysis/](https://github.com/scribe-project/nodalida_2023_combined_training/tree/main/analysis) contains the analyses in the paper. [analysis.ipynb](https://github.com/scribe-project/nodalida_2023_combined_training/blob/main/analysis/analysis.ipynb) contains the analyses without a language model, and [analysis_w_lm.ipynb](https://github.com/scribe-project/nodalida_2023_combined_training/blob/main/analysis/analysis_w_lm.ipynb) contains the analyses with a language model.
* [make_datasets/](https://github.com/scribe-project/nodalida_2023_combined_training/tree/main/make_datasets) contains the code for making the different datasets used for testing and training. It also contains a [notebook](https://github.com/scribe-project/nodalida_2023_combined_training/blob/main/make_datasets/dataset_stats.ipynb) for retrieving stats about the different datasets.

Note also [this repository](https://github.com/scribe-project/asr-standardized-combined), which contains the code for standardizing the datasets used in this paper.
