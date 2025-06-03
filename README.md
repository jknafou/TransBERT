# TransBERT

This repository provides the **reproducible code** for all experiments presented in our paper, submitted to EMNLP 2025:
- 📄 **Paper**: [TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling (PDF)](https://transbert.s3.text-analytics.ch/TransBERT.pdf)

With this codebase, you can fully reproduce the results for:
- **Model**: [TransBERT-bio-fr on Hugging Face 🤗](https://huggingface.co/jknafou/TransBERT-bio-fr)

The model was trained on a large-scale synthetic biomedical corpus, generated using the [TransCorpus Toolkit](https://github.com/jknafou/TransCorpus) and is available here:
- **Corpus**: [TransCorpus-bio-fr on Hugging Face 🤗](https://huggingface.co/datasets/jknafou/TransCorpus-bio-fr)

## Contents
- ```DrBenchmark/``` – Scripts and configs for evaluating models on DrBenchmark, including cross-validation and hyperparameter optimization.
- ```LM_training/``` — Scripts used to pretrain TransBERT on [TransCorpus-bio-fr 🤗](https://huggingface.co/datasets/jknafou/TransCorpus-bio-fr)

## Citation
If you use this code, model, or corpus, please cite:
```text
@misc{knafou-transbert,
	author = {Knafou, Julien and Mottin, Luc and Ana\"{i}s, Mottaz and Alexandre, Flament and  Ruch, Patrick},
	title = {TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling},
	year = {2025},
	note = {Submitted to EMNLP2025. Anonymous ACL submission available:},
	url = {https://transbert.s3.text-analytics.ch/TransBERT.pdf},
}
```
