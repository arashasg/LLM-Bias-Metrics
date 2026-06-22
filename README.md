# LLM Bias Metrics

Brief teaching repo for applying bias metrics to one model, Phi-4, on the StereoSet intrasentence dataset. The notebook uses saved model outputs so the metrics can be inspected without rerunning generation.

## Main Notebook

Open `stereoset_phi4_bias_metrics.ipynb` and run the cells in order. It covers:

- loading StereoSet triplets from `dataset/stereoset_dev.json`
- loading saved Phi-4 perplexity scores from `predictions/stereoset/phi-4_perplexity_scores.json`
- computing CAT, language modeling bias, LMS, and ICAT
- loading saved toxicity scores from `predictions/toxicity/phi-4_toxicity_scores.json`
- computing expected maximum toxicity, toxicity fraction, and empirical probability of biased generation
- loading or recomputing the HONEST hurtful-completion score from `predictions/honest/phi-4_honest_score.json`

## Files

- `stereoset_phi4_bias_metrics.ipynb`: end-to-end metric walkthrough
- `toxicity_metrics.py`: helper functions for toxicity-based metrics
- `honest_metrics.py`: helper functions for HONEST scoring
- `dataset/`: StereoSet development data
- `predictions/`: saved Phi-4 perplexity, toxicity, and HONEST outputs

## Notes

The notebook includes an optional Phi-4 scoring section, but it is disabled by default. Use the saved predictions unless you have a GPU and the required model dependencies installed.

## Citation

This teaching repo is based on:

```bibtex
@inproceedings{asgari2026quantifying,
  title     = {Quantifying Metric and Model Agreement in Bias Evaluation of Large Language Models},
  author    = {Asgari, Arash and Wu, Huan and Naziri, Amirreza and Kolahdouzi, Mojtaba and Seyyed-Kalantari, Laleh},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026},
  address   = {San Diego, California, USA},
  publisher = {Association for Computational Linguistics},
  note      = {Long paper. ACL ARR 2026 January Submission, Submission Number 7014}
}
```
