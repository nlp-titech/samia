# SaMIA: Sampling-based Pseudo-Likelihood for Membership Inference Attacks

This repository contains the source code for [SaMIA: Sampling-based Pseudo-Likelihood for Membership Inference Attacks](https://arxiv.org/abs/2404.11262).

## Membership Inference Attack (MIA) with SaMIA

### Step 1: Generate candidate texts (samples) from LLMs
[WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA), a dataset for evaluating MIA, has been placed under the directory `wikimia/`.

Samples can be generated using the following command:
```
python src/sampling.py --model_name gpt-j-6B --text_length 32 --num_samples 10 --prefix_ratio 0.5
```
where
- `model_name` specifies the model used for sampling, chosen from {gpt-j-6B, opt-6.7b, pythia-6.9b, Llama-2-7b`};
- `text_length` specifies the length of texts, chosen from {32, 64, 128, 256};
- `num_samples` specifies the number of samples to be generated;
- `prefix_ratio` specifies the percentage of texts used as the prefix (prompt for generating the following text).

The generated samples will be collected in `sample/{model_name}/{text_length}.jsonl`.

!Note that in order to use [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf), be sure to obtain the permission for access and assign your huggingface token to variable `YOUR_HUGGINGFACE_TOKEN`.


### Step 2: Evaluate SaMIA on LLMs using the generated samples

We evaluate the leakage detection performance of SaMIA based on the surface similarity between generated samples, under the directory `sample/`, and the original texts, under the directory `wikimia/`.
The surface similarity is measured using ROUGE-N (N=1).
Evaluation can be conducted using the following command:
```
python src/eval_samia.py  --model_name gpt-j-6B --text_length 32 --num_samples 10 --prefix_ratio 0.5
```
Argument `model_name`, `text_length`, `num_samples`, and `prefix_ratio` functions similarly as in the previous section, with two additional flags available here:
- `--zlib` enables zlib entropy compression;
- `--save` dumps the computed rouge-n results of leaked (seen) and unleaked (unseen) texts into files.

The source code will print `AUC-ROC` and `TPR@10%FPR` at stdout.

## Citation

If you find this repository useful for your research, please cite us with:
```
@misc{kaneko2024samplingbased,
      title={Sampling-based Pseudo-Likelihood for Membership Inference Attacks}, 
      author={Masahiro Kaneko and Youmi Ma and Yuki Wata and Naoaki Okazaki},
      year={2024},
      eprint={2404.11262},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
