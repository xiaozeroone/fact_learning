
# Co-occurrence is not Factual Association in Language Models

This repo provides code for finetuning language models to learn novel factual knowledge from text, and evaluating the learned knowledge on various reasoning tasks. The code is used by [Co-occurrence is not Factual Association in Language Models](https://openreview.net/pdf?id=xabStWAUtr).

The synthetic fact learning dataset used in the paper (Country-city-animals) is released on the [Huggingface hub](https://huggingface.co/datasets/xiaozeroone/Country-city-animals).


## Training
Finetune llama-3 8B on the *Narrative* corpus of Country-city-animals (on one 80GB GPU)
```
python train.py \
    --config-name finetune_full.yaml \
    model=meta-llama/Meta-Llama-3-8B \
    subset=Corpus_narrative \
    save=save/llama3_8b_narrative
```
replace `meta-llama/Meta-Llama-3-8B` with the model to finetune and `save/llama3_8b_narrative` with the path to save the finetuned model. Replace `Corpus_narrative` with the name of the corpus to finetune on. See [here](https://huggingface.co/datasets/xiaozeroone/Country-city-animals) for all the available corpora in the Country-city-animals dataset.

Other training options, including hyperparameters, are specified in `config/finetune_full.yaml`. Use `finetune_lora.yaml` for training with low-rank adaptation (LoRA).

### Training with active forgetting
To further finetune a model (trained with LoRA in the last step) with active forgetting, run
```
python train.py \
    --config-name finetune_lora.yaml \
    +forget=True \
    +forget_start_layer=10 \
    +model_tuned=save/llama3_8b_narrative \
    model=meta-llama/Meta-Llama-3-8B \
    subset=Corpus_narrative \
    save=save/llama3_8b_narrative_forget
```
replace `save/llama3_8b_narrative` with the path to the model finetuned in the last step and set `forget_start_layer` to the layer where all layers above it will be forgotten st the start of finetuning.


## Evaluation
Evaluate finetuned model on the *QA* task
```
python lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=save/llama3_8b_narrative,use_accelerate=True \
    --no_cache \
    --tasks Eval_QA \
    --num_fewshot 5 \
    --batch_size 4
```
replace `save/llama3_8b_narrative` with the path to the model to evaluate and replace `Eval_QA` with the name of the task to evaluate on. See [here](https://huggingface.co/datasets/xiaozeroone/Country-city-animals) for all the eval tasks in the Country-city-animals dataset.


## Citation
```
@inproceedings{
  zhang2024cooccurrence,
  title={Co-occurrence is not Factual Association in Language Models},
  author={Xiao Zhang and Miao Li and Ji Wu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=xabStWAUtr}
}
```