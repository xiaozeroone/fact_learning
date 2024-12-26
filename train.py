"""A simple script to finetune a language model."""

import hydra
import transformers
import torch
import re
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from peft.peft_model import load_peft_weights, set_peft_model_state_dict
from accelerate import Accelerator
from datasets import load_dataset


@hydra.main(version_base=None, config_path="config")
def main(config: DictConfig) -> None:

    # Set seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Prepare model
    model = AutoModelForCausalLM.from_pretrained(
        config.model, 
        use_cache=False,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    if 'lora' in config:
        model = get_peft_model(
            model,
            LoraConfig(**OmegaConf.to_object(config.lora))
        )
        model.enable_input_require_grads()
    
    if 'forget' in config and config.forget:
        # Only load the trained LoRA adapter weights of layers below `forget_start_layer`. 
        # This is equivalent to resetting the layers above `forget_start_layer` to pretrained value
        adapters_weights = load_peft_weights(config.model_tuned, device='cuda')
        new_adapters_weights = {}
        for key in adapters_weights:
            layer_index = int(re.match(r".*\.[^.]*\.(\d+)\.", key).group(1))
            if layer_index < config.forget_start_layer:
                new_adapters_weights[key] = adapters_weights[key]
        set_peft_model_state_dict(model, new_adapters_weights, adapter_name='default')


    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token            # patch llama tokenizer

    def tokenize_func(eg):
        result = tokenizer(
            eg["text"],
            truncation=True,
            max_length=config.max_token_length,
            padding=False,
            return_tensors=None,
        )
        return result
                                              
    with Accelerator().main_process_first():
        dataset = load_dataset(config.dataset, config.subset)['train']
        dataset = dataset.map(tokenize_func)

    # Prepare trainer
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.accumulate_grad_batches,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.max_epochs,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=config.lr,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler,
            max_grad_norm=config.gradient_clip_val,
            bf16=True,                
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            output_dir='tmp',
            seed=config.seed,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8,
        ),
    )

    # Perform finetuning
    trainer.train_dataset = dataset
    trainer.train()
    
    if Accelerator().is_main_process:
        trainer.save_model(config.save)
        if 'lora' not in config:
            tokenizer.save_pretrained(config.save)     # also save tokenizer for convenience of loading


if __name__ == "__main__":
    main()
