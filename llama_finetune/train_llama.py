#!/usr/bin/env python3
"""
Fine-tuning script for Llama models using PyTorch and Transformers.
Supports various fine-tuning strategies including full fine-tuning, LoRA, and QLoRA.
"""

import argparse
import json
import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from datasets import load_dataset, Dataset as HFDataset
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import wandb
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """Custom dataset for instruction-following fine-tuning."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the instruction-response pair
        if 'instruction' in item and 'response' in item:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
        elif 'input' in item and 'output' in item:
            prompt = f"### Input:\n{item['input']}\n\n### Output:\n{item['output']}"
        else:
            # Fallback to text field
            prompt = item.get('text', str(item))
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def setup_model_and_tokenizer(model_name, use_4bit=False, use_8bit=False, use_lora=False, lora_config=None):
    """Setup model and tokenizer with optional quantization and LoRA."""
    
    # Configure quantization
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_4bit or use_8bit else torch.float16
    )
    
    # Prepare model for training
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA if requested
    if use_lora:
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def load_and_prepare_dataset(dataset_name, tokenizer, max_length=512, test_size=0.1):
    """Load and prepare dataset for training."""
    
    if dataset_name.endswith('.json'):
        # Load local JSON file
        with open(dataset_name, 'r') as f:
            data = json.load(f)
        dataset = HFDataset.from_list(data)
    else:
        # Load from HuggingFace Hub
        dataset = load_dataset(dataset_name)
        if isinstance(dataset, dict):
            dataset = dataset['train']
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    # Create custom datasets
    train_dataset = InstructionDataset(dataset['train'], tokenizer, max_length)
    eval_dataset = InstructionDataset(dataset['test'], tokenizer, max_length)
    
    return train_dataset, eval_dataset


def train_model(model, train_dataset, eval_dataset, output_dir, args):
    """Train the model using custom training loop or HF Trainer."""
    
    if args.use_trainer:
        # Use HuggingFace Trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=0.1,
            learning_rate=args.learning_rate,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb" if args.use_wandb else None,
            run_name=f"llama-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            dataloader_pin_memory=False,
            bf16=True,
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=train_dataset.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        trainer.save_model()
        
    else:
        # Custom training loop
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        
        num_training_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        # Training loop
        model.train()
        global_step = 0
        
        for epoch in range(args.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if args.use_wandb:
                        wandb.log({
                            "train_loss": loss.item() * args.gradient_accumulation_steps,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "global_step": global_step
                        })
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Evaluation
                if global_step % 100 == 0:
                    eval_loss = evaluate_model(model, eval_dataloader)
                    logger.info(f"Step {global_step}: Eval loss = {eval_loss:.4f}")
                    
                    if args.use_wandb:
                        wandb.log({"eval_loss": eval_loss, "global_step": global_step})
                    
                    model.train()
                
                # Save checkpoint
                if global_step % 500 == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(checkpoint_dir)
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        model.save_pretrained(output_dir)


def evaluate_model(model, eval_dataloader):
    """Evaluate the model on validation set."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch['input_ids'].size(0)
            total_samples += batch['input_ids'].size(0)
    
    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama model")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name or path (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Dataset name from HF Hub or path to local JSON file")
    parser.add_argument("--output_dir", type=str, default="./llama-finetuned",
                       help="Output directory for the fine-tuned model")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Optimization arguments
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Training configuration
    parser.add_argument("--use_trainer", action="store_true",
                       help="Use HuggingFace Trainer instead of custom training loop")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="llama-finetune",
                       help="W&B project name")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"llama-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup LoRA config if using LoRA
    lora_config = None
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
    
    # Setup model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        use_lora=args.use_lora,
        lora_config=lora_config
    )
    
    # Load and prepare dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        args.dataset_name, 
        tokenizer, 
        args.max_length
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Start training
    logger.info("Starting training...")
    train_model(model, train_dataset, eval_dataset, args.output_dir, args)
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()