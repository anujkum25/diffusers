#!/bin/bash

# Advanced Llama Fine-tuning Example
# This script demonstrates advanced features like custom datasets and W&B logging

echo "=== Advanced Llama Fine-tuning Example ==="
echo ""

# Check if HuggingFace token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set"
    echo "You may need to login to HuggingFace: huggingface-cli login"
    echo ""
fi

# Option 1: Use HuggingFace dataset
echo "Option 1: Training with Alpaca dataset from HuggingFace"
echo ""

python ../train_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./llama-alpaca-lora \
    --use_lora \
    --use_4bit \
    --use_trainer \
    --use_wandb \
    --wandb_project "llama-alpaca-finetune" \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1

echo ""
echo "Training with Alpaca dataset completed!"
echo ""

# Option 2: Full fine-tuning (requires high-memory GPU)
echo "Option 2: Full fine-tuning (commented out - requires ~80GB GPU memory)"
echo ""

# Uncomment the following for full fine-tuning:
# python ../train_llama.py \
#     --model_name meta-llama/Llama-2-7b-hf \
#     --dataset_name tatsu-lab/alpaca \
#     --output_dir ./llama-alpaca-full \
#     --use_trainer \
#     --num_epochs 1 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1e-5 \
#     --max_length 512

# Option 3: Code fine-tuning with CodeLlama
echo "Option 3: Code fine-tuning with CodeLlama"
echo ""

# Create a code-specific dataset (you can replace this with your own)
python ../prepare_dataset.py --create_sample --output code_dataset.json

# Replace sample with actual code dataset if available
python ../train_llama.py \
    --model_name codellama/CodeLlama-7b-hf \
    --dataset_name code_dataset.json \
    --output_dir ./codellama-finetuned \
    --use_lora \
    --use_4bit \
    --use_trainer \
    --num_epochs 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32

echo ""
echo "CodeLlama fine-tuning completed!"
echo ""

# Test all models
echo "Testing fine-tuned models..."
echo ""

models=("./llama-alpaca-lora" "./codellama-finetuned")
prompts=("Explain the importance of data preprocessing in machine learning" "Write a Python function to implement quicksort")

for i in "${!models[@]}"; do
    model=${models[$i]}
    prompt=${prompts[$i]}
    
    if [ -d "$model" ]; then
        echo "Testing model: $model"
        echo "Prompt: $prompt"
        echo ""
        
        python ../inference.py \
            --model_path "$model" \
            --prompt "$prompt" \
            --max_new_tokens 150 \
            --temperature 0.7
        
        echo ""
        echo "---"
        echo ""
    fi
done

echo "=== Advanced Training Complete! ==="
echo ""
echo "Available models:"
for model in ./llama-* ./codellama-*; do
    if [ -d "$model" ]; then
        echo "  - $model"
    fi
done
echo ""
echo "Start interactive chat with any model:"
echo "python ../inference.py --model_path ./llama-alpaca-lora --chat"