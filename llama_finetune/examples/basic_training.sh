#!/bin/bash

# Basic Llama Fine-tuning Example
# This script demonstrates how to fine-tune Llama-2-7B with LoRA and 4-bit quantization

echo "=== Llama Fine-tuning Example ==="
echo "This script will:"
echo "1. Create a sample dataset"
echo "2. Fine-tune Llama-2-7B using LoRA + 4-bit quantization"
echo "3. Test the model with inference"
echo ""

# Create sample dataset
echo "Step 1: Creating sample dataset..."
python ../prepare_dataset.py --create_sample --output sample_dataset.json

if [ $? -ne 0 ]; then
    echo "Error: Failed to create sample dataset"
    exit 1
fi

echo "Sample dataset created: sample_dataset.json"
echo ""

# Fine-tune the model
echo "Step 2: Starting fine-tuning..."
echo "Note: This requires access to Llama-2 model on HuggingFace"
echo "Make sure you have accepted the license and have HF_TOKEN set"
echo ""

python ../train_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name sample_dataset.json \
    --output_dir ./llama-sample-finetuned \
    --use_lora \
    --use_4bit \
    --use_trainer \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "Training completed! Model saved to: ./llama-sample-finetuned"
echo ""

# Test inference
echo "Step 3: Testing inference..."
echo "Running a quick test with the fine-tuned model"
echo ""

python ../inference.py \
    --model_path ./llama-sample-finetuned \
    --prompt "Explain what machine learning is in simple terms" \
    --max_new_tokens 100

if [ $? -ne 0 ]; then
    echo "Error: Inference failed"
    exit 1
fi

echo ""
echo "=== Fine-tuning Complete! ==="
echo "You can now use the model with:"
echo "python ../inference.py --model_path ./llama-sample-finetuned --chat"
echo ""
echo "Or run inference on custom prompts:"
echo "python ../inference.py --model_path ./llama-sample-finetuned --prompt 'Your question here'"