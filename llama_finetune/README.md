# Llama Fine-tuning with PyTorch

This repository contains a comprehensive solution for fine-tuning Llama models using PyTorch, Transformers, and PEFT (Parameter-Efficient Fine-Tuning).

## Features

- 🚀 **Multiple Training Strategies**: Full fine-tuning, LoRA, and QLoRA
- 📊 **Flexible Data Loading**: Support for HuggingFace datasets and custom JSON files
- 🔧 **Memory Optimization**: 4-bit and 8-bit quantization support
- 📈 **Training Monitoring**: Weights & Biases integration
- 🎯 **Easy Inference**: Simple inference script for testing models
- 🛠️ **Dataset Preparation**: Utilities for preparing custom datasets

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Install Flash Attention for better performance:

```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Prepare Your Dataset

Create a sample dataset:

```bash
python prepare_dataset.py --create_sample --output sample_dataset.json
```

Or convert your existing dataset:

```bash
python prepare_dataset.py --input your_data.json --output prepared_dataset.json --format alpaca
```

### 2. Fine-tune the Model

Basic fine-tuning with LoRA (recommended for most use cases):

```bash
python train_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name sample_dataset.json \
    --output_dir ./llama-finetuned \
    --use_lora \
    --use_4bit \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### 3. Test the Fine-tuned Model

Interactive chat:

```bash
python inference.py --model_path ./llama-finetuned --chat
```

Single prompt:

```bash
python inference.py \
    --model_path ./llama-finetuned \
    --prompt "Explain quantum computing in simple terms"
```

## Training Options

### Model Selection

Supported Llama variants:
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-13b-hf`
- `meta-llama/Llama-2-70b-hf`
- `codellama/CodeLlama-7b-hf`
- `codellama/CodeLlama-13b-hf`

### Training Strategies

#### 1. Full Fine-tuning
```bash
python train_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name your_dataset.json \
    --output_dir ./llama-full-ft \
    --num_epochs 1 \
    --batch_size 1 \
    --learning_rate 1e-5
```

#### 2. LoRA Fine-tuning (Recommended)
```bash
python train_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name your_dataset.json \
    --output_dir ./llama-lora \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

#### 3. QLoRA (4-bit quantization + LoRA)
```bash
python train_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name your_dataset.json \
    --output_dir ./llama-qlora \
    --use_lora \
    --use_4bit \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-4
```

### Memory Requirements

| Strategy | Model Size | GPU Memory | Training Time |
|----------|------------|------------|---------------|
| Full FT  | 7B         | ~80GB      | Longest       |
| LoRA     | 7B         | ~40GB      | Medium        |
| QLoRA    | 7B         | ~16GB      | Fastest       |

## Dataset Format

The training script expects datasets in instruction-response format:

```json
[
  {
    "instruction": "Explain the concept of machine learning.",
    "response": "Machine learning is a subset of artificial intelligence..."
  },
  {
    "instruction": "Write a Python function to calculate factorial.",
    "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
  }
]
```

### Supported Input Formats

The `prepare_dataset.py` script can convert from various formats:

1. **Alpaca format**:
```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

2. **ShareGPT format**:
```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

3. **Code format**:
```json
{
  "problem": "...",
  "solution": "..."
}
```

## Configuration

You can use the provided `config.yaml` template for easier configuration management:

```yaml
# Edit config.yaml with your settings
python train_llama.py --config config.yaml
```

## Monitoring Training

Enable Weights & Biases logging:

```bash
python train_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name your_dataset.json \
    --use_wandb \
    --wandb_project my-llama-project \
    # ... other args
```

## Advanced Usage

### Custom Target Modules for LoRA

For different model architectures, you might need to adjust the target modules:

```bash
# For Llama models (default)
--lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

# For other models, check the model architecture
```

### Gradient Accumulation

For effective larger batch sizes with limited GPU memory:

```bash
python train_llama.py \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    # Effective batch size = 2 * 8 = 16
```

### Mixed Precision Training

The script automatically uses BF16 for better performance on modern GPUs:

```python
# Automatically enabled for quantized models
# Uses torch.bfloat16 for 4-bit/8-bit models
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use 4-bit quantization
   - Use LoRA instead of full fine-tuning

2. **Slow Training**:
   - Enable mixed precision (BF16)
   - Increase batch size if memory allows
   - Use multiple GPUs with `accelerate`

3. **Poor Model Performance**:
   - Check dataset quality and format
   - Adjust learning rate (try 1e-4 to 5e-4 for LoRA)
   - Increase training epochs
   - Use warmup steps

### Performance Tips

1. **Use QLoRA for 7B+ models on consumer GPUs**
2. **Set appropriate learning rates**:
   - Full fine-tuning: 1e-5 to 5e-5
   - LoRA: 1e-4 to 5e-4
3. **Monitor training loss and adjust hyperparameters**
4. **Use gradient checkpointing for larger models**

## Examples

### Fine-tuning for Code Generation

```bash
python train_llama.py \
    --model_name codellama/CodeLlama-7b-hf \
    --dataset_name code_dataset.json \
    --output_dir ./codellama-finetuned \
    --use_lora \
    --use_4bit \
    --max_length 1024 \
    --num_epochs 2 \
    --learning_rate 2e-4
```

### Fine-tuning for Instruction Following

```bash
python train_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./llama-instruct \
    --use_lora \
    --use_4bit \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the same license as the underlying models and libraries used.

## Acknowledgments

- HuggingFace Transformers team
- PEFT library developers
- Meta AI for Llama models
- The open-source ML community