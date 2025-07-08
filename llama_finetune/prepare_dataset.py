#!/usr/bin/env python3
"""
Dataset preparation utility for Llama fine-tuning.
Converts various dataset formats to the standard instruction-response format.
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_alpaca_format(data: List[Dict]) -> List[Dict]:
    """Convert Alpaca-style dataset to standard format."""
    converted = []
    for item in data:
        if 'input' in item and item['input'].strip():
            # Instruction with input
            instruction = f"{item['instruction']}\n\nInput: {item['input']}"
        else:
            # Instruction only
            instruction = item['instruction']
        
        converted.append({
            'instruction': instruction,
            'response': item['output']
        })
    return converted


def convert_conversation_format(data: List[Dict]) -> List[Dict]:
    """Convert conversation-style dataset to instruction-response format."""
    converted = []
    for item in data:
        if 'conversations' in item:
            conversations = item['conversations']
            for i in range(0, len(conversations), 2):
                if i + 1 < len(conversations):
                    human_msg = conversations[i].get('value', '')
                    assistant_msg = conversations[i + 1].get('value', '')
                    
                    converted.append({
                        'instruction': human_msg,
                        'response': assistant_msg
                    })
    return converted


def convert_sharegpt_format(data: List[Dict]) -> List[Dict]:
    """Convert ShareGPT-style dataset to instruction-response format."""
    converted = []
    for item in data:
        if 'conversations' in item:
            conversations = item['conversations']
            current_instruction = ""
            
            for conv in conversations:
                if conv.get('from') == 'human':
                    current_instruction = conv.get('value', '')
                elif conv.get('from') == 'gpt' and current_instruction:
                    converted.append({
                        'instruction': current_instruction,
                        'response': conv.get('value', '')
                    })
                    current_instruction = ""
    return converted


def convert_code_format(data: List[Dict]) -> List[Dict]:
    """Convert code-specific dataset to instruction-response format."""
    converted = []
    for item in data:
        if 'problem' in item and 'solution' in item:
            converted.append({
                'instruction': f"Write a Python function to solve the following problem:\n\n{item['problem']}",
                'response': item['solution']
            })
        elif 'prompt' in item and 'completion' in item:
            converted.append({
                'instruction': item['prompt'],
                'response': item['completion']
            })
    return converted


def detect_format(data: List[Dict]) -> str:
    """Detect the format of the input dataset."""
    if not data:
        return 'unknown'
    
    sample = data[0]
    
    # Check for Alpaca format
    if 'instruction' in sample and 'output' in sample:
        return 'alpaca'
    
    # Check for conversation format
    if 'conversations' in sample:
        conversations = sample['conversations']
        if conversations and 'from' in conversations[0]:
            return 'sharegpt'
        else:
            return 'conversation'
    
    # Check for code format
    if 'problem' in sample and 'solution' in sample:
        return 'code'
    
    if 'prompt' in sample and 'completion' in sample:
        return 'code'
    
    # Check if already in instruction-response format
    if 'instruction' in sample and 'response' in sample:
        return 'standard'
    
    return 'unknown'


def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.suffix == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return data


def save_dataset(data: List[Dict], output_path: str):
    """Save dataset to JSON format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_sample_dataset() -> List[Dict]:
    """Create a sample dataset for testing."""
    return [
        {
            "instruction": "Explain the concept of machine learning in simple terms.",
            "response": "Machine learning is a type of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, without being explicitly programmed for each specific task."
        },
        {
            "instruction": "Write a Python function to calculate the factorial of a number.",
            "response": "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n```"
        },
        {
            "instruction": "What are the benefits of using renewable energy?",
            "response": "Renewable energy offers several benefits: 1) Environmental protection by reducing greenhouse gas emissions, 2) Energy independence and security, 3) Economic benefits through job creation, 4) Long-term cost savings, and 5) Sustainable resource utilization."
        },
        {
            "instruction": "How do you implement a binary search algorithm?",
            "response": "Binary search works by repeatedly dividing a sorted array in half and comparing the target value with the middle element. Here's a Python implementation:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1  # Target not found\n```"
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Llama fine-tuning")
    parser.add_argument("--input", type=str, help="Input dataset file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--format", type=str, choices=['alpaca', 'conversation', 'sharegpt', 'code', 'auto'],
                       default='auto', help="Input dataset format")
    parser.add_argument("--create_sample", action="store_true", help="Create a sample dataset for testing")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to include")
    
    args = parser.parse_args()
    
    if args.create_sample:
        logger.info("Creating sample dataset...")
        data = create_sample_dataset()
    else:
        if not args.input:
            raise ValueError("--input is required when not creating a sample dataset")
        
        logger.info(f"Loading dataset from {args.input}")
        data = load_dataset(args.input)
        
        # Detect format if auto
        if args.format == 'auto':
            args.format = detect_format(data)
            logger.info(f"Detected format: {args.format}")
        
        # Convert to standard format
        if args.format == 'alpaca':
            data = convert_alpaca_format(data)
        elif args.format == 'conversation':
            data = convert_conversation_format(data)
        elif args.format == 'sharegpt':
            data = convert_sharegpt_format(data)
        elif args.format == 'code':
            data = convert_code_format(data)
        elif args.format == 'standard':
            pass  # Already in correct format
        else:
            logger.warning(f"Unknown format: {args.format}. Assuming standard format.")
    
    # Limit samples if requested
    if args.max_samples and len(data) > args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")
    
    # Validate data
    valid_data = []
    for i, item in enumerate(data):
        if 'instruction' in item and 'response' in item:
            if item['instruction'].strip() and item['response'].strip():
                valid_data.append(item)
        else:
            logger.warning(f"Skipping invalid item at index {i}: {item}")
    
    logger.info(f"Prepared {len(valid_data)} valid samples")
    
    # Save dataset
    save_dataset(valid_data, args.output)
    logger.info(f"Dataset saved to {args.output}")
    
    # Print sample
    if valid_data:
        logger.info("Sample entry:")
        sample = valid_data[0]
        logger.info(f"Instruction: {sample['instruction'][:100]}...")
        logger.info(f"Response: {sample['response'][:100]}...")


if __name__ == "__main__":
    main()