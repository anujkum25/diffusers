#!/usr/bin/env python3
"""
Inference script for fine-tuned Llama models.
Supports both LoRA and full fine-tuned models.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaInference:
    """Wrapper class for Llama model inference."""
    
    def __init__(self, model_path: str, base_model: Optional[str] = None, 
                 use_4bit: bool = False, use_8bit: bool = False, device: str = "auto"):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the fine-tuned model
            base_model: Base model name (required for LoRA models)
            use_4bit: Whether to use 4-bit quantization
            use_8bit: Whether to use 8-bit quantization
            device: Device to load the model on
        """
        self.model_path = model_path
        self.base_model = base_model
        self.device = device
        
        # Setup quantization config
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
        tokenizer_path = model_path if base_model is None else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Check if this is a LoRA model
        try:
            peft_config = PeftConfig.from_pretrained(model_path)
            is_peft_model = True
            if base_model is None:
                base_model = peft_config.base_model_name_or_path
                logger.info(f"Detected LoRA model. Base model: {base_model}")
        except:
            is_peft_model = False
            logger.info("Loading full fine-tuned model")
        
        # Load model
        if is_peft_model:
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_4bit or use_8bit else torch.float16
            )
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_4bit or use_8bit else torch.float16
            )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 50, do_sample: bool = True) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            Generated text
        """
        # Format prompt for instruction following
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = response[len(formatted_prompt):].strip()
        
        return response
    
    def chat(self):
        """Interactive chat mode."""
        print("Llama Fine-tuned Model Chat")
        print("Type 'quit' to exit, 'clear' to clear history")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\nYou: ").strip()
                
                if prompt.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'clear':
                    print("Chat history cleared!")
                    continue
                
                if not prompt:
                    continue
                
                print("Assistant: ", end="", flush=True)
                response = self.generate(prompt)
                print(response)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Llama model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str,
                       help="Base model name (required for LoRA models)")
    parser.add_argument("--prompt", type=str,
                       help="Single prompt for generation")
    parser.add_argument("--prompts_file", type=str,
                       help="File containing prompts (one per line)")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--no_sample", action="store_true",
                       help="Disable sampling (use greedy decoding)")
    
    # Optimization arguments
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to load the model on")
    
    # Mode arguments
    parser.add_argument("--chat", action="store_true",
                       help="Start interactive chat mode")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = LlamaInference(
        model_path=args.model_path,
        base_model=args.base_model,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        device=args.device
    )
    
    if args.chat:
        # Interactive chat mode
        inference.chat()
    elif args.prompt:
        # Single prompt
        response = inference.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.no_sample
        )
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")
    elif args.prompts_file:
        # Batch processing
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1} ---")
            print(f"Input: {prompt}")
            response = inference.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=not args.no_sample
            )
            print(f"Output: {response}")
    else:
        print("Please provide --prompt, --prompts_file, or use --chat mode")


if __name__ == "__main__":
    main()