#!/usr/bin/env python3
"""
Setup script for Llama fine-tuning environment.
Checks requirements and helps with initial setup.
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def run_command(command, description):
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_gpu():
    """Check if CUDA GPU is available."""
    print("\n=== GPU Check ===")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("❌ CUDA not available")
            print("   You can still use CPU, but training will be very slow")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_huggingface_auth():
    """Check HuggingFace authentication."""
    print("\n=== HuggingFace Authentication ===")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Try to get user info
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not logged in to HuggingFace")
        print("   Run: huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")
        return False

def install_requirements():
    """Install required packages."""
    print("\n=== Installing Requirements ===")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        "Installing Python packages"
    )

def create_sample_config():
    """Create a sample configuration file."""
    print("\n=== Creating Sample Configuration ===")
    config_path = Path(__file__).parent / "my_config.yaml"
    
    if config_path.exists():
        print(f"✅ Configuration already exists: {config_path}")
        return True
    
    try:
        with open("config.yaml", "r") as f:
            template_config = f.read()
        
        with open(config_path, "w") as f:
            f.write(template_config)
        
        print(f"✅ Created configuration: {config_path}")
        print("   Edit this file to customize your training settings")
        return True
    
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Test dataset creation
    success = run_command(
        "python prepare_dataset.py --create_sample --output test_dataset.json",
        "Creating test dataset"
    )
    
    if success:
        # Check if file was created
        if Path("test_dataset.json").exists():
            print("✅ Test dataset created successfully")
            # Clean up
            os.remove("test_dataset.json")
        else:
            print("❌ Test dataset file not found")
            success = False
    
    return success

def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*50)
    print("🎉 SETUP COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. 📝 Edit my_config.yaml with your preferred settings")
    print("2. 🤗 Ensure you have access to Llama models on HuggingFace")
    print("3. 🚀 Start with the basic example:")
    print("   cd examples && ./basic_training.sh")
    print("\nOr run training manually:")
    print("python train_llama.py \\")
    print("    --model_name meta-llama/Llama-2-7b-hf \\")
    print("    --dataset_name tatsu-lab/alpaca \\")
    print("    --use_lora --use_4bit")
    print("\n📚 See README.md for detailed documentation")

def main():
    """Main setup function."""
    print("🦙 LLAMA FINE-TUNING SETUP")
    print("="*30)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    checks = []
    
    # Install requirements
    checks.append(install_requirements())
    
    # Check GPU
    checks.append(check_gpu())
    
    # Check HuggingFace auth
    checks.append(check_huggingface_auth())
    
    # Create sample config
    checks.append(create_sample_config())
    
    # Test basic functionality
    checks.append(test_basic_functionality())
    
    # Summary
    print("\n" + "="*30)
    print("SETUP SUMMARY")
    print("="*30)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All checks passed!")
        print_next_steps()
    else:
        print("⚠️  Some checks failed. Please resolve the issues above.")
        print("\nYou can still proceed, but some features may not work correctly.")
        
        if input("\nShow next steps anyway? (y/N): ").lower().startswith('y'):
            print_next_steps()

if __name__ == "__main__":
    main()