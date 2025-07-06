#!/usr/bin/env python3
"""
DeepSeek R1 LEAN Fine-tuning Setup Script
=========================================

This script automates the setup process for the DeepSeek R1 fine-tuning environment
for LEAN algorithmic trading code generation.

Usage:
    python setup.py [--generate-dataset] [--install-deps] [--test-env]
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU (slow)")
    except ImportError:
        print("⚠️  PyTorch not installed")
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print("❌ Docker not available")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        'model_cache',
        'logs',
        'backtest_reports',
        'deepseek-lean-finetuned',
        'Lean/Results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created")

def generate_sample_dataset():
    """Generate a sample training dataset"""
    print("📊 Generating sample training dataset...")
    
    try:
        from dataset_generator import LEANDatasetGenerator
        
        generator = LEANDatasetGenerator()
        dataset = generator.generate_comprehensive_dataset(100)  # Small sample for testing
        generator.save_dataset(dataset, "lean_training_dataset_sample.json")
        
        print(f"✅ Sample dataset generated: {len(dataset)} examples")
        return True
    except Exception as e:
        print(f"❌ Failed to generate dataset: {e}")
        return False

def test_environment():
    """Test the environment setup"""
    print("🧪 Testing environment...")
    
    # Test basic imports
    try:
        import torch
        import transformers
        import peft
        import datasets
        print("✅ Core libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test LEAN structure
    lean_files = [
        'Lean/Algorithm.Python/main.py',
        'config.json',
        'docker-compose.yml'
    ]
    
    for file_path in lean_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"⚠️  Missing: {file_path}")
    
    return True

def create_sample_config():
    """Create sample configuration files"""
    print("⚙️ Creating sample configuration...")
    
    # Create .env file if it doesn't exist
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write("""# DeepSeek R1 Fine-tuning Environment Configuration
# =================================================

# Optional: Weights & Biases for monitoring
# WANDB_PROJECT=deepseek-lean-finetuning
# WANDB_ENTITY=your-username

# Model cache directory
HF_HOME=./model_cache

# CUDA settings (adjust based on your GPU setup)
CUDA_VISIBLE_DEVICES=0

# Training settings
TOKENIZERS_PARALLELISM=false
""")
        print("✅ Created .env file")
    
    # Verify config files exist
    config_files = ['finetuning_config.yaml', 'config.json']
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ Found: {config_file}")
        else:
            print(f"⚠️  Missing: {config_file}")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup DeepSeek R1 LEAN fine-tuning environment")
    parser.add_argument('--generate-dataset', action='store_true', help='Generate sample training dataset')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--test-env', action='store_true', help='Test environment')
    parser.add_argument('--all', action='store_true', help='Run all setup steps')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        args.all = True  # Default to all if no specific flags
    
    print("🚀 DeepSeek R1 LEAN Fine-tuning Setup")
    print("=" * 50)
    
    # Check requirements first
    if not check_requirements():
        print("❌ Requirements check failed. Please install missing components.")
        return 1
    
    success = True
    
    # Install dependencies
    if args.install_deps or args.all:
        if not install_dependencies():
            success = False
    
    # Create directories
    if args.all:
        create_directories()
    
    # Create configuration
    if args.all:
        create_sample_config()
    
    # Generate dataset
    if args.generate_dataset or args.all:
        if not generate_sample_dataset():
            success = False
    
    # Test environment
    if args.test_env or args.all:
        if not test_environment():
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Generate full training dataset: python dataset_generator.py")
        print("2. Start fine-tuning: python deepseek_finetuning_setup.py")
        print("3. Test inference: python inference_pipeline.py")
        print("\n📖 See DEEPSEEK_FINETUNING_GUIDE.md for detailed instructions")
    else:
        print("❌ Setup completed with errors. Please check the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())