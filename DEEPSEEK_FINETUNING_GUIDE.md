# DeepSeek R1 Fine-tuning for LEAN Backtesting - Complete Guide

This comprehensive guide covers the complete setup for fine-tuning DeepSeek R1 using QLoRA to generate LEAN algorithmic trading code from natural language prompts, including an end-to-end pipeline from prompt to backtest results.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Quick Start](#quick-start)
4. [Installation & Setup](#installation--setup)
5. [Dataset Generation](#dataset-generation)
6. [Fine-tuning Process](#fine-tuning-process)
7. [Inference Pipeline](#inference-pipeline)
8. [End-to-End Workflow](#end-to-end-workflow)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)

---

## Overview

### 🎯 **What This Setup Does**

This project provides a complete solution for:

- **🔧 Fine-tuning DeepSeek R1** using QLoRA for efficient training
- **📊 Generating LEAN algorithms** from natural language descriptions
- **🚀 Running automated backtests** on generated strategies
- **📈 Producing comprehensive reports** with performance analysis

### 🏗️ **Architecture**

```
Prompt → Fine-tuned DeepSeek R1 → LEAN Code → Docker Backtest → Results Report
```

### ✨ **Key Features**

- **QLoRA Fine-tuning**: Memory-efficient training with 4-bit quantization
- **Custom Dataset**: 1000+ LEAN algorithm examples with variations
- **Code Validation**: Syntax checking and LEAN pattern validation
- **Automated Backtesting**: Docker-based LEAN execution
- **Comprehensive Reporting**: Detailed performance analysis and trade logs

---

## Requirements

### 🖥️ **Hardware Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 8GB VRAM (RTX 3070) | 16GB+ VRAM (RTX 4080/A100) |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 50GB free | 100GB+ SSD |
| **CPU** | 8 cores | 16+ cores |

### 🐧 **Software Requirements**

- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU acceleration)
- **Docker**: Latest version
- **Git**: For repository management

---

## Quick Start

### 🚀 **5-Minute Setup**

```bash
# 1. Clone repository
git clone <your-repo-url>
cd deepseek-lean-finetuning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate training dataset
python dataset_generator.py

# 4. Start fine-tuning
python deepseek_finetuning_setup.py

# 5. Run inference pipeline
python inference_pipeline.py
```

### 🧪 **Quick Test**

```python
from inference_pipeline import PromptToBacktestPipeline

# Initialize pipeline
pipeline = PromptToBacktestPipeline()

# Generate and test a strategy
result = pipeline.run_pipeline(
    "Create a moving average crossover strategy for SPY",
    run_backtest=True
)

print(f"Success: {result['success']}")
print(f"Report: {result.get('report_file', 'N/A')}")
```

---

## Installation & Setup

### 📦 **Step 1: Environment Setup**

```bash
# Create virtual environment
python -m venv deepseek_env
source deepseek_env/bin/activate  # On Windows: deepseek_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 🔧 **Step 2: Configure Environment**

Create `.env` file:
```bash
# Optional: Weights & Biases for monitoring
WANDB_PROJECT=deepseek-lean-finetuning
WANDB_ENTITY=your-username

# Model cache directory
HF_HOME=./model_cache

# CUDA settings
CUDA_VISIBLE_DEVICES=0
```

### 🐳 **Step 3: Docker Setup**

Ensure Docker is running for LEAN backtesting:
```bash
# Start Docker daemon
sudo systemctl start docker

# Test Docker
docker --version
docker-compose --version
```

---

## Dataset Generation

### 📚 **Creating Training Data**

The dataset generator creates comprehensive LEAN algorithm examples:

```bash
# Generate 1000 training examples
python dataset_generator.py
```

**Generated Dataset Structure:**
```json
[
  {
    "prompt": "Create a moving average crossover strategy...",
    "response": "from AlgorithmImports import *\n\nclass Strategy(QCAlgorithm):\n..."
  }
]
```

### 🎯 **Strategy Types Included**

| Strategy Type | Examples | Description |
|---------------|----------|-------------|
| **Moving Average** | SMA/EMA Crossovers | Trend-following strategies |
| **Mean Reversion** | RSI, Bollinger Bands | Counter-trend strategies |
| **Momentum** | Breakout strategies | Trend continuation |
| **Pairs Trading** | Statistical arbitrage | Market-neutral strategies |
| **Risk Management** | Stop losses, position sizing | Risk control techniques |

### 🔄 **Data Augmentation**

The generator automatically creates variations:
- Different symbols (SPY, AAPL, MSFT, etc.)
- Various timeframes (Daily, Hourly, etc.)
- Different parameter values
- Multiple position sizing approaches

---

## Fine-tuning Process

### ⚙️ **Configuration**

Fine-tuning parameters in `finetuning_config.yaml`:

```yaml
# Model settings
model:
  name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  max_length: 4096

# QLoRA settings
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training settings
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
```

### 🏋️ **Starting Fine-tuning**

```bash
# Start training with default config
python deepseek_finetuning_setup.py

# Or with custom config
python deepseek_finetuning_setup.py --config custom_config.yaml
```

### 📊 **Monitoring Training**

**Option 1: Weights & Biases**
```python
import wandb
wandb.init(project="deepseek-lean-finetuning")
```

**Option 2: TensorBoard**
```bash
tensorboard --logdir ./logs
```

### 💾 **Training Output**

Fine-tuning produces:
- `./deepseek-lean-finetuned/` - Model adapters
- `./logs/` - Training logs
- Checkpoints at regular intervals

---

## Inference Pipeline

### 🧠 **Code Generation**

The inference pipeline converts prompts to LEAN code:

```python
from inference_pipeline import LEANCodeGenerator

# Initialize generator
generator = LEANCodeGenerator("./deepseek-lean-finetuned")
generator.load_model()

# Generate code
prompt = "Create an RSI strategy that buys when RSI < 30"
code = generator.generate_lean_code(prompt)
print(code)
```

### ✅ **Code Validation**

Automatic validation includes:
- **Syntax checking** using AST parsing
- **LEAN pattern validation** (required imports, class structure)
- **Method verification** (Initialize, OnData, etc.)

```python
is_valid, errors = generator.validate_code(code)
if not is_valid:
    print("Validation errors:", errors)
```

### 🏃 **Backtest Execution**

Automated LEAN backtesting:

```python
# Run backtest
result = generator.run_backtest(code)

if result['success']:
    print("Backtest completed successfully!")
    print("Results:", result['result'])
else:
    print("Backtest failed:", result['error'])
```

---

## End-to-End Workflow

### 🔄 **Complete Pipeline**

```python
from inference_pipeline import PromptToBacktestPipeline

# Initialize pipeline
pipeline = PromptToBacktestPipeline()

# Define trading strategy prompt
prompt = """
Create a momentum strategy that:
1. Trades SPY and QQQ
2. Buys on 20-day high breakouts
3. Uses 2% position sizing
4. Has 5% stop losses
"""

# Run complete pipeline
result = pipeline.run_pipeline(prompt, run_backtest=True)

# Check results
if result['success']:
    print(f"✅ Success! Report: {result['report_file']}")
    print(f"📊 Generated code validated: {result['validation']['is_valid']}")
    print(f"🚀 Backtest completed: {result['backtest']['success']}")
else:
    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
```

### 📋 **Pipeline Steps**

1. **🔧 Model Loading**: Load fine-tuned DeepSeek R1
2. **🧠 Code Generation**: Convert prompt to LEAN algorithm
3. **✅ Validation**: Check syntax and LEAN patterns
4. **🐳 Docker Setup**: Prepare LEAN environment
5. **🚀 Backtest Execution**: Run strategy simulation
6. **📊 Result Parsing**: Extract performance metrics
7. **📝 Report Generation**: Create comprehensive analysis

### 📈 **Generated Reports**

Reports include:
- **Strategy Description**: Original prompt and generated code
- **Performance Metrics**: Returns, Sharpe ratio, drawdown
- **Trade Analysis**: Entry/exit points, win rate
- **Risk Metrics**: Volatility, maximum drawdown
- **Execution Details**: Order history and timing

---

## Configuration

### 🔧 **Fine-tuning Configuration**

Key parameters in `finetuning_config.yaml`:

```yaml
# Memory efficiency
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"

# Learning parameters
training:
  learning_rate: 2e-4        # Lower for stability
  weight_decay: 0.01         # Regularization
  warmup_steps: 100          # Gradual learning rate increase
  
# Monitoring
monitoring:
  wandb:
    enabled: true
    project: "deepseek-lean-finetuning"
```

### 📊 **Dataset Configuration**

Control dataset generation:

```python
# Custom dataset generation
generator = LEANDatasetGenerator()

# Generate specific strategy types
dataset = generator.generate_comprehensive_dataset(
    num_samples=2000,
    strategy_focus=["momentum", "mean_reversion"],
    include_risk_management=True
)
```

### 🐳 **Docker Configuration**

LEAN backtesting setup in `docker-compose.yml`:

```yaml
services:
  lean-engine:
    image: quantconnect/lean:latest
    volumes:
      - ./Lean/Algorithm.Python:/Lean/Algorithm.Python:ro
      - ./Lean/Data:/Lean/Data:ro
      - ./Lean/Results:/Results
    environment:
      - LEAN_CONFIG_FILE=/Lean/config.json
```

---

## Troubleshooting

### 🐛 **Common Issues**

#### **GPU Memory Issues**
```python
# Reduce batch size
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16

# Use gradient checkpointing
  gradient_checkpointing: true
```

#### **Model Loading Errors**
```bash
# Clear cache and reinstall
rm -rf ~/.cache/huggingface
pip install --upgrade transformers torch
```

#### **Docker Issues**
```bash
# Restart Docker service
sudo systemctl restart docker

# Check Docker permissions
sudo usermod -aG docker $USER
```

#### **Code Generation Quality**
```yaml
# Adjust generation parameters
generation:
  temperature: 0.5      # Lower for more conservative generation
  top_p: 0.8           # Reduce for more focused output
  repetition_penalty: 1.2  # Prevent repetition
```

### 🔍 **Debugging Tips**

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test with simpler prompts**:
   ```python
   simple_prompt = "Create a basic buy and hold strategy for SPY"
   ```

3. **Validate training data**:
   ```python
   # Check dataset quality
   with open('lean_training_dataset.json', 'r') as f:
       data = json.load(f)
       print(f"Dataset size: {len(data)}")
       print(f"Sample prompt: {data[0]['prompt'][:100]}...")
   ```

### 📞 **Getting Help**

- **Training Issues**: Check GPU memory and reduce batch size
- **Generation Issues**: Adjust temperature and sampling parameters
- **Docker Issues**: Ensure Docker daemon is running
- **LEAN Issues**: Verify algorithm syntax and imports

---

## Advanced Topics

### 🚀 **Performance Optimization**

#### **Multi-GPU Training**
```python
# Use DataParallel or DistributedDataParallel
training_args = TrainingArguments(
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    per_device_train_batch_size=2
)
```

#### **Memory Optimization**
```python
# Enable gradient checkpointing and mixed precision
training:
  gradient_checkpointing: true
  fp16: true
  dataloader_pin_memory: true
```

### 🎯 **Custom Strategy Types**

Add new strategy categories:

```python
class CustomStrategyGenerator:
    def generate_options_strategy(self):
        return {
            "prompt": "Create an options volatility strategy...",
            "response": "# Options-specific LEAN code..."
        }
    
    def generate_crypto_strategy(self):
        return {
            "prompt": "Build a cryptocurrency arbitrage strategy...",
            "response": "# Crypto-specific LEAN code..."
        }
```

### 📊 **Advanced Monitoring**

#### **Custom Metrics**
```python
def compute_custom_metrics(eval_pred):
    predictions, labels = eval_pred
    # Custom metric calculation
    return {"custom_score": custom_metric}

trainer = Trainer(
    compute_metrics=compute_custom_metrics
)
```

#### **Real-time Monitoring**
```python
# Custom callback for real-time monitoring
class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, **kwargs):
        # Custom logging logic
        pass
```

### 🔄 **Continuous Learning**

#### **Incremental Training**
```python
# Resume from checkpoint
finetuner = DeepSeekFineTuner()
finetuner.setup_model_and_tokenizer()

# Load previous checkpoint
trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-1000")
```

#### **Active Learning**
```python
# Select best examples for additional training
def select_best_examples(generated_results):
    # Logic to identify high-performing strategies
    return selected_examples
```

---

## 🎉 **Conclusion**

This setup provides a complete solution for fine-tuning DeepSeek R1 to generate LEAN algorithmic trading code. The end-to-end pipeline from natural language prompts to backtest results enables rapid strategy development and testing.

### 🏆 **Key Benefits**

- **🚀 Rapid Development**: Generate strategies in seconds
- **✅ Quality Assurance**: Automated validation and testing
- **📊 Comprehensive Analysis**: Detailed performance reports
- **🔧 Highly Configurable**: Flexible parameters and settings
- **📈 Scalable**: Support for multiple strategy types and assets

### 🔜 **Next Steps**

1. **Expand Dataset**: Add more strategy types and market conditions
2. **Improve Generation**: Fine-tune generation parameters
3. **Add Features**: Real-time trading, paper trading integration
4. **Optimize Performance**: Multi-GPU training, faster inference

---

## 📚 **Additional Resources**

- [LEAN Documentation](https://lean.quantconnect.com/)
- [DeepSeek Model Hub](https://huggingface.co/deepseek-ai)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

*Last updated: January 2025*