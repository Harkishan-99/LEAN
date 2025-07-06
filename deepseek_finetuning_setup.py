#!/usr/bin/env python3
"""
DeepSeek R1 Fine-tuning Setup for LEAN Backtesting Code Generation
=================================================================

This script sets up the complete fine-tuning pipeline for DeepSeek R1 using QLoRA
to generate LEAN algorithm backtesting code from natural language prompts.

Features:
- QLoRA fine-tuning configuration
- Custom dataset creation for LEAN algorithms
- Training pipeline with monitoring
- Inference pipeline for code generation
- Integration with LEAN Docker environment
"""

import os
import json
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset, load_dataset
import bitsandbytes as bnb
from datetime import datetime
import logging
import wandb
from typing import Dict, List, Optional, Any
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekFineTuner:
    """Fine-tuning manager for DeepSeek R1 with QLoRA for LEAN code generation"""
    
    def __init__(self, config_path: str = "finetuning_config.yaml"):
        """Initialize the fine-tuner with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            "model": {
                "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "max_length": 4096,
                "torch_dtype": "float16"
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "training": {
                "output_dir": "./deepseek-lean-finetuned",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "evaluation_strategy": "steps",
                "eval_steps": 500,
                "fp16": True,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False
            },
            "dataset": {
                "train_file": "lean_training_dataset.json",
                "val_split": 0.1,
                "max_samples": 1000
            },
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge configurations
                def merge_configs(default, user):
                    for key, value in user.items():
                        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                            merge_configs(default[key], value)
                        else:
                            default[key] = value
                merge_configs(default_config, user_config)
        
        return default_config
    
    def setup_model_and_tokenizer(self):
        """Setup the model and tokenizer with quantization"""
        logger.info(f"Loading model: {self.config['model']['name']}")
        
        # Configure quantization
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype']),
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant']
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=getattr(torch, self.config['model']['torch_dtype'])
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info("Model and tokenizer setup complete")
        
    def create_training_dataset(self):
        """Create and prepare the training dataset"""
        logger.info("Creating training dataset...")
        
        # Generate dataset if it doesn't exist
        dataset_path = self.config['dataset']['train_file']
        if not os.path.exists(dataset_path):
            logger.info("Dataset not found, generating new dataset...")
            self.generate_lean_dataset(dataset_path)
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Limit samples if specified
        max_samples = self.config['dataset']['max_samples']
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
        
        # Create Dataset object
        dataset = Dataset.from_list(data)
        
        # Split into train/validation
        val_split = self.config['dataset']['val_split']
        if val_split > 0:
            dataset = dataset.train_test_split(test_size=val_split, seed=42)
            self.train_dataset = dataset['train']
            self.eval_dataset = dataset['test']
        else:
            self.train_dataset = dataset
            self.eval_dataset = None
        
        logger.info(f"Dataset created: {len(self.train_dataset)} training samples")
        if self.eval_dataset:
            logger.info(f"Validation samples: {len(self.eval_dataset)}")
    
    def tokenize_function(self, examples):
        """Tokenize the examples for training"""
        # Combine prompt and response for training
        texts = []
        for prompt, response in zip(examples['prompt'], examples['response']):
            text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|end|>"
            texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config['model']['max_length'],
            return_tensors="pt"
        )
        
        # Set labels for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def train(self):
        """Run the fine-tuning training"""
        logger.info("Starting fine-tuning training...")
        
        # Tokenize datasets
        train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        eval_dataset = None
        if self.eval_dataset:
            eval_dataset = self.eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=self.eval_dataset.column_names
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            evaluation_strategy=self.config['training']['evaluation_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            fp16=self.config['training']['fp16'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            report_to="wandb" if wandb.run else None,
            run_name=f"deepseek-lean-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config['training']['output_dir'])
        
        logger.info("Training completed successfully")
    
    def generate_lean_dataset(self, output_path: str):
        """Generate training dataset for LEAN algorithm creation"""
        logger.info("Generating LEAN training dataset...")
        
        # This will be populated with comprehensive examples
        dataset = []
        
        # Add the generated examples (this would be expanded significantly)
        examples = self.get_lean_training_examples()
        dataset.extend(examples)
        
        # Save dataset
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Dataset saved to {output_path} with {len(dataset)} examples")
    
    def get_lean_training_examples(self) -> List[Dict[str, str]]:
        """Get comprehensive training examples for LEAN algorithms"""
        return [
            {
                "prompt": "Create a moving average crossover strategy for LEAN that trades when the 10-day SMA crosses above/below the 20-day SMA. Use SPY as the symbol, $100,000 starting capital, and trade from 2020 to 2023.",
                "response": """from AlgorithmImports import *

class MovingAverageCrossoverAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        # Set start and end dates
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        
        # Set starting cash
        self.SetCash(100000)
        
        # Add SPY equity
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Create moving averages
        self.short_sma = self.SMA(self.spy, 10)
        self.long_sma = self.SMA(self.spy, 20)
        
        # Previous values for crossover detection
        self.previous_short = 0
        self.previous_long = 0
        
    def OnData(self, data):
        # Check if indicators are ready
        if not (self.short_sma.IsReady and self.long_sma.IsReady):
            return
            
        # Get current values
        current_short = self.short_sma.Current.Value
        current_long = self.long_sma.Current.Value
        
        # Check for crossover signals
        if self.previous_short != 0 and self.previous_long != 0:
            # Golden cross (buy signal)
            if self.previous_short <= self.previous_long and current_short > current_long:
                if not self.Portfolio[self.spy].Invested:
                    self.SetHoldings(self.spy, 1.0)
                    self.Log(f"BUY: Golden cross detected")
            
            # Death cross (sell signal)
            elif self.previous_short >= self.previous_long and current_short < current_long:
                if self.Portfolio[self.spy].Invested:
                    self.Liquidate(self.spy)
                    self.Log(f"SELL: Death cross detected")
        
        # Update previous values
        self.previous_short = current_short
        self.previous_long = current_long"""
            },
            {
                "prompt": "Build a mean reversion strategy using Bollinger Bands with RSI confirmation. Trade AAPL with 2% position sizing, RSI oversold at 30, overbought at 70, and Bollinger Bands with 20-period and 2 standard deviations.",
                "response": """from AlgorithmImports import *

class BollingerBandsMeanReversionAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add AAPL
        self.aapl = self.AddEquity("AAPL", Resolution.Daily).Symbol
        
        # Create indicators
        self.bb = self.BB(self.aapl, 20, 2)
        self.rsi = self.RSI(self.aapl, 14)
        
        # Position sizing
        self.position_size = 0.02
        
    def OnData(self, data):
        if not (self.bb.IsReady and self.rsi.IsReady):
            return
            
        price = data[self.aapl].Close
        upper_band = self.bb.UpperBand.Current.Value
        lower_band = self.bb.LowerBand.Current.Value
        rsi_value = self.rsi.Current.Value
        
        # Buy signal: Price touches lower band and RSI oversold
        if price <= lower_band and rsi_value <= 30:
            if not self.Portfolio[self.aapl].Invested:
                self.SetHoldings(self.aapl, self.position_size)
                self.Log(f"BUY: Mean reversion signal - Price: {price:.2f}, RSI: {rsi_value:.2f}")
        
        # Sell signal: Price touches upper band and RSI overbought
        elif price >= upper_band and rsi_value >= 70:
            if self.Portfolio[self.aapl].Invested:
                self.Liquidate(self.aapl)
                self.Log(f"SELL: Mean reversion signal - Price: {price:.2f}, RSI: {rsi_value:.2f}")"""
            },
            {
                "prompt": "Create a momentum strategy that buys stocks when they break above their 50-day high and sells when they fall below their 20-day low. Include position sizing based on ATR and risk management with stop losses.",
                "response": """from AlgorithmImports import *

class MomentumBreakoutAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add universe of stocks
        self.symbols = [self.AddEquity(ticker, Resolution.Daily).Symbol 
                       for ticker in ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]]
        
        # Indicators for each symbol
        self.highs = {}
        self.lows = {}
        self.atr = {}
        self.stop_prices = {}
        
        for symbol in self.symbols:
            self.highs[symbol] = self.MAX(symbol, 50)
            self.lows[symbol] = self.MIN(symbol, 20)
            self.atr[symbol] = self.ATR(symbol, 14)
            self.stop_prices[symbol] = 0
        
        # Risk parameters
        self.risk_per_trade = 0.02  # 2% risk per trade
        
    def OnData(self, data):
        for symbol in self.symbols:
            if not (self.highs[symbol].IsReady and self.lows[symbol].IsReady and self.atr[symbol].IsReady):
                continue
                
            if symbol not in data or data[symbol] is None:
                continue
                
            price = data[symbol].Close
            high_50 = self.highs[symbol].Current.Value
            low_20 = self.lows[symbol].Current.Value
            atr_value = self.atr[symbol].Current.Value
            
            holding = self.Portfolio[symbol]
            
            # Entry signal: Break above 50-day high
            if price > high_50 and not holding.Invested:
                # Position sizing based on ATR
                risk_amount = self.Portfolio.TotalPortfolioValue * self.risk_per_trade
                stop_distance = atr_value * 2  # 2 ATR stop loss
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    max_position = self.Portfolio.TotalPortfolioValue * 0.1  # Max 10% per position
                    position_size = min(position_size, max_position)
                    
                    quantity = int(position_size / price)
                    if quantity > 0:
                        self.MarketOrder(symbol, quantity)
                        self.stop_prices[symbol] = price - stop_distance
                        self.Log(f"BUY {symbol}: Breakout at {price:.2f}, Stop: {self.stop_prices[symbol]:.2f}")
            
            # Exit signals
            elif holding.Invested:
                # Stop loss
                if price <= self.stop_prices[symbol]:
                    self.Liquidate(symbol)
                    self.Log(f"SELL {symbol}: Stop loss at {price:.2f}")
                
                # Momentum exit: Fall below 20-day low
                elif price < low_20:
                    self.Liquidate(symbol)
                    self.Log(f"SELL {symbol}: Below 20-day low at {price:.2f}")"""
            },
            {
                "prompt": "Design a pairs trading strategy that trades the spread between two correlated stocks. Use statistical arbitrage with z-score entry/exit signals and include proper risk management.",
                "response": """from AlgorithmImports import *
import numpy as np

class PairsTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add pair of stocks
        self.stock1 = self.AddEquity("KO", Resolution.Daily).Symbol  # Coca-Cola
        self.stock2 = self.AddEquity("PEP", Resolution.Daily).Symbol  # PepsiCo
        
        # Lookback period for correlation and spread calculation
        self.lookback = 60
        self.price_history1 = RollingWindow[float](self.lookback)
        self.price_history2 = RollingWindow[float](self.lookback)
        
        # Trading parameters
        self.entry_zscore = 2.0
        self.exit_zscore = 0.0
        self.stop_loss_zscore = 3.0
        
        # Position tracking
        self.in_position = False
        self.position_type = None  # 'long_spread' or 'short_spread'
        
        # Schedule daily updates
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.AfterMarketOpen(self.stock1, 30), 
                        self.UpdateSpread)
    
    def OnData(self, data):
        # Update price history
        if self.stock1 in data and self.stock2 in data:
            self.price_history1.Add(data[self.stock1].Close)
            self.price_history2.Add(data[self.stock2].Close)
    
    def UpdateSpread(self):
        if not (self.price_history1.IsReady and self.price_history2.IsReady):
            return
        
        # Get price arrays
        prices1 = np.array([x for x in self.price_history1])
        prices2 = np.array([x for x in self.price_history2])
        
        # Calculate hedge ratio using linear regression
        returns1 = np.diff(np.log(prices1))
        returns2 = np.diff(np.log(prices2))
        
        hedge_ratio = np.cov(returns1, returns2)[0,1] / np.var(returns2)
        
        # Calculate spread
        spread = prices1 - hedge_ratio * prices2
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        if spread_std == 0:
            return
        
        # Calculate current z-score
        current_spread = prices1[0] - hedge_ratio * prices2[0]
        zscore = (current_spread - spread_mean) / spread_std
        
        # Trading logic
        if not self.in_position:
            # Enter long spread (buy stock1, sell stock2)
            if zscore < -self.entry_zscore:
                self.SetHoldings(self.stock1, 0.5)
                self.SetHoldings(self.stock2, -0.5 * hedge_ratio)
                self.in_position = True
                self.position_type = 'long_spread'
                self.Log(f"ENTER LONG SPREAD: Z-score {zscore:.2f}")
            
            # Enter short spread (sell stock1, buy stock2)
            elif zscore > self.entry_zscore:
                self.SetHoldings(self.stock1, -0.5)
                self.SetHoldings(self.stock2, 0.5 * hedge_ratio)
                self.in_position = True
                self.position_type = 'short_spread'
                self.Log(f"ENTER SHORT SPREAD: Z-score {zscore:.2f}")
        
        else:
            # Exit conditions
            exit_condition = abs(zscore) <= self.exit_zscore
            stop_loss_condition = abs(zscore) >= self.stop_loss_zscore
            
            if exit_condition or stop_loss_condition:
                self.Liquidate()
                self.in_position = False
                self.position_type = None
                reason = "PROFIT" if exit_condition else "STOP LOSS"
                self.Log(f"EXIT SPREAD ({reason}): Z-score {zscore:.2f}")"""
            }
        ]
    
    def generate_code(self, prompt: str, max_length: int = 1024) -> str:
        """Generate LEAN algorithm code from prompt"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Format prompt
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response_start = generated_text.find("<|assistant|>\n") + len("<|assistant|>\n")
        generated_code = generated_text[response_start:].strip()
        
        return generated_code

def main():
    """Main training pipeline"""
    # Initialize wandb for monitoring (optional)
    # wandb.init(project="deepseek-lean-finetuning")
    
    # Create fine-tuner
    finetuner = DeepSeekFineTuner()
    
    # Setup model and tokenizer
    finetuner.setup_model_and_tokenizer()
    
    # Create training dataset
    finetuner.create_training_dataset()
    
    # Run training
    finetuner.train()
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()