#!/usr/bin/env python3
"""
DeepSeek LEAN Code Generation and Backtesting Pipeline
=====================================================

This script provides an end-to-end pipeline for:
1. Taking natural language prompts about trading strategies
2. Generating LEAN algorithm code using fine-tuned DeepSeek R1
3. Running backtests on the generated code
4. Producing comprehensive result reports

Features:
- Fine-tuned model inference
- Code validation and syntax checking
- Automated LEAN backtesting
- Performance analysis and reporting
- Error handling and code debugging
"""

import os
import json
import torch
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import ast
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LEANCodeGenerator:
    """Generates and tests LEAN algorithm code using fine-tuned DeepSeek R1"""
    
    def __init__(self, model_path: str = "./deepseek-lean-finetuned", config_path: str = "finetuning_config.yaml"):
        """Initialize the code generator"""
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.lean_path = "./Lean"
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading fine-tuned model from {self.model_path}")
        
        # Load base model
        base_model_name = self.config.get('model', {}).get('name', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load fine-tuned adapters
        if os.path.exists(self.model_path):
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            logger.info("Fine-tuned adapters loaded successfully")
        else:
            self.model = base_model
            logger.warning("Fine-tuned model not found, using base model")
        
        self.model.eval()
    
    def generate_lean_code(self, prompt: str, max_length: int = 2048) -> str:
        """Generate LEAN algorithm code from natural language prompt"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format prompt for the model
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate code
        generation_config = self.config.get('generation', {})
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=generation_config.get('temperature', 0.7),
                top_p=generation_config.get('top_p', 0.9),
                top_k=generation_config.get('top_k', 50),
                do_sample=generation_config.get('do_sample', True),
                repetition_penalty=generation_config.get('repetition_penalty', 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        try:
            response_start = generated_text.find("<|assistant|>\n") + len("<|assistant|>\n")
            generated_code = generated_text[response_start:].strip()
            
            # Clean up any remaining tokens
            if "<|end|>" in generated_code:
                generated_code = generated_code[:generated_code.find("<|end|>")]
            
            return generated_code.strip()
        except Exception as e:
            logger.error(f"Error extracting generated code: {e}")
            return generated_text
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate the generated LEAN code"""
        errors = []
        
        try:
            # Basic syntax check
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax Error: {e}")
        
        # Check for required LEAN patterns
        required_patterns = [
            r'from\s+AlgorithmImports\s+import\s+\*',
            r'class\s+\w+\(QCAlgorithm\):',
            r'def\s+Initialize\(self\):',
            r'def\s+OnData\(self,\s*data\):'
        ]
        
        for pattern in required_patterns:
            if not re.search(pattern, code):
                errors.append(f"Missing required pattern: {pattern}")
        
        # Check for common LEAN methods
        lean_methods = ['SetStartDate', 'SetEndDate', 'SetCash', 'AddEquity']
        for method in lean_methods:
            if method not in code:
                errors.append(f"Missing common LEAN method: {method}")
        
        return len(errors) == 0, errors
    
    def run_backtest(self, code: str, algorithm_name: str = None) -> Dict[str, Any]:
        """Run a backtest on the generated LEAN code"""
        if algorithm_name is None:
            algorithm_name = f"Generated_Algorithm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create temporary algorithm file
        algorithm_file = os.path.join(self.lean_path, "Algorithm.Python", f"{algorithm_name}.py")
        
        try:
            # Write algorithm to file
            with open(algorithm_file, 'w') as f:
                f.write(code)
            
            # Update config to use the new algorithm
            config_file = "config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            config['algorithm-location'] = f"Algorithm.Python/{algorithm_name}.py"
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Run the backtest using Docker
            result = self._run_lean_docker()
            
            return {
                'success': True,
                'algorithm_name': algorithm_name,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {
                'success': False,
                'error': str(e),
                'algorithm_name': algorithm_name
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(algorithm_file):
                os.remove(algorithm_file)
    
    def _run_lean_docker(self) -> Dict[str, Any]:
        """Run LEAN using Docker"""
        try:
            # Run Docker Compose
            cmd = ["docker-compose", "up", "--build"]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd="."
            )
            
            if process.returncode == 0:
                # Parse results
                results_dir = os.path.join(self.lean_path, "Results")
                return self._parse_backtest_results(results_dir)
            else:
                raise Exception(f"Docker process failed: {process.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Backtest timed out after 5 minutes")
        except Exception as e:
            raise Exception(f"Error running Docker: {e}")
    
    def _parse_backtest_results(self, results_dir: str) -> Dict[str, Any]:
        """Parse backtest results from LEAN output"""
        results = {}
        
        # Look for results files
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    data = json.load(f)
                    results[filename] = data
        
        return results
    
    def generate_report(self, backtest_result: Dict[str, Any], prompt: str, code: str) -> str:
        """Generate comprehensive backtest report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# LEAN Algorithm Backtest Report

**Generated on:** {timestamp}

## Strategy Prompt

{prompt}

## Generated Algorithm Code

```python
{code}
```

## Backtest Results

"""
        
        if backtest_result.get('success'):
            # Extract key metrics from results
            results = backtest_result.get('result', {})
            
            report += """
### Performance Summary

| Metric | Value |
|--------|-------|
"""
            
            # Add performance metrics if available
            for file_name, data in results.items():
                if 'Statistics' in data:
                    stats = data['Statistics']
                    for key, value in stats.items():
                        report += f"| {key} | {value} |\n"
            
            report += """
### Trade Analysis

"""
            
            # Add trade information if available
            for file_name, data in results.items():
                if 'Orders' in data:
                    orders = data['Orders']
                    report += f"**Total Orders:** {len(orders)}\n\n"
                    
                    if orders:
                        report += "| Time | Symbol | Direction | Quantity | Price |\n"
                        report += "|------|--------|-----------|----------|-------|\n"
                        
                        for order in orders[:10]:  # Show first 10 orders
                            report += f"| {order.get('Time', 'N/A')} | {order.get('Symbol', 'N/A')} | {order.get('Direction', 'N/A')} | {order.get('Quantity', 'N/A')} | {order.get('Price', 'N/A')} |\n"
                        
                        if len(orders) > 10:
                            report += f"\n... and {len(orders) - 10} more orders\n"
        
        else:
            report += f"""
### Error Details

**Error:** {backtest_result.get('error', 'Unknown error')}

The backtest failed to run successfully. Please check the generated code and try again.
"""
        
        return report

class PromptToBacktestPipeline:
    """End-to-end pipeline from prompt to backtest results"""
    
    def __init__(self, model_path: str = "./deepseek-lean-finetuned"):
        """Initialize the pipeline"""
        self.generator = LEANCodeGenerator(model_path)
        self.reports_dir = "./backtest_reports"
        
        # Create reports directory
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def run_pipeline(self, prompt: str, run_backtest: bool = True) -> Dict[str, Any]:
        """Run the complete pipeline from prompt to results"""
        logger.info("Starting prompt-to-backtest pipeline")
        
        pipeline_result = {
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Step 1: Load model if not already loaded
            if self.generator.model is None:
                logger.info("Loading model...")
                self.generator.load_model()
            
            # Step 2: Generate code
            logger.info("Generating LEAN algorithm code...")
            generated_code = self.generator.generate_lean_code(prompt)
            pipeline_result['generated_code'] = generated_code
            
            # Step 3: Validate code
            logger.info("Validating generated code...")
            is_valid, validation_errors = self.generator.validate_code(generated_code)
            pipeline_result['validation'] = {
                'is_valid': is_valid,
                'errors': validation_errors
            }
            
            if not is_valid:
                logger.warning(f"Code validation failed: {validation_errors}")
                if not run_backtest:
                    return pipeline_result
            
            # Step 4: Run backtest (if requested and validation passed)
            if run_backtest:
                logger.info("Running backtest...")
                backtest_result = self.generator.run_backtest(generated_code)
                pipeline_result['backtest'] = backtest_result
                
                # Step 5: Generate report
                logger.info("Generating report...")
                report = self.generator.generate_report(backtest_result, prompt, generated_code)
                pipeline_result['report'] = report
                
                # Save report to file
                report_filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                report_path = os.path.join(self.reports_dir, report_filename)
                
                with open(report_path, 'w') as f:
                    f.write(report)
                
                pipeline_result['report_file'] = report_path
                pipeline_result['success'] = backtest_result.get('success', False)
            else:
                pipeline_result['success'] = is_valid
            
            logger.info("Pipeline completed successfully")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_result['error'] = str(e)
            return pipeline_result

def main():
    """Main function for testing the pipeline"""
    # Example prompts to test
    test_prompts = [
        "Create a simple moving average crossover strategy using 10-day and 20-day SMAs for SPY with $100,000 starting capital.",
        "Build an RSI mean reversion strategy that buys when RSI < 30 and sells when RSI > 70 for AAPL.",
        "Design a momentum strategy that trades tech stocks based on 50-day high breakouts with proper risk management."
    ]
    
    # Initialize pipeline
    pipeline = PromptToBacktestPipeline()
    
    # Run pipeline for each test prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Testing Prompt {i}: {prompt[:50]}...")
        print(f"{'='*60}")
        
        result = pipeline.run_pipeline(prompt, run_backtest=True)
        
        if result['success']:
            print(f"✅ Pipeline successful!")
            print(f"📊 Report saved to: {result.get('report_file', 'N/A')}")
        else:
            print(f"❌ Pipeline failed:")
            print(f"Error: {result.get('error', 'Unknown error')}")
            if 'validation' in result and not result['validation']['is_valid']:
                print(f"Validation errors: {result['validation']['errors']}")

if __name__ == "__main__":
    main()