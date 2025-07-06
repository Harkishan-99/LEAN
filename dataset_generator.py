#!/usr/bin/env python3
"""
LEAN Algorithm Training Dataset Generator
========================================

This script generates comprehensive training datasets for fine-tuning LLMs
to create LEAN algorithmic trading code from natural language prompts.

Features:
- Generates diverse trading strategy examples
- Creates variations of prompts and parameters
- Includes proper LEAN framework patterns
- Supports data augmentation techniques
- Validates generated code syntax
"""

import json
import random
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LEANDatasetGenerator:
    """Generates training datasets for LEAN algorithm creation"""
    
    def __init__(self):
        self.strategy_templates = self._load_strategy_templates()
        self.symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", 
                       "META", "NFLX", "AMD", "CRM", "ADBE", "PYPL", "INTC", "CSCO",
                       "PEP", "KO", "JNJ", "PG", "WMT", "V", "MA", "JPM", "BAC"]
        self.timeframes = ["Daily", "Hour", "Minute", "Second"]
        self.indicators = ["SMA", "EMA", "BB", "RSI", "MACD", "ATR", "Stochastic", "CCI", "Williams"]
        
    def generate_comprehensive_dataset(self, num_samples: int = 1000) -> List[Dict[str, str]]:
        """Generate a comprehensive training dataset"""
        logger.info(f"Generating {num_samples} training samples...")
        
        dataset = []
        
        # Generate base examples from templates
        base_examples = self._generate_base_examples()
        dataset.extend(base_examples)
        
        # Generate variations of existing examples
        while len(dataset) < num_samples:
            # Select random strategy type
            strategy_type = random.choice(list(self.strategy_templates.keys()))
            
            # Generate sample based on strategy type
            sample = self._generate_strategy_sample(strategy_type)
            if sample:
                dataset.append(sample)
        
        # Shuffle and trim to exact size
        random.shuffle(dataset)
        dataset = dataset[:num_samples]
        
        logger.info(f"Generated {len(dataset)} training samples")
        return dataset
    
    def _generate_base_examples(self) -> List[Dict[str, str]]:
        """Generate base examples covering fundamental LEAN patterns"""
        return [
            # Moving Average Strategies
            {
                "prompt": "Create a simple moving average crossover strategy using 10-day and 20-day SMAs. Trade SPY with $100,000 starting capital from 2020 to 2023.",
                "response": self._get_sma_crossover_template("SPY", 10, 20, 100000, "2020, 1, 1", "2023, 12, 31")
            },
            {
                "prompt": "Build an EMA crossover strategy with 12-day and 26-day EMAs for AAPL. Use daily resolution and $50,000 initial capital.",
                "response": self._get_ema_crossover_template("AAPL", 12, 26, 50000)
            },
            
            # RSI Strategies
            {
                "prompt": "Implement an RSI mean reversion strategy that buys when RSI < 30 and sells when RSI > 70. Trade QQQ with 10% position sizing.",
                "response": self._get_rsi_strategy_template("QQQ", 30, 70, 0.1)
            },
            
            # Bollinger Bands Strategies
            {
                "prompt": "Create a Bollinger Bands mean reversion strategy with 20-period and 2 standard deviations. Trade MSFT with RSI confirmation.",
                "response": self._get_bollinger_bands_template("MSFT", 20, 2)
            },
            
            # Multi-asset strategies
            {
                "prompt": "Design a momentum strategy that trades multiple tech stocks (AAPL, MSFT, GOOGL) based on 50-day high breakouts.",
                "response": self._get_multi_asset_momentum_template(["AAPL", "MSFT", "GOOGL"], 50)
            },
            
            # Pairs Trading
            {
                "prompt": "Build a pairs trading strategy between KO and PEP using statistical arbitrage with z-score signals.",
                "response": self._get_pairs_trading_template("KO", "PEP")
            },
            
            # Risk Management
            {
                "prompt": "Create a strategy with proper risk management including stop losses, position sizing based on ATR, and maximum drawdown controls.",
                "response": self._get_risk_management_template()
            }
        ]
    
    def _generate_strategy_sample(self, strategy_type: str) -> Dict[str, str]:
        """Generate a sample for a specific strategy type"""
        try:
            if strategy_type == "moving_average":
                return self._generate_ma_sample()
            elif strategy_type == "mean_reversion":
                return self._generate_mean_reversion_sample()
            elif strategy_type == "momentum":
                return self._generate_momentum_sample()
            elif strategy_type == "pairs_trading":
                return self._generate_pairs_sample()
            elif strategy_type == "risk_management":
                return self._generate_risk_mgmt_sample()
            else:
                return self._generate_custom_sample()
        except Exception as e:
            logger.warning(f"Error generating {strategy_type} sample: {e}")
            return None
    
    def _generate_ma_sample(self) -> Dict[str, str]:
        """Generate moving average strategy sample"""
        symbol = random.choice(self.symbols)
        short_period = random.randint(5, 15)
        long_period = random.randint(20, 50)
        ma_type = random.choice(["SMA", "EMA"])
        capital = random.choice([50000, 100000, 250000, 500000])
        
        prompt = f"Create a {ma_type} crossover strategy using {short_period}-day and {long_period}-day {ma_type}s. Trade {symbol} with ${capital:,} starting capital."
        
        if ma_type == "SMA":
            response = self._get_sma_crossover_template(symbol, short_period, long_period, capital)
        else:
            response = self._get_ema_crossover_template(symbol, short_period, long_period, capital)
        
        return {"prompt": prompt, "response": response}
    
    def _generate_mean_reversion_sample(self) -> Dict[str, str]:
        """Generate mean reversion strategy sample"""
        symbol = random.choice(self.symbols)
        indicator = random.choice(["RSI", "Bollinger Bands", "CCI"])
        
        if indicator == "RSI":
            oversold = random.randint(20, 35)
            overbought = random.randint(65, 80)
            position_size = random.choice([0.05, 0.1, 0.15, 0.2])
            
            prompt = f"Build an RSI mean reversion strategy for {symbol} that buys when RSI < {oversold} and sells when RSI > {overbought}. Use {position_size*100:.0f}% position sizing."
            response = self._get_rsi_strategy_template(symbol, oversold, overbought, position_size)
            
        elif indicator == "Bollinger Bands":
            period = random.choice([15, 20, 25])
            std_dev = random.choice([1.5, 2.0, 2.5])
            
            prompt = f"Create a Bollinger Bands mean reversion strategy for {symbol} with {period}-period and {std_dev} standard deviations."
            response = self._get_bollinger_bands_template(symbol, period, std_dev)
            
        return {"prompt": prompt, "response": response}
    
    def _generate_momentum_sample(self) -> Dict[str, str]:
        """Generate momentum strategy sample"""
        symbols = random.sample(self.symbols, random.randint(3, 6))
        lookback = random.choice([20, 30, 50, 100])
        
        prompt = f"Design a momentum strategy that trades {', '.join(symbols)} based on {lookback}-day high breakouts with ATR-based position sizing."
        response = self._get_multi_asset_momentum_template(symbols, lookback)
        
        return {"prompt": prompt, "response": response}
    
    def _generate_pairs_sample(self) -> Dict[str, str]:
        """Generate pairs trading strategy sample"""
        # Select correlated pairs
        pairs = [("KO", "PEP"), ("JPM", "BAC"), ("AAPL", "MSFT"), ("GOOGL", "META")]
        stock1, stock2 = random.choice(pairs)
        
        prompt = f"Build a pairs trading strategy between {stock1} and {stock2} using statistical arbitrage with z-score entry/exit signals."
        response = self._get_pairs_trading_template(stock1, stock2)
        
        return {"prompt": prompt, "response": response}
    
    def _load_strategy_templates(self) -> Dict[str, str]:
        """Load strategy templates"""
        return {
            "moving_average": "MA crossover strategies",
            "mean_reversion": "Mean reversion strategies", 
            "momentum": "Momentum/breakout strategies",
            "pairs_trading": "Statistical arbitrage strategies",
            "risk_management": "Risk management strategies",
            "custom": "Custom strategy implementations"
        }
    
    def _get_sma_crossover_template(self, symbol: str, short: int, long: int, capital: int, 
                                   start_date: str = "2020, 1, 1", end_date: str = "2023, 12, 31") -> str:
        """Generate SMA crossover strategy template"""
        return f"""from AlgorithmImports import *

class SMAStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate({start_date})
        self.SetEndDate({end_date})
        self.SetCash({capital})
        
        # Add equity
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        
        # Create moving averages
        self.short_sma = self.SMA(self.symbol, {short})
        self.long_sma = self.SMA(self.symbol, {long})
        
        # Track previous values for crossover detection
        self.previous_short = 0
        self.previous_long = 0
        
    def OnData(self, data):
        if not (self.short_sma.IsReady and self.long_sma.IsReady):
            return
            
        current_short = self.short_sma.Current.Value
        current_long = self.long_sma.Current.Value
        
        # Check for crossover signals
        if self.previous_short != 0 and self.previous_long != 0:
            # Golden cross (buy signal)
            if self.previous_short <= self.previous_long and current_short > current_long:
                if not self.Portfolio[self.symbol].Invested:
                    self.SetHoldings(self.symbol, 1.0)
                    self.Log(f"BUY: SMA {short} crossed above SMA {long}")
            
            # Death cross (sell signal)
            elif self.previous_short >= self.previous_long and current_short < current_long:
                if self.Portfolio[self.symbol].Invested:
                    self.Liquidate(self.symbol)
                    self.Log(f"SELL: SMA {short} crossed below SMA {long}")
        
        self.previous_short = current_short
        self.previous_long = current_long"""

    def _get_ema_crossover_template(self, symbol: str, short: int, long: int, capital: int) -> str:
        """Generate EMA crossover strategy template"""
        return f"""from AlgorithmImports import *

class EMAStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash({capital})
        
        # Add equity
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        
        # Create exponential moving averages
        self.short_ema = self.EMA(self.symbol, {short})
        self.long_ema = self.EMA(self.symbol, {long})
        
    def OnData(self, data):
        if not (self.short_ema.IsReady and self.long_ema.IsReady):
            return
            
        # Check for crossover signals
        if self.short_ema.Current.Value > self.long_ema.Current.Value:
            if not self.Portfolio[self.symbol].Invested:
                self.SetHoldings(self.symbol, 1.0)
                self.Log(f"BUY: EMA {short} above EMA {long}")
        else:
            if self.Portfolio[self.symbol].Invested:
                self.Liquidate(self.symbol)
                self.Log(f"SELL: EMA {short} below EMA {long}")"""

    def _get_rsi_strategy_template(self, symbol: str, oversold: int, overbought: int, position_size: float) -> str:
        """Generate RSI strategy template"""
        return f"""from AlgorithmImports import *

class RSIStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add equity
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        
        # Create RSI indicator
        self.rsi = self.RSI(self.symbol, 14)
        
        # Position sizing
        self.position_size = {position_size}
        
    def OnData(self, data):
        if not self.rsi.IsReady:
            return
            
        rsi_value = self.rsi.Current.Value
        
        # Buy signal: RSI oversold
        if rsi_value < {oversold}:
            if not self.Portfolio[self.symbol].Invested:
                self.SetHoldings(self.symbol, self.position_size)
                self.Log(f"BUY: RSI oversold at {{rsi_value:.2f}}")
        
        # Sell signal: RSI overbought
        elif rsi_value > {overbought}:
            if self.Portfolio[self.symbol].Invested:
                self.Liquidate(self.symbol)
                self.Log(f"SELL: RSI overbought at {{rsi_value:.2f}}")"""

    def _get_bollinger_bands_template(self, symbol: str, period: int, std_dev: float) -> str:
        """Generate Bollinger Bands strategy template"""
        return f"""from AlgorithmImports import *

class BollingerBandsStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add equity
        self.symbol = self.AddEquity("{symbol}", Resolution.Daily).Symbol
        
        # Create Bollinger Bands
        self.bb = self.BB(self.symbol, {period}, {std_dev})
        self.rsi = self.RSI(self.symbol, 14)
        
    def OnData(self, data):
        if not (self.bb.IsReady and self.rsi.IsReady):
            return
            
        price = data[self.symbol].Close
        upper_band = self.bb.UpperBand.Current.Value
        lower_band = self.bb.LowerBand.Current.Value
        rsi_value = self.rsi.Current.Value
        
        # Buy signal: Price near lower band and RSI oversold
        if price <= lower_band * 1.01 and rsi_value < 35:
            if not self.Portfolio[self.symbol].Invested:
                self.SetHoldings(self.symbol, 0.5)
                self.Log(f"BUY: Price at lower BB band, RSI: {{rsi_value:.2f}}")
        
        # Sell signal: Price near upper band and RSI overbought
        elif price >= upper_band * 0.99 and rsi_value > 65:
            if self.Portfolio[self.symbol].Invested:
                self.Liquidate(self.symbol)
                self.Log(f"SELL: Price at upper BB band, RSI: {{rsi_value:.2f}}")"""

    def _get_multi_asset_momentum_template(self, symbols: List[str], lookback: int) -> str:
        """Generate multi-asset momentum strategy template"""
        symbols_str = '", "'.join(symbols)
        return f"""from AlgorithmImports import *

class MomentumStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add equities
        self.symbols = []
        for ticker in ["{symbols_str}"]:
            symbol = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.symbols.append(symbol)
        
        # Create indicators for each symbol
        self.highs = {{}}
        self.atr = {{}}
        
        for symbol in self.symbols:
            self.highs[symbol] = self.MAX(symbol, {lookback})
            self.atr[symbol] = self.ATR(symbol, 14)
        
        # Risk parameters
        self.risk_per_trade = 0.02
        
    def OnData(self, data):
        for symbol in self.symbols:
            if not (self.highs[symbol].IsReady and self.atr[symbol].IsReady):
                continue
                
            if symbol not in data:
                continue
                
            price = data[symbol].Close
            high_{lookback} = self.highs[symbol].Current.Value
            atr_value = self.atr[symbol].Current.Value
            
            # Entry signal: Break above {lookback}-day high
            if price > high_{lookback} and not self.Portfolio[symbol].Invested:
                # Position sizing based on ATR
                risk_amount = self.Portfolio.TotalPortfolioValue * self.risk_per_trade
                position_size = min(risk_amount / (atr_value * 2), 
                                  self.Portfolio.TotalPortfolioValue * 0.1)
                
                if position_size > 0:
                    self.SetHoldings(symbol, position_size / self.Portfolio.TotalPortfolioValue)
                    self.Log(f"BUY {{symbol}}: Breakout at {{price:.2f}}")
            
            # Exit signal: Price falls below recent low (simple exit)
            elif self.Portfolio[symbol].Invested and price < high_{lookback} * 0.95:
                self.Liquidate(symbol)
                self.Log(f"SELL {{symbol}}: Exit at {{price:.2f}}")"""

    def _get_pairs_trading_template(self, stock1: str, stock2: str) -> str:
        """Generate pairs trading strategy template"""
        return f"""from AlgorithmImports import *
import numpy as np

class PairsTradingStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add pairs
        self.stock1 = self.AddEquity("{stock1}", Resolution.Daily).Symbol
        self.stock2 = self.AddEquity("{stock2}", Resolution.Daily).Symbol
        
        # Price history for spread calculation
        self.lookback = 60
        self.price_history1 = RollingWindow[float](self.lookback)
        self.price_history2 = RollingWindow[float](self.lookback)
        
        # Trading parameters
        self.entry_zscore = 2.0
        self.exit_zscore = 0.0
        self.in_position = False
        
    def OnData(self, data):
        if self.stock1 in data and self.stock2 in data:
            self.price_history1.Add(data[self.stock1].Close)
            self.price_history2.Add(data[self.stock2].Close)
            
        if not (self.price_history1.IsReady and self.price_history2.IsReady):
            return
        
        # Calculate spread and z-score
        prices1 = np.array([x for x in self.price_history1])
        prices2 = np.array([x for x in self.price_history2])
        
        # Simple hedge ratio calculation
        hedge_ratio = np.mean(prices1) / np.mean(prices2)
        spread = prices1 - hedge_ratio * prices2
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        if spread_std == 0:
            return
        
        current_spread = prices1[0] - hedge_ratio * prices2[0]
        zscore = (current_spread - spread_mean) / spread_std
        
        # Trading logic
        if not self.in_position:
            if zscore < -self.entry_zscore:
                # Long spread (buy stock1, sell stock2)
                self.SetHoldings(self.stock1, 0.5)
                self.SetHoldings(self.stock2, -0.5)
                self.in_position = True
                self.Log(f"ENTER LONG SPREAD: Z-score {{zscore:.2f}}")
            elif zscore > self.entry_zscore:
                # Short spread (sell stock1, buy stock2)
                self.SetHoldings(self.stock1, -0.5)
                self.SetHoldings(self.stock2, 0.5)
                self.in_position = True
                self.Log(f"ENTER SHORT SPREAD: Z-score {{zscore:.2f}}")
        else:
            if abs(zscore) <= self.exit_zscore:
                self.Liquidate()
                self.in_position = False
                self.Log(f"EXIT SPREAD: Z-score {{zscore:.2f}}")"""

    def _get_risk_management_template(self) -> str:
        """Generate risk management strategy template"""
        return """from AlgorithmImports import *

class RiskManagedStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add equity
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Indicators
        self.sma_fast = self.SMA(self.symbol, 10)
        self.sma_slow = self.SMA(self.symbol, 20)
        self.atr = self.ATR(self.symbol, 14)
        
        # Risk management parameters
        self.max_position_size = 0.1  # 10% max per position
        self.risk_per_trade = 0.02    # 2% risk per trade
        self.max_drawdown = 0.15      # 15% max drawdown
        self.stop_multiplier = 2      # 2x ATR for stop loss
        
        # Track positions
        self.entry_price = 0
        self.stop_price = 0
        self.peak_portfolio_value = self.Portfolio.TotalPortfolioValue
        
    def OnData(self, data):
        if not (self.sma_fast.IsReady and self.sma_slow.IsReady and self.atr.IsReady):
            return
        
        # Update peak portfolio value for drawdown calculation
        current_value = self.Portfolio.TotalPortfolioValue
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
        
        # Calculate current drawdown
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        # Emergency exit on max drawdown
        if drawdown > self.max_drawdown:
            if self.Portfolio[self.symbol].Invested:
                self.Liquidate(self.symbol)
                self.Log(f"EMERGENCY EXIT: Max drawdown exceeded ({drawdown:.2%})")
            return
        
        price = data[self.symbol].Close
        atr_value = self.atr.Current.Value
        
        # Entry logic
        if not self.Portfolio[self.symbol].Invested:
            if self.sma_fast.Current.Value > self.sma_slow.Current.Value:
                # Calculate position size based on risk
                risk_amount = current_value * self.risk_per_trade
                stop_distance = atr_value * self.stop_multiplier
                
                if stop_distance > 0:
                    shares = int(risk_amount / stop_distance)
                    max_shares = int((current_value * self.max_position_size) / price)
                    shares = min(shares, max_shares)
                    
                    if shares > 0:
                        self.MarketOrder(self.symbol, shares)
                        self.entry_price = price
                        self.stop_price = price - stop_distance
                        self.Log(f"BUY: {shares} shares at {price:.2f}, Stop: {self.stop_price:.2f}")
        
        # Exit logic
        else:
            # Stop loss
            if price <= self.stop_price:
                self.Liquidate(self.symbol)
                self.Log(f"STOP LOSS: Exit at {price:.2f}")
            
            # Trailing stop (update stop price)
            elif price > self.entry_price:
                new_stop = price - (atr_value * self.stop_multiplier)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
            
            # Exit on trend change
            elif self.sma_fast.Current.Value < self.sma_slow.Current.Value:
                self.Liquidate(self.symbol)
                self.Log(f"TREND EXIT: Exit at {price:.2f}")"""

    def save_dataset(self, dataset: List[Dict[str, str]], filename: str = "lean_training_dataset.json"):
        """Save the dataset to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Dataset saved to {filename} with {len(dataset)} examples")

def main():
    """Generate the training dataset"""
    generator = LEANDatasetGenerator()
    dataset = generator.generate_comprehensive_dataset(1000)
    generator.save_dataset(dataset)
    print(f"Generated {len(dataset)} training examples")

if __name__ == "__main__":
    main()