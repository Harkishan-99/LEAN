#!/usr/bin/env python3
"""
Standalone backtesting engine that replicates the LEAN algorithm logic
without requiring Docker or the full LEAN framework.
"""

import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    def __init__(self, initial_cash: float = 100000, start_date: str = "2015-03-01", end_date: str = "2015-12-31"):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.positions = {}
        self.portfolio_value = []
        self.trades = []
        self.data = {}
        self.indicators = {}
        
    def load_data(self, tickers: List[str], db_path: str = "database.db"):
        """Load stock data from SQLite database"""
        conn = sqlite3.connect(db_path)
        
        for ticker in tickers:
            query = """
            SELECT date, open, high, low, close, volume 
            FROM stock_data 
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
            """
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(ticker, self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))
            )
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                self.data[ticker] = df
                print(f"Loaded {len(df)} records for {ticker}")
        
        conn.close()
        
    def calculate_indicators(self, ticker: str, short_window: int = 10, long_window: int = 20):
        """Calculate moving averages for the given ticker"""
        if ticker not in self.data:
            return
            
        df = self.data[ticker]
        df['SMA_short'] = df['close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=long_window).mean()
        
        self.indicators[ticker] = {
            'short_window': short_window,
            'long_window': long_window
        }
        
    def get_portfolio_value(self, current_date: pd.Timestamp) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for ticker, position in self.positions.items():
            if position['quantity'] > 0 and ticker in self.data:
                # Get current price
                ticker_data = self.data[ticker]
                current_data = ticker_data[ticker_data.index <= current_date]
                if not current_data.empty:
                    current_price = current_data.iloc[-1]['close']
                    total_value += position['quantity'] * current_price
                    
        return total_value
        
    def execute_trade(self, ticker: str, quantity: int, price: float, date: pd.Timestamp):
        """Execute a trade"""
        if ticker not in self.positions:
            self.positions[ticker] = {'quantity': 0, 'avg_price': 0}
            
        if quantity > 0:  # Buy
            cost = quantity * price
            if cost <= self.cash:
                self.cash -= cost
                old_quantity = self.positions[ticker]['quantity']
                old_avg_price = self.positions[ticker]['avg_price']
                
                new_quantity = old_quantity + quantity
                new_avg_price = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
                
                self.positions[ticker]['quantity'] = new_quantity
                self.positions[ticker]['avg_price'] = new_avg_price
                
                self.trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'value': cost
                })
                return True
        else:  # Sell
            quantity = abs(quantity)
            if self.positions[ticker]['quantity'] >= quantity:
                proceeds = quantity * price
                self.cash += proceeds
                self.positions[ticker]['quantity'] -= quantity
                
                self.trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'value': proceeds
                })
                return True
                
        return False
        
    def run_backtest(self, tickers: List[str]):
        """Run the moving average crossover backtest"""
        print("Starting backtest...")
        print(f"Initial Cash: ${self.initial_cash:,.2f}")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Tickers: {', '.join(tickers)}")
        print("-" * 50)
        
        # Load data and calculate indicators
        self.load_data(tickers)
        for ticker in tickers:
            if ticker in self.data:
                self.calculate_indicators(ticker)
        
        # Get all trading dates
        all_dates = set()
        for ticker in tickers:
            if ticker in self.data:
                all_dates.update(self.data[ticker].index)
        
        all_dates = sorted(list(all_dates))
        
        # Run backtest day by day
        for date in all_dates:
            for ticker in tickers:
                if ticker not in self.data:
                    continue
                    
                ticker_data = self.data[ticker]
                current_data = ticker_data[ticker_data.index <= date]
                
                if len(current_data) < 20:  # Not enough data for indicators
                    continue
                    
                current_row = current_data.iloc[-1]
                current_price = current_row['close']
                
                # Check if indicators are ready
                if pd.isna(current_row['SMA_short']) or pd.isna(current_row['SMA_long']):
                    continue
                    
                short_ma = current_row['SMA_short']
                long_ma = current_row['SMA_long']
                
                # Trading logic: Moving Average Crossover
                position_size = self.cash * 0.2  # Use 20% of available cash per position
                
                # Buy signal: short MA crosses above long MA
                if short_ma > long_ma and ticker not in self.positions or self.positions.get(ticker, {}).get('quantity', 0) == 0:
                    if self.cash > position_size:
                        quantity = int(position_size / current_price)
                        if quantity > 0:
                            if self.execute_trade(ticker, quantity, current_price, date):
                                print(f"{date.strftime('%Y-%m-%d')} BUY {ticker}: {quantity} shares at ${current_price:.2f}")
                
                # Sell signal: short MA crosses below long MA
                elif short_ma < long_ma and ticker in self.positions and self.positions[ticker]['quantity'] > 0:
                    quantity = self.positions[ticker]['quantity']
                    if self.execute_trade(ticker, -quantity, current_price, date):
                        print(f"{date.strftime('%Y-%m-%d')} SELL {ticker}: {quantity} shares at ${current_price:.2f}")
            
            # Record portfolio value
            portfolio_val = self.get_portfolio_value(date)
            self.portfolio_value.append({
                'date': date,
                'portfolio_value': portfolio_val,
                'cash': self.cash
            })
        
        self.generate_report()
        
    def generate_report(self):
        """Generate backtest results report"""
        if not self.portfolio_value:
            print("No portfolio data available")
            return
            
        final_value = self.portfolio_value[-1]['portfolio_value']
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        # Calculate additional metrics
        portfolio_df = pd.DataFrame(self.portfolio_value)
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Risk metrics
        volatility = portfolio_df['returns'].std() * np.sqrt(252) * 100  # Annualized
        max_drawdown = self.calculate_max_drawdown(portfolio_df['portfolio_value'])
        
        # Win/Loss analysis
        winning_trades = [t for t in self.trades if t['action'] == 'SELL']
        total_trades = len(winning_trades)
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS REPORT")
        print("="*60)
        print(f"Initial Capital: ${self.initial_cash:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Volatility: {volatility:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Final Cash: ${self.cash:,.2f}")
        
        # Position summary
        print("\nFinal Positions:")
        print("-" * 30)
        for ticker, position in self.positions.items():
            if position['quantity'] > 0:
                current_price = self.data[ticker].iloc[-1]['close']
                market_value = position['quantity'] * current_price
                print(f"{ticker}: {position['quantity']} shares @ ${current_price:.2f} = ${market_value:,.2f}")
        
        # Trade summary
        if self.trades:
            print(f"\nTrade Summary:")
            print("-" * 30)
            trades_df = pd.DataFrame(self.trades)
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            print(f"Buy Trades: {len(buy_trades)}")
            print(f"Sell Trades: {len(sell_trades)}")
            print(f"Total Volume Traded: ${trades_df['value'].sum():,.2f}")
        
        # Save results
        self.save_results()
        
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak * 100
        return abs(drawdown.min())
        
    def save_results(self):
        """Save backtest results to files"""
        os.makedirs("Lean/Results", exist_ok=True)
        
        # Save portfolio values
        portfolio_df = pd.DataFrame(self.portfolio_value)
        portfolio_df.to_csv("Lean/Results/portfolio_values.csv", index=False)
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv("Lean/Results/trades.csv", index=False)
        
        # Save summary report
        final_value = self.portfolio_value[-1]['portfolio_value']
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        summary = {
            "backtest_period": {
                "start_date": self.start_date.strftime('%Y-%m-%d'),
                "end_date": self.end_date.strftime('%Y-%m-%d')
            },
            "performance": {
                "initial_capital": self.initial_cash,
                "final_portfolio_value": final_value,
                "total_return_percent": total_return,
                "final_cash": self.cash
            },
            "positions": {ticker: pos for ticker, pos in self.positions.items() if pos['quantity'] > 0},
            "trade_count": len(self.trades),
            "generated_at": datetime.now().isoformat()
        }
        
        with open("Lean/Results/backtest_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults saved to Lean/Results/")
        print("- portfolio_values.csv")
        print("- trades.csv") 
        print("- backtest_summary.json")

def main():
    """Main function to run the backtest"""
    # Algorithm parameters (matching the LEAN algorithm)
    tickers = ["ABB", "ADANIENT", "ASIANPAINT", "AMBUJACEM"]  # Using available tickers
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_cash=100000,
        start_date="2015-03-01",
        end_date="2015-12-31"
    )
    
    # Run backtest
    engine.run_backtest(tickers)

if __name__ == "__main__":
    main()