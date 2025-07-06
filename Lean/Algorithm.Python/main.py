from AlgorithmImports import *

class BasicTemplateAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        # Set the cash we'd like to use for our backtest
        self.SetCash(100000)
        
        # Start and end dates for the backtest
        self.SetStartDate(2015, 3, 1)
        self.SetEndDate(2015, 12, 31)
        
        # Add equity data - we'll use some of the Indian stocks from your data
        # Note: LEAN expects lowercase ticker symbols for file names
        self.symbols = []
        
        # Add some stocks from your database
        tickers = ["ABB", "ADANIENSOL", "ADANIENT", "ASIANPAINT", "AMBUJACEM"]
        
        for ticker in tickers:
            try:
                symbol = self.AddEquity(ticker, Resolution.Daily).Symbol
                self.symbols.append(symbol)
                self.Log(f"Added {ticker} to portfolio")
            except Exception as e:
                self.Log(f"Failed to add {ticker}: {str(e)}")
        
        # Create indicators for moving average crossover strategy
        self.sma_short = {}
        self.sma_long = {}
        
        for symbol in self.symbols:
            self.sma_short[symbol] = self.SMA(symbol, 10, Resolution.Daily)
            self.sma_long[symbol] = self.SMA(symbol, 20, Resolution.Daily)
        
        # Track our portfolio
        self.previous_positions = {}
        
    def OnData(self, data):
        # Check if we have data for all symbols
        if not all(data.ContainsKey(symbol) for symbol in self.symbols):
            return
            
        # Implement simple moving average crossover strategy
        for symbol in self.symbols:
            # Skip if indicators are not ready
            if not (self.sma_short[symbol].IsReady and self.sma_long[symbol].IsReady):
                continue
                
            # Get current indicator values
            short_ma = self.sma_short[symbol].Current.Value
            long_ma = self.sma_long[symbol].Current.Value
            
            # Get current price
            price = data[symbol].Close
            
            # Check if we have enough cash for this trade
            available_cash = self.Portfolio.Cash
            position_size = available_cash * 0.2  # Use 20% of available cash per position
            
            # Buy signal: short MA crosses above long MA
            if short_ma > long_ma and not self.Portfolio[symbol].Invested:
                if available_cash > position_size:
                    quantity = int(position_size / price)
                    if quantity > 0:
                        self.MarketOrder(symbol, quantity)
                        self.Log(f"BUY {symbol.Value}: {quantity} shares at {price}")
            
            # Sell signal: short MA crosses below long MA
            elif short_ma < long_ma and self.Portfolio[symbol].Invested:
                quantity = self.Portfolio[symbol].Quantity
                self.MarketOrder(symbol, -quantity)
                self.Log(f"SELL {symbol.Value}: {quantity} shares at {price}")
    
    def OnEndOfAlgorithm(self):
        # Log final portfolio value
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        
        # Log individual positions
        for symbol in self.symbols:
            if self.Portfolio[symbol].Invested:
                holding = self.Portfolio[symbol]
                self.Log(f"{symbol.Value}: {holding.Quantity} shares, Value: ${holding.HoldingsValue:,.2f}")
        
        # Calculate and log performance metrics
        total_return = (self.Portfolio.TotalPortfolioValue - self.InitialCash) / self.InitialCash * 100
        self.Log(f"Total Return: {total_return:.2f}%")