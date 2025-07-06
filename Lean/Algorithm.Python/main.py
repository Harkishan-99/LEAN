from AlgorithmImports import *

class BasicTemplateAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        """Initialize algorithm settings and indicators"""
        # Set the cash we'd like to use for our backtest
        self.SetCash(100000)
        
        # Start and end dates for the backtest
        self.SetStartDate(2015, 3, 1)
        self.SetEndDate(2015, 12, 31)
        
        # Set the timezone
        self.SetTimeZone(TimeZones.NewYork)
        
        # Add equity data - we'll use some of the Indian stocks from our database
        # Note: LEAN expects lowercase ticker symbols for file names
        self.symbols = []
        
        # Add some stocks from our database that have good data coverage
        tickers = ["ABB", "ADANIENT", "ASIANPAINT", "AMBUJACEM"]
        
        for ticker in tickers:
            try:
                # Add equity with daily resolution
                symbol = self.AddEquity(ticker, Resolution.Daily, 
                                      fillDataForward=True, 
                                      leverage=1.0).Symbol
                self.symbols.append(symbol)
                self.Log(f"Added {ticker} to portfolio")
            except Exception as e:
                self.Log(f"Failed to add {ticker}: {str(e)}")
        
        # Create indicators for moving average crossover strategy
        self.sma_short = {}
        self.sma_long = {}
        self.previous_short = {}
        self.previous_long = {}
        
        # Initialize moving averages for each symbol
        for symbol in self.symbols:
            # 10-day and 20-day simple moving averages
            self.sma_short[symbol] = self.SMA(symbol, 10, Resolution.Daily)
            self.sma_long[symbol] = self.SMA(symbol, 20, Resolution.Daily)
            
            # Store previous values to detect crossovers
            self.previous_short[symbol] = 0
            self.previous_long[symbol] = 0
        
        # Track our positions and trades
        self.position_size = 0.2  # Use 20% of portfolio per position
        self.min_price = 1.0      # Minimum price filter
        
        # Schedule function to log portfolio status
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Friday), 
                        self.TimeRules.BeforeMarketClose("ABB", 30), 
                        self.LogPortfolioStatus)
        
    def OnData(self, data):
        """Main algorithm logic executed on each data point"""
        
        # Check if we have data for our symbols
        if not data.Bars:
            return
            
        # Process each symbol
        for symbol in self.symbols:
            
            # Skip if we don't have data for this symbol
            if symbol not in data.Bars:
                continue
                
            # Skip if indicators are not ready
            if not (self.sma_short[symbol].IsReady and self.sma_long[symbol].IsReady):
                continue
                
            # Get current indicator values
            current_short = self.sma_short[symbol].Current.Value
            current_long = self.sma_long[symbol].Current.Value
            current_price = data[symbol].Close
            
            # Skip if price is too low (penny stock filter)
            if current_price < self.min_price:
                continue
            
            # Get previous values
            prev_short = self.previous_short[symbol]
            prev_long = self.previous_long[symbol]
            
            # Detect crossover signals only if we have previous values
            if prev_short > 0 and prev_long > 0:
                
                # Buy signal: short MA crosses above long MA (Golden Cross)
                if prev_short <= prev_long and current_short > current_long:
                    if not self.Portfolio[symbol].Invested:
                        self.ExecuteBuyOrder(symbol, current_price)
                
                # Sell signal: short MA crosses below long MA (Death Cross)
                elif prev_short >= prev_long and current_short < current_long:
                    if self.Portfolio[symbol].Invested:
                        self.ExecuteSellOrder(symbol, current_price)
            
            # Update previous values for next iteration
            self.previous_short[symbol] = current_short
            self.previous_long[symbol] = current_long
    
    def ExecuteBuyOrder(self, symbol, price):
        """Execute a buy order with position sizing"""
        try:
            # Calculate position size (20% of portfolio)
            portfolio_value = self.Portfolio.TotalPortfolioValue
            position_value = portfolio_value * self.position_size
            
            # Calculate quantity to buy
            quantity = int(position_value / price)
            
            # Check if we have enough cash and quantity is reasonable
            if quantity > 0 and self.Portfolio.Cash >= (quantity * price):
                self.MarketOrder(symbol, quantity)
                self.Log(f"BUY {symbol.Value}: {quantity} shares at ${price:.2f} "
                        f"(Total: ${quantity * price:.2f})")
        except Exception as e:
            self.Log(f"Error executing buy order for {symbol.Value}: {str(e)}")
    
    def ExecuteSellOrder(self, symbol, price):
        """Execute a sell order to close position"""
        try:
            # Get current position
            holding = self.Portfolio[symbol]
            if holding.Quantity > 0:
                self.MarketOrder(symbol, -holding.Quantity)
                self.Log(f"SELL {symbol.Value}: {holding.Quantity} shares at ${price:.2f} "
                        f"(Total: ${holding.Quantity * price:.2f})")
        except Exception as e:
            self.Log(f"Error executing sell order for {symbol.Value}: {str(e)}")
    
    def LogPortfolioStatus(self):
        """Log weekly portfolio status"""
        try:
            self.Log(f"=== Portfolio Status ===")
            self.Log(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
            self.Log(f"Cash: ${self.Portfolio.Cash:,.2f}")
            self.Log(f"Holdings Value: ${self.Portfolio.TotalHoldingsValue:,.2f}")
            
            # Log individual positions
            for symbol in self.symbols:
                holding = self.Portfolio[symbol]
                if holding.Invested:
                    self.Log(f"{symbol.Value}: {holding.Quantity} shares, "
                            f"Value: ${holding.HoldingsValue:,.2f}, "
                            f"P&L: ${holding.UnrealizedProfit:,.2f}")
        except Exception as e:
            self.Log(f"Error logging portfolio status: {str(e)}")
    
    def OnEndOfAlgorithm(self):
        """Final algorithm summary"""
        try:
            self.Log("="*50)
            self.Log("BACKTEST COMPLETED")
            self.Log("="*50)
            
            # Log final portfolio value
            final_value = self.Portfolio.TotalPortfolioValue
            initial_value = 100000  # Our starting cash
            total_return = (final_value - initial_value) / initial_value * 100
            
            self.Log(f"Initial Capital: ${initial_value:,.2f}")
            self.Log(f"Final Portfolio Value: ${final_value:,.2f}")
            self.Log(f"Total Return: {total_return:.2f}%")
            self.Log(f"Final Cash: ${self.Portfolio.Cash:,.2f}")
            
            # Log final positions
            self.Log("\nFinal Positions:")
            for symbol in self.symbols:
                holding = self.Portfolio[symbol]
                if holding.Invested:
                    self.Log(f"{symbol.Value}: {holding.Quantity} shares, "
                            f"Value: ${holding.HoldingsValue:,.2f}, "
                            f"P&L: ${holding.UnrealizedProfit:,.2f}")
            
            self.Log("="*50)
            
        except Exception as e:
            self.Log(f"Error in OnEndOfAlgorithm: {str(e)}")
    
    def OnOrderEvent(self, orderEvent):
        """Log order events for debugging"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Order Filled: {orderEvent.Symbol.Value} "
                    f"{orderEvent.Direction} {orderEvent.FillQuantity} shares "
                    f"at ${orderEvent.FillPrice:.2f}")
    
    def OnSecuritiesChanged(self, changes):
        """Handle security additions/removals"""
        for security in changes.AddedSecurities:
            self.Log(f"Added security: {security.Symbol.Value}")
        
        for security in changes.RemovedSecurities:
            self.Log(f"Removed security: {security.Symbol.Value}")