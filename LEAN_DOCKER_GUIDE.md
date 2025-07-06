# LEAN Algorithm Backtesting with Docker - Complete Guide

This guide provides comprehensive instructions for running algorithmic trading backtests using QuantConnect's LEAN engine with Docker.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Detailed Setup](#detailed-setup)
6. [Configuration](#configuration)
7. [Running Backtests](#running-backtests)
8. [Understanding Results](#understanding-results)
9. [Algorithm Development](#algorithm-development)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)

---

## Overview

**LEAN** (Live Engine ANT) is QuantConnect's open-source algorithmic trading engine. This setup allows you to:

- Run backtests locally using Docker
- Test trading strategies on historical data
- Generate comprehensive performance reports
- Develop and debug algorithms in Python
- Use real market data for accurate simulations

**Key Features:**
- ✅ Dockerized LEAN engine for consistent environment
- ✅ Pre-configured with Indian equity market data
- ✅ Moving Average Crossover strategy example
- ✅ Automated result generation
- ✅ Easy-to-use scripts and configuration

---

## Prerequisites

### System Requirements
- **Operating System:** Linux, macOS, or Windows with WSL2
- **RAM:** Minimum 4GB, recommended 8GB+
- **Storage:** At least 2GB free space
- **CPU:** Modern multi-core processor

### Required Software
1. **Docker Engine** (version 20.0+)
2. **Docker Compose** (version 2.0+)
3. **Bash shell** (for running scripts)

### Installation Commands

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group (optional, to avoid sudo)
sudo usermod -aG docker $USER
```

**Verification:**
```bash
docker --version
docker-compose --version
docker info
```

---

## Project Structure

```
/workspace/
├── Lean/
│   ├── Algorithm.Python/
│   │   └── main.py                 # Your trading algorithm
│   ├── Data/
│   │   └── equity/usa/daily/       # Market data (CSV files)
│   │       ├── abb.csv
│   │       ├── adanient.csv
│   │       ├── asianpaint.csv
│   │       └── ambujacem.csv
│   └── Results/                    # Generated backtest results
├── config.json                     # LEAN configuration
├── docker-compose.yml              # Docker setup
├── run_lean_backtest.sh            # Automated runner script
├── data_converter.py               # Data preparation utility
└── database.db                     # Source market data
```

---

## Quick Start

### 1. Run Your First Backtest

```bash
# Make the script executable
chmod +x run_lean_backtest.sh

# Run the backtest
./run_lean_backtest.sh
```

### 2. View Results

After the backtest completes, check:
```bash
# List generated files
ls -la Lean/Results/

# View backtest log
cat lean_backtest.log

# Check algorithm output
grep -E "(BUY|SELL|Final Portfolio)" lean_backtest.log
```

---

## Detailed Setup

### Step 1: Prepare Market Data

The project includes a data converter that transforms SQLite data into LEAN format:

```bash
# Convert database to LEAN CSV format
python3 data_converter.py
```

**Output:** CSV files in `Lean/Data/equity/usa/daily/` with format:
```csv
Date,Open,High,Low,Close,Volume
20150303,1215.4389,1232.7405,1195.8448,1201.8138,64532
```

### Step 2: Configure LEAN

Edit `config.json` for your requirements:

```json
{
  "environment": "backtesting",
  "algorithm-type-name": "BasicTemplateAlgorithm",
  "algorithm-language": "Python",
  "algorithm-location": "Algorithm.Python/main.py",
  "data-folder": "/Lean/Data/",
  "results-destination-folder": "/Results/"
}
```

### Step 3: Set Up Docker

The `docker-compose.yml` defines the LEAN environment:

```yaml
services:
  lean-engine:
    image: quantconnect/lean:latest
    container_name: lean-backtest
    volumes:
      - ./Lean/Algorithm.Python:/Lean/Algorithm.Python:ro
      - ./Lean/Data:/Lean/Data:ro
      - ./Lean/Results:/Results
      - ./config.json:/Lean/config.json:ro
    working_dir: /Lean
    command: dotnet QuantConnect.Lean.Launcher.dll
```

---

## Configuration

### Algorithm Parameters

In `Lean/Algorithm.Python/main.py`, modify these key settings:

```python
def Initialize(self):
    # Backtest period
    self.SetStartDate(2015, 3, 1)
    self.SetEndDate(2015, 12, 31)
    
    # Initial capital
    self.SetCash(100000)
    
    # Stock universe
    tickers = ["ABB", "ADANIENT", "ASIANPAINT", "AMBUJACEM"]
    
    # Strategy parameters
    self.position_size = 0.2  # 20% per position
    self.sma_short = 10       # Short moving average
    self.sma_long = 20        # Long moving average
```

### LEAN Configuration Options

Key `config.json` parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `environment` | Execution mode | `"backtesting"` |
| `algorithm-language` | Programming language | `"Python"` |
| `log-level` | Logging verbosity | `"trace"` |
| `show-missing-data-logs` | Show data gaps | `true` |
| `force-exchange-always-open` | Ignore market hours | `false` |

---

## Running Backtests

### Method 1: Using the Automated Script (Recommended)

```bash
./run_lean_backtest.sh
```

**Features:**
- ✅ Automatic Docker checks
- ✅ Results validation
- ✅ Log parsing
- ✅ Error handling
- ✅ Cleanup after completion

### Method 2: Manual Docker Commands

```bash
# Clean up existing containers
docker-compose down --remove-orphans

# Run the backtest
docker-compose up --build

# Alternative with logging
docker-compose up --build 2>&1 | tee backtest.log

# Clean up
docker-compose down
```

### Method 3: Direct Docker Commands

```bash
# Build and run LEAN container
docker run -v $(pwd)/Lean/Algorithm.Python:/Lean/Algorithm.Python:ro \
           -v $(pwd)/Lean/Data:/Lean/Data:ro \
           -v $(pwd)/Lean/Results:/Results \
           -v $(pwd)/config.json:/Lean/config.json:ro \
           -w /Lean \
           quantconnect/lean:latest \
           dotnet QuantConnect.Lean.Launcher.dll
```

---

## Understanding Results

### Generated Files

After a successful backtest, you'll find:

```
Lean/Results/
├── algorithm.json              # Algorithm metadata
├── alpha-results.json         # Alpha model results
├── benchmark.json             # Benchmark performance
├── drawdown.json             # Drawdown analysis
├── statistics.json           # Performance statistics
├── trades.json              # Individual trade records
├── unrealized.json          # Unrealized P&L
└── rolling-window.json      # Rolling performance metrics
```

### Key Metrics

**Performance Statistics (`statistics.json`):**
```json
{
  "Total Return": "5.23%",
  "Annual Return": "5.23%",
  "Annual Volatility": "12.45%",
  "Sharpe Ratio": "0.42",
  "Maximum Drawdown": "8.17%",
  "Compounding Annual Return": "5.180%"
}
```

**Trade Analysis:**
```bash
# View trade summary
cat Lean/Results/trades.json | jq '.[] | {Symbol, Direction, Quantity, Price}'

# Count winning/losing trades
grep -o '"Direction":"[^"]*"' Lean/Results/trades.json | sort | uniq -c
```

### Interpreting Logs

**Key log sections to monitor:**

1. **Algorithm Initialization:**
```
Algorithm.Initialize(): Added ABB to portfolio
Algorithm.Initialize(): Added ADANIENT to portfolio
```

2. **Trade Execution:**
```
BUY ABB: 18 shares at $1090.09 (Total: $19621.60)
SELL ABB: 18 shares at $1105.10 (Total: $19891.76)
```

3. **Final Results:**
```
Final Portfolio Value: $105,234.67
Total Return: 5.23%
```

---

## Algorithm Development

### Basic Strategy Template

```python
from AlgorithmImports import *

class MyStrategy(QCAlgorithm):
    
    def Initialize(self):
        """Algorithm initialization"""
        self.SetCash(100000)
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2020, 12, 31)
        
        # Add securities
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Create indicators
        self.sma = self.SMA(self.spy, 20)
        
    def OnData(self, data):
        """Main trading logic"""
        if not self.sma.IsReady:
            return
            
        price = data[self.spy].Close
        
        if price > self.sma.Current.Value:
            self.SetHoldings(self.spy, 1.0)  # Go long
        else:
            self.Liquidate(self.spy)         # Exit position
```

### Advanced Features

**Multiple Timeframes:**
```python
# Add different resolutions
self.daily_spy = self.AddEquity("SPY", Resolution.Daily).Symbol
self.hourly_spy = self.AddEquity("SPY", Resolution.Hour).Symbol
```

**Custom Indicators:**
```python
# Create custom indicator
self.rsi = self.RSI(self.spy, 14)
self.bb = self.BB(self.spy, 20, 2)
```

**Risk Management:**
```python
def OnData(self, data):
    # Position sizing based on volatility
    volatility = self.Securities[self.spy].VolatilityModel.Volatility
    position_size = min(0.1 / volatility, 0.5)  # Risk-adjusted sizing
```

**Order Management:**
```python
# Limit orders
self.LimitOrder(self.spy, 100, data[self.spy].Close * 0.99)

# Stop losses
self.StopMarketOrder(self.spy, -100, data[self.spy].Close * 0.95)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Docker Permission Errors
```bash
# Error: permission denied while connecting to Docker daemon
# Solution: Add user to docker group or use sudo
sudo docker-compose up --build
```

#### 2. Data Not Found
```bash
# Error: No data found for symbol
# Solution: Check data file format and location
ls -la Lean/Data/equity/usa/daily/
head -5 Lean/Data/equity/usa/daily/abb.csv
```

#### 3. Algorithm Not Loading
```python
# Error: Algorithm assembly not found
# Solution: Check algorithm file syntax
python3 -m py_compile Lean/Algorithm.Python/main.py
```

#### 4. Memory Issues
```yaml
# Add memory limits to docker-compose.yml
services:
  lean-engine:
    mem_limit: 2g
    memswap_limit: 2g
```

#### 5. Time Zone Issues
```python
# Set correct timezone in algorithm
self.SetTimeZone(TimeZones.NewYork)  # For US markets
self.SetTimeZone(TimeZones.Calcutta) # For Indian markets
```

### Debugging Techniques

**Enable Verbose Logging:**
```json
{
  "log-level": "trace",
  "debug-mode": true,
  "debugging": true
}
```

**Check Container Logs:**
```bash
# View live logs
docker-compose logs -f lean-engine

# Export logs to file
docker-compose logs lean-engine > debug.log
```

**Validate Data Files:**
```bash
# Check CSV format
head -10 Lean/Data/equity/usa/daily/abb.csv

# Count data points
wc -l Lean/Data/equity/usa/daily/*.csv
```

---

## Advanced Topics

### Custom Data Sources

**Adding New Market Data:**
```python
# In your algorithm
self.AddData(MyCustomData, "SYMBOL", Resolution.Daily)

class MyCustomData(BaseData):
    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource("path/to/data", SubscriptionTransportMedium.LocalFile)
```

### Performance Optimization

**Optimize Data Loading:**
```json
{
  "show-missing-data-logs": false,
  "force-exchange-always-open": true,
  "maximum-data-points-per-chart-series": 1000
}
```

**Memory Management:**
```python
# In algorithm Initialize()
self.SetWarmup(TimeSpan.FromDays(30))  # Reduce warmup period
```

### Multi-Asset Strategies

```python
def Initialize(self):
    # Add multiple asset classes
    self.stocks = ["SPY", "QQQ", "IWM"]
    self.bonds = ["TLT", "IEF"]
    self.commodities = ["GLD", "SLV"]
    
    for symbol in self.stocks + self.bonds + self.commodities:
        self.AddEquity(symbol, Resolution.Daily)
```

### Live Trading Preparation

**Configuration for Live Trading:**
```json
{
  "environment": "live-paper",
  "live-mode-brokerage": "InteractiveBrokersBrokerage",
  "ib-account": "your-account",
  "ib-user-name": "your-username",
  "ib-password": "your-password"
}
```

---

## Best Practices

### 1. Data Quality
- ✅ Validate data completeness before backtesting
- ✅ Handle missing data appropriately
- ✅ Use corporate actions (splits, dividends)
- ✅ Account for survivorship bias

### 2. Algorithm Design
- ✅ Keep strategies simple and interpretable
- ✅ Implement proper risk management
- ✅ Use realistic transaction costs
- ✅ Test on out-of-sample data

### 3. Backtesting Hygiene
- ✅ Avoid look-ahead bias
- ✅ Use walk-forward analysis
- ✅ Test multiple time periods
- ✅ Consider regime changes

### 4. Performance Analysis
- ✅ Focus on risk-adjusted returns
- ✅ Analyze drawdown patterns
- ✅ Consider transaction costs
- ✅ Validate with paper trading

---

## Resources

### Documentation
- [LEAN Documentation](https://www.lean.io/docs/)
- [QuantConnect API Reference](https://www.quantconnect.com/docs/v2)
- [Algorithm Examples](https://github.com/QuantConnect/Lean/tree/master/Algorithm.Python)

### Community
- [QuantConnect Forum](https://www.quantconnect.com/forum)
- [LEAN GitHub Repository](https://github.com/QuantConnect/Lean)
- [Discord Community](https://discord.gg/quantconnect)

### Books
- "Algorithmic Trading" by Ernie Chan
- "Quantitative Trading" by Ernie Chan
- "Machine Learning for Asset Managers" by Marcos López de Prado

---

## Appendix

### A. Sample Configuration Files

**Minimal config.json:**
```json
{
  "environment": "backtesting",
  "algorithm-type-name": "BasicTemplateAlgorithm",
  "algorithm-language": "Python",
  "algorithm-location": "Algorithm.Python/main.py",
  "data-folder": "/Lean/Data/",
  "results-destination-folder": "/Results/"
}
```

**Production docker-compose.yml:**
```yaml
version: '3.8'
services:
  lean-engine:
    image: quantconnect/lean:latest
    container_name: lean-backtest
    volumes:
      - ./Lean/Algorithm.Python:/Lean/Algorithm.Python:ro
      - ./Lean/Data:/Lean/Data:ro
      - ./Lean/Results:/Results
      - ./config.json:/Lean/config.json:ro
    working_dir: /Lean
    command: dotnet QuantConnect.Lean.Launcher.dll
    mem_limit: 4g
    environment:
      - LEAN_CONFIG_FILE=/Lean/config.json
```

### B. Useful Commands

**Docker Management:**
```bash
# Remove all containers
docker rm $(docker ps -aq)

# Remove unused images
docker image prune -f

# View resource usage
docker stats lean-backtest
```

**Log Analysis:**
```bash
# Extract performance metrics
grep -E "(Total Return|Sharpe|Drawdown)" lean_backtest.log

# Count trades
grep -c "Order Filled" lean_backtest.log

# Find errors
grep -i "error\|exception\|failed" lean_backtest.log
```

---

*This guide provides a comprehensive foundation for running LEAN backtests with Docker. For specific questions or advanced use cases, refer to the official LEAN documentation or the QuantConnect community.*