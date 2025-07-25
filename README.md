# LEAN Algorithmic Trading Backtest Setup

This repository contains a complete setup for running algorithmic trading backtests using QuantConnect's LEAN engine with Docker.

## 🚀 Quick Start

```bash
# 1. Run the backtest
./run_lean_backtest.sh

# 2. View results
ls -la Lean/Results/
cat lean_backtest.log
```

## 📊 What's Included

- **✅ Complete LEAN Docker Environment** - Pre-configured for backtesting
- **✅ Indian Equity Market Data** - 101 stocks from 2015-2025
- **✅ Sample Moving Average Strategy** - Professional algorithm example
- **✅ Automated Scripts** - One-command backtesting
- **✅ Comprehensive Documentation** - Complete setup guide

## 📁 Project Structure

```
/workspace/
├── 📁 Lean/
│   ├── 📁 Algorithm.Python/
│   │   └── main.py                 # Trading algorithm (Moving Average Crossover)
│   ├── 📁 Data/equity/usa/daily/   # Market data (101 CSV files)
│   └── 📁 Results/                 # Generated backtest results
├── ⚙️ config.json                  # LEAN configuration
├── 🐳 docker-compose.yml           # Docker setup
├── 🔧 run_lean_backtest.sh         # Automated runner script
├── 📊 data_converter.py            # Data preparation utility
├── 💾 database.db                  # Source market data (26MB)
└── 📖 LEAN_DOCKER_GUIDE.md         # Complete documentation (200+ lines)
```

## 🎯 Sample Strategy

The included algorithm implements a **Moving Average Crossover** strategy:

- **Indicators:** 10-day and 20-day Simple Moving Averages
- **Buy Signal:** Short MA crosses above Long MA (Golden Cross)
- **Sell Signal:** Short MA crosses below Long MA (Death Cross)
- **Universe:** ABB, ADANIENT, ASIANPAINT, AMBUJACEM
- **Period:** March 2015 - December 2015
- **Initial Capital:** $100,000

## 📈 Expected Results

The strategy generates:
- Detailed trade logs
- Performance statistics (returns, volatility, drawdown)
- JSON result files for analysis
- Comprehensive backtest logs

## 🛠 Technologies Used

- **LEAN Engine** - QuantConnect's algorithmic trading framework
- **Docker** - Containerized execution environment
- **Python** - Algorithm development language
- **SQLite** - Historical market data storage

## 📖 Documentation

See `LEAN_DOCKER_GUIDE.md` for comprehensive documentation including:

- Prerequisites and installation
- Detailed configuration options
- Algorithm development guide
- Troubleshooting and debugging
- Advanced features and best practices

## 🔧 Requirements

- Docker Engine (20.0+)
- Docker Compose (2.0+)
- Linux/macOS/Windows with WSL2
- 4GB+ RAM recommended

## 🚦 Getting Started

1. **Ensure Docker is running:**
   ```bash
   docker --version
   docker-compose --version
   ```

2. **Run the backtest:**
   ```bash
   chmod +x run_lean_backtest.sh
   ./run_lean_backtest.sh
   ```

3. **View results:**
   ```bash
   ls -la Lean/Results/
   grep -E "(BUY|SELL|Final Portfolio)" lean_backtest.log
   ```

## 🎓 Learning Resources

- [LEAN Documentation](https://www.lean.io/docs/)
- [QuantConnect Forum](https://www.quantconnect.com/forum)
- [Algorithm Examples](https://github.com/QuantConnect/Lean/tree/master/Algorithm.Python)

## 📝 Notes

- The setup includes real Indian equity market data (2015-2025)
- Data is automatically converted to LEAN format
- Docker handles all dependencies and environment setup
- Results are saved in JSON format for easy analysis

---

**Ready to run your first algorithmic trading backtest? Execute `./run_lean_backtest.sh` and watch your strategy come to life!**