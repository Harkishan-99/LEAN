#!/bin/bash

# LEAN Backtest Runner Script
# This script runs a LEAN backtest using Docker

echo "================================================"
echo "LEAN ALGORITHM BACKTEST RUNNER"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install it and try again."
    exit 1
fi

print_status "Starting LEAN backtest..."

# Clean up any existing containers
print_status "Cleaning up existing containers..."
docker-compose down --remove-orphans 2>/dev/null

# Ensure Results directory exists and is writable
print_status "Preparing results directory..."
mkdir -p Lean/Results
chmod 755 Lean/Results

# Run the backtest
print_status "Running LEAN backtest with Docker..."
echo "Algorithm: BasicTemplateAlgorithm"
echo "Data: Indian Equity Stocks (ABB, ADANIENT, ASIANPAINT, AMBUJACEM)"
echo "Period: 2015-03-01 to 2015-12-31"
echo "Strategy: Moving Average Crossover (10-day vs 20-day SMA)"
echo ""

# Run with sudo if needed
if docker-compose up --build 2>&1 | tee lean_backtest.log; then
    print_status "Backtest completed successfully!"
else
    print_warning "Backtest completed with warnings or Docker permission issues."
    print_status "Trying with sudo privileges..."
    sudo docker-compose up --build 2>&1 | tee lean_backtest_sudo.log
fi

# Check if results were generated
if [ -d "Lean/Results" ] && [ "$(ls -A Lean/Results)" ]; then
    print_status "Results generated in Lean/Results/"
    echo ""
    echo "Generated files:"
    ls -la Lean/Results/
    echo ""
    
    # Show a summary of the log
    if [ -f "lean_backtest.log" ]; then
        print_status "Backtest Summary:"
        echo "=================="
        # Extract key information from the log
        grep -E "(Algorithm|Final Portfolio|Total Return|ERROR|Exception)" lean_backtest.log | tail -10
    fi
else
    print_warning "No results found in Lean/Results/"
    print_warning "Check the log files for errors:"
    echo "  - lean_backtest.log"
    echo "  - lean_backtest_sudo.log (if created)"
fi

# Clean up containers
print_status "Cleaning up Docker containers..."
docker-compose down --remove-orphans 2>/dev/null

echo ""
echo "================================================"
echo "BACKTEST COMPLETED"
echo "================================================"
echo "Check the following files:"
echo "  - Lean/Results/ (for backtest results)"
echo "  - lean_backtest.log (for full backtest log)"
echo "  - LEAN_DOCKER_GUIDE.md (for documentation)"