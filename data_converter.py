#!/usr/bin/env python3
"""
Convert SQLite stock data to LEAN format CSV files.
LEAN expects CSV files with format: Date,Open,High,Low,Close,Volume
"""
import sqlite3
import os
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_sqlite_to_lean_format(db_path, output_dir):
    """
    Convert SQLite database to LEAN format CSV files.
    
    Args:
        db_path: Path to SQLite database
        output_dir: Directory to save LEAN format CSV files
    """
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Get all unique tickers
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM stock_data ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        
        logging.info(f"Found {len(tickers)} tickers in database")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert each ticker
        for ticker in tickers:
            try:
                # Query data for ticker
                query = """
                SELECT date, open, high, low, close, volume 
                FROM stock_data 
                WHERE ticker = ? 
                ORDER BY date ASC
                """
                
                df = pd.read_sql_query(query, conn, params=(ticker,))
                
                # Convert date format
                df['date'] = pd.to_datetime(df['date'])
                df['date'] = df['date'].dt.strftime('%Y%m%d')
                
                # Rename columns to match LEAN format
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                # Round prices to 4 decimal places
                price_columns = ['Open', 'High', 'Low', 'Close']
                df[price_columns] = df[price_columns].round(4)
                
                # Ensure volume is integer
                df['Volume'] = df['Volume'].astype(int)
                
                # Save to CSV
                output_file = os.path.join(output_dir, f"{ticker.lower()}.csv")
                df.to_csv(output_file, index=False)
                
                logging.info(f"Converted {ticker}: {len(df)} records -> {output_file}")
                
            except Exception as e:
                logging.error(f"Error converting {ticker}: {str(e)}")
                continue
        
        conn.close()
        logging.info("Data conversion completed successfully")
        
    except Exception as e:
        logging.error(f"Error in data conversion: {str(e)}")

def get_ticker_info(db_path):
    """Get information about available tickers and date ranges."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Get ticker info
        query = """
        SELECT ticker, 
               COUNT(*) as record_count,
               MIN(date) as start_date,
               MAX(date) as end_date,
               AVG(close) as avg_price
        FROM stock_data 
        GROUP BY ticker 
        ORDER BY ticker
        """
        
        df = pd.read_sql_query(query, conn)
        print("\nTicker Information:")
        print("==================")
        print(df.to_string(index=False))
        
        conn.close()
        
    except Exception as e:
        logging.error(f"Error getting ticker info: {str(e)}")

if __name__ == "__main__":
    db_path = "database.db"
    output_dir = "Lean/Data/equity/usa/daily"
    
    # Get ticker information
    get_ticker_info(db_path)
    
    # Convert data
    convert_sqlite_to_lean_format(db_path, output_dir)