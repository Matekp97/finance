import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def download_stock_data(ticker, start_date=None, end_date=None, period='1y'):
    """
    Download stock data from Yahoo Finance.

    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date for download
        end_date: End date for download
        period: Period string (e.g., '1y', '2y') if dates not provided

    Returns:
        DataFrame with stock data
    """
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            data = yf.download(ticker, period=period, progress=False)

        # Fix MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None


def download_multiple_stocks(tickers, start_date=None, end_date=None, period='1y'):
    """
    Download data for multiple stock tickers.

    Args:
        tickers: List of stock symbols
        start_date: Start date for download
        end_date: End date for download
        period: Period string if dates not provided

    Returns:
        Dictionary with ticker as key and DataFrame as value
    """
    stocks_data = {}

    for ticker in tickers:
        print(f"⬇️  Downloading {ticker}...", end=" ")
        data = download_stock_data(ticker, start_date, end_date, period)

        if data is not None and not data.empty:
            stocks_data[ticker] = data
            print(f"✅ {len(data)} days downloaded")
        else:
            print(f"❌ Failed")

    return stocks_data


def prepare_stock_data(ticker, period='1y'):
    """
    Download and prepare stock data with additional features.

    Args:
        ticker: Stock symbol
        period: Time period (e.g., '1y', '2y', '6mo')

    Returns:
        DataFrame with original data plus calculated features
    """
    print(f"📥 Downloading {ticker}...")
    df = download_stock_data(ticker, period=period)

    if df is None or df.empty:
        return None

    # Remove any rows with missing values
    df = df.dropna()

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Daily range percentage
    df['Daily_Range_Pct'] = ((df['High'] - df['Low']) / df['Low']) * 100

    # Gap from previous close
    df['Gap_Pct'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100

    # Volume ratio vs 20-day moving average
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']

    # Remove rows with NaN created by calculations
    df = df.dropna()

    print(f"✅ {ticker}: {len(df)} days of data prepared")
    print(f"📅 Period: {df.index[0].date()} → {df.index[-1].date()}")

    return df