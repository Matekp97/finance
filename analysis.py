import pandas as pd
import numpy as np


def calculate_stock_metrics(data):
    """
    Calculate key metrics for a stock.

    Args:
        data: DataFrame with stock data (must have 'Close' column)

    Returns:
        Dictionary with metrics
    """
    price_start = data['Close'].iloc[0]
    price_end = data['Close'].iloc[-1]

    # Total return percentage
    total_return = ((price_end - price_start) / price_start) * 100

    # Volatility (annualized standard deviation of daily returns)
    daily_returns = data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100

    metrics = {
        'price_start': price_start,
        'price_end': price_end,
        'total_return': total_return,
        'volatility': volatility,
        'daily_returns_mean': daily_returns.mean() * 100,
        'daily_returns_std': daily_returns.std() * 100
    }

    return metrics


def analyze_multiple_stocks(stocks_data):
    """
    Analyze multiple stocks and return comparative metrics.

    Args:
        stocks_data: Dictionary with ticker as key and DataFrame as value

    Returns:
        DataFrame with comparative metrics for all stocks
    """
    results = []

    for ticker, data in stocks_data.items():
        metrics = calculate_stock_metrics(data)
        metrics['ticker'] = ticker
        results.append(metrics)

    return pd.DataFrame(results)


def find_high_volume_days(data, threshold=2.0):
    """
    Find days with anomalously high volume.

    Args:
        data: DataFrame with 'Volume_Ratio' column
        threshold: Minimum volume ratio to be considered anomalous

    Returns:
        DataFrame with high volume days
    """
    if 'Volume_Ratio' not in data.columns:
        raise ValueError("Data must have 'Volume_Ratio' column. Use prepare_stock_data().")

    return data[data['Volume_Ratio'] > threshold]


def find_high_volatility_days(data, n=10):
    """
    Find the most volatile days by intraday range.

    Args:
        data: DataFrame with 'Daily_Range_Pct' column
        n: Number of top days to return

    Returns:
        DataFrame with most volatile days
    """
    if 'Daily_Range_Pct' not in data.columns:
        raise ValueError("Data must have 'Daily_Range_Pct' column. Use prepare_stock_data().")

    return data.nlargest(n, 'Daily_Range_Pct')


def calculate_volume_stats(data):
    """
    Calculate volume statistics.

    Args:
        data: DataFrame with 'Volume' column

    Returns:
        Dictionary with volume statistics
    """
    return {
        'mean': data['Volume'].mean(),
        'median': data['Volume'].median(),
        'max': data['Volume'].max(),
        'min': data['Volume'].min(),
        'std': data['Volume'].std()
    }


def calculate_correlation(data, col1, col2):
    """
    Calculate correlation between two columns.

    Args:
        data: DataFrame
        col1: First column name
        col2: Second column name

    Returns:
        Correlation coefficient
    """
    return data[col1].corr(data[col2])


def normalize_prices(stocks_data, base=100):
    """
    Normalize stock prices to a common base for comparison.

    Args:
        stocks_data: Dictionary with ticker as key and DataFrame as value
        base: Base value for normalization (default: 100)

    Returns:
        Dictionary with normalized price series
    """
    normalized = {}

    for ticker, data in stocks_data.items():
        normalized[ticker] = (data['Close'] / data['Close'].iloc[0]) * base

    return normalized