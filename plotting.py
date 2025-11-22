import matplotlib.pyplot as plt


def setup_plot_style():
    """Configure matplotlib style for professional-looking charts."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (14, 7)


def plot_multiple_stocks(stocks_data, normalized=False):
    """
    Plot comparison of multiple stocks.

    Args:
        stocks_data: Dictionary with ticker as key and DataFrame as value
        normalized: If True, normalize prices to base 100
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Chart 1: Absolute or normalized prices
    if normalized:
        axes[0].set_title('Normalized Performance (base 100)', fontsize=14, fontweight='bold')
        for ticker, data in stocks_data.items():
            normalized_prices = (data['Close'] / data['Close'].iloc[0]) * 100
            axes[0].plot(normalized_prices.index, normalized_prices, label=ticker, linewidth=2)
        axes[0].set_ylabel('Performance (base 100)', fontsize=12)
        axes[0].axhline(y=100, color='black', linestyle='--', alpha=0.5)
    else:
        axes[0].set_title('Closing Prices', fontsize=14, fontweight='bold')
        for ticker, data in stocks_data.items():
            axes[0].plot(data.index, data['Close'], label=ticker, linewidth=2)
        axes[0].set_ylabel('Price ($)', fontsize=12)

    axes[0].legend(loc='best', ncol=5)
    axes[0].grid(True, alpha=0.3)

    # Chart 2: Normalized performance
    axes[1].set_title('Normalized Performance (base 100)', fontsize=14, fontweight='bold')
    for ticker, data in stocks_data.items():
        normalized = (data['Close'] / data['Close'].iloc[0]) * 100
        axes[1].plot(normalized.index, normalized, label=ticker, linewidth=2)

    axes[1].set_ylabel('Performance (base 100)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(loc='best', ncol=5)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=100, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_price_and_volume(data, ticker):
    """
    Create a professional chart with price and volume.

    Args:
        data: DataFrame with stock data
        ticker: Stock symbol for chart title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Calculate colors based on daily movement
    colors = ['green' if close >= open_price else 'red'
              for close, open_price in zip(data['Close'], data['Open'])]

    # Chart 1: Price with daily range bars
    ax1.bar(data.index, data['High'] - data['Low'],
            bottom=data['Low'], width=0.6, color=colors, alpha=0.3)
    ax1.plot(data.index, data['Close'], color='black', linewidth=1.5, label='Close')
    ax1.set_title(f'{ticker} - Price and Volume', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Chart 2: Volume bars
    volume_colors = ['green' if close >= open_price else 'red'
                     for close, open_price in zip(data['Close'], data['Open'])]
    ax2.bar(data.index, data['Volume'], color=volume_colors, alpha=0.7)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add volume moving average
    volume_ma = data['Volume'].rolling(window=20).mean()
    ax2.plot(data.index, volume_ma, color='blue', linewidth=2, label='Volume MA(20)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_single_stock(data, ticker):
    """
    Simple plot of a single stock's closing price.

    Args:
        data: DataFrame with stock data
        ticker: Stock symbol for chart title
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], linewidth=2, color='blue')
    plt.title(f'{ticker} - Closing Price', fontsize=14, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()