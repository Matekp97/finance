from datetime import datetime, timedelta
from data_fetcher import download_multiple_stocks, prepare_stock_data
from analysis import (
    analyze_multiple_stocks,
    find_high_volume_days,
    find_high_volatility_days,
    calculate_volume_stats,
    calculate_correlation
)
from plotting import setup_plot_style, plot_multiple_stocks, plot_price_and_volume


def main():
    """Main analysis script for stock data."""

    # Configure plotting style
    setup_plot_style()

    # Configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
               'NVDA', 'META', 'JPM', 'V', 'WMT']

    # Analysis period: last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)

    print(f"📅 Downloading data from {start_date.date()} to {end_date.date()}")
    print("="*60)

    # Download data for multiple stocks
    stocks_data = download_multiple_stocks(tickers, start_date, end_date)

    print("="*60)
    print(f"✅ Downloaded data for {len(stocks_data)} stocks\n")

    # Comparative analysis
    print("📊 COMPARATIVE ANALYSIS - Last 2 years")
    print("="*80)
    print(f"{'Ticker':<10} {'Start Price':<15} {'End Price':<15} {'Return %':<12} {'Volatility %':<15}")
    print("="*80)

    metrics_df = analyze_multiple_stocks(stocks_data)

    for _, row in metrics_df.iterrows():
        print(f"{row['ticker']:<10} ${row['price_start']:<14.2f} ${row['price_end']:<14.2f} "
              f"{row['total_return']:<11.2f}% {row['volatility']:<14.2f}%")

    print("="*80)

    # Plot comparative charts
    plot_multiple_stocks(stocks_data, normalized=False)

    # Detailed analysis for a specific stock
    ticker_detail = 'PLTR'
    print(f"\n\n📊 Detailed analysis for {ticker_detail}")
    print("="*60)

    pltr_data = prepare_stock_data(ticker_detail, period='1y')

    if pltr_data is not None:
        # Find high volume days
        high_volume_days = find_high_volume_days(pltr_data, threshold=2.0)
        print(f"\n🔥 HIGH VOLUME DAYS (>2x average) for {ticker_detail}:")
        print(f"Total: {len(high_volume_days)} days out of {len(pltr_data)}")

        if len(high_volume_days) > 0:
            print("\nTop 5 days by volume ratio:")
            top_5 = high_volume_days.nlargest(5, 'Volume_Ratio')[
                ['Close', 'Volume', 'Volume_Ratio', 'Daily_Return']
            ]
            print(top_5)

        # Find high volatility days
        high_vol_days = find_high_volatility_days(pltr_data, n=10)
        print(f"\n⚡ TOP 10 MOST VOLATILE DAYS (highest intraday range %):")
        print(high_vol_days[['Close', 'Daily_Range_Pct', 'Daily_Return', 'Volume_Ratio']])

        # Volume statistics
        vol_stats = calculate_volume_stats(pltr_data)
        print("\n📊 VOLUME STATISTICS:")
        print(f"Average daily volume: {vol_stats['mean']:,.0f}")
        print(f"Median volume: {vol_stats['median']:,.0f}")
        print(f"Maximum volume: {vol_stats['max']:,.0f}")
        print(f"Minimum volume: {vol_stats['min']:,.0f}")
        print(f"Standard deviation: {vol_stats['std']:,.0f}")

        # Correlation analysis
        correlation = calculate_correlation(
            pltr_data,
            'Volume_Ratio',
            pltr_data['Daily_Return'].abs()
        )
        print(f"\n🔗 Correlation between Volume Ratio and |Daily Return|: {correlation:.3f}")
        print("(Values close to 1 = strong correlation, close to 0 = no correlation)")

        # Plot price and volume chart
        plot_price_and_volume(pltr_data, ticker_detail)


if __name__ == "__main__":
    main()