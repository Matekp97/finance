from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_fetcher import download_multiple_stocks, prepare_stock_data
from analysis import (
    analyze_multiple_stocks,
    find_high_volume_days,
    find_high_volatility_days,
    calculate_volume_stats,
    calculate_correlation
)
from plotting import setup_plot_style, plot_multiple_stocks, plot_price_and_volume
from strategy import analyze_trades_detailed, analyze_drawdown, moving_average_crossover_strategy

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
    ticker_detail = 'AAPL'
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
            'Daily_Return'
        )
        print(f"\n🔗 Correlation between Volume Ratio and |Daily Return|: {correlation:.3f}")
        print("(Values close to 1 = strong correlation, close to 0 = no correlation)")

        # Plot price and volume chart
        plot_price_and_volume(pltr_data, ticker_detail)


        # Test della strategia su AAPL (meno volatile di PLTR per il primo test)
        start = '2010-01-01'
        end = '2025-11-20'

        results = moving_average_crossover_strategy(ticker_detail, start, end, 
                                                    fast_period=20, slow_period=50,
                                                    initial_capital=10000)

        # Visualizza i primi segnali
        print("\n🎯 PRIMI SEGNALI GENERATI:")
        trades = results[results['Position'] != 0][['Close', 'MA_Fast', 'MA_Slow', 'Position']].head(10)
        trades['Action'] = trades['Position'].map({1: '🟢 BUY', -1: '🔴 SELL'})
        print(trades[['Close', 'Action']])

        trades_df = analyze_trades_detailed(results, initial_capital=10000)

        print("\n📊 ANALISI TRADE-BY-TRADE")
        print("="*100)
        print(trades_df.to_string(index=False))
        print("="*100)

        # Statistiche sui trade
        print("\n📈 STATISTICHE DETTAGLIATE:")
        print(f"Numero totale trade: {len(trades_df)}")
        print(f"Trade vincenti: {len(trades_df[trades_df['Result'] == 'WIN'])}")
        print(f"Trade perdenti: {len(trades_df[trades_df['Result'] == 'LOSS'])}")
        print(f"\nWin Rate: {(len(trades_df[trades_df['Result'] == 'WIN']) / len(trades_df) * 100):.2f}%")

        winning_trades = trades_df[trades_df['Result'] == 'WIN']
        losing_trades = trades_df[trades_df['Result'] == 'LOSS']

        if len(winning_trades) > 0:
            print(f"\nAverage Win: {winning_trades['PnL_%'].mean():.2f}%")
            print(f"Average Win ($): ${winning_trades['PnL_$'].mean():.2f}")
            print(f"Largest Win: {winning_trades['PnL_%'].max():.2f}%")

        if len(losing_trades) > 0:
            print(f"\nAverage Loss: {losing_trades['PnL_%'].mean():.2f}%")
            print(f"Average Loss ($): ${losing_trades['PnL_$'].mean():.2f}")
            print(f"Largest Loss: {losing_trades['PnL_%'].min():.2f}%")

        # Risk/Reward Ratio
        if len(losing_trades) > 0 and len(winning_trades) > 0:
            risk_reward = abs(winning_trades['PnL_%'].mean() / losing_trades['PnL_%'].mean())
            print(f"\nRisk/Reward Ratio: {risk_reward:.2f}")

        print(f"\nHolding period medio: {trades_df['Holding_Days'].mean():.1f} giorni")

        # Grafico distribuzione P&L
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Histogram dei rendimenti
        axes[0, 0].hist(trades_df['PnL_%'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Distribuzione Rendimenti per Trade', fontweight='bold')
        axes[0, 0].set_xlabel('P&L (%)')
        axes[0, 0].set_ylabel('Frequenza')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Win vs Loss comparison
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        axes[0, 1].bar(['Winning Trades', 'Losing Trades'], [win_count, loss_count],
                    color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Win vs Loss Trades', fontweight='bold')
        axes[0, 1].set_ylabel('Numero di Trade')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. P&L cumulativo trade-by-trade
        trades_df['Cumulative_PnL_$'] = trades_df['PnL_$'].cumsum()
        axes[1, 0].plot(trades_df['Trade_Num'], trades_df['Cumulative_PnL_$'], 
                        marker='o', linewidth=2, markersize=6, color='green')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1, 0].set_title('P&L Cumulativo per Trade', fontweight='bold')
        axes[1, 0].set_xlabel('Numero Trade')
        axes[1, 0].set_ylabel('P&L Cumulativo ($)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Holding period distribution
        axes[1, 1].hist(trades_df['Holding_Days'], bins=15, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribuzione Holding Period', fontweight='bold')
        axes[1, 1].set_xlabel('Giorni')
        axes[1, 1].set_ylabel('Frequenza')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


        # Esegui analisi drawdown
        analyze_drawdown(results)



if __name__ == "__main__":
    main()