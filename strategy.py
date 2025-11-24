import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def analyze_trades_detailed(results, initial_capital=10000):
    """
    Analizza ogni singolo trade della strategia.
    
    Returns:
    - DataFrame con tutti i trade
    - Metriche dettagliate
    """
    
    # Identifica i punti di entrata (BUY) e uscita (SELL)
    buy_signals = results[results['Position'] == 1].copy()
    sell_signals = results[results['Position'] == -1].copy()
    
    trades_list = []
    
    # Pairing: ogni BUY con il successivo SELL
    for i in range(min(len(buy_signals), len(sell_signals))):
        entry_date = buy_signals.index[i]
        entry_price = buy_signals['Close'].iloc[i]
        
        # Trova il primo SELL dopo questo BUY
        future_sells = sell_signals[sell_signals.index > entry_date]
        if len(future_sells) > 0:
            exit_date = future_sells.index[0]
            exit_price = future_sells['Close'].iloc[0]
            
            # Calcola P&L
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_dollar = (exit_price - entry_price) * (initial_capital / entry_price)
            
            # Holding period
            holding_days = (exit_date - entry_date).days
            
            trade = {
                'Trade_Num': i + 1,
                'Entry_Date': entry_date,
                'Entry_Price': entry_price,
                'Exit_Date': exit_date,
                'Exit_Price': exit_price,
                'PnL_%': pnl_pct,
                'PnL_$': pnl_dollar,
                'Holding_Days': holding_days,
                'Result': 'WIN' if pnl_pct > 0 else 'LOSS'
            }
            trades_list.append(trade)
    
    trades_df = pd.DataFrame(trades_list)
    
    return trades_df


def analyze_drawdown(results):
    """
    Analisi dettagliata del drawdown della strategia.
    """
    
    equity = results['Strategy_Equity']
    
    # Calcola running maximum
    running_max = equity.cummax()
    
    # Drawdown in dollari e percentuale
    drawdown_dollars = equity - running_max
    drawdown_pct = (drawdown_dollars / running_max) * 100
    
    # Trova il max drawdown
    max_dd_idx = drawdown_pct.idxmin()
    max_dd_value = drawdown_pct.min()
    
    # Trova quando è iniziato il drawdown (ultimo peak prima del max DD)
    peak_idx = running_max[:max_dd_idx].idxmax()
    
    # Trova quando si è recuperato (se si è recuperato)
    recovery_idx = None
    peak_value = running_max[peak_idx]
    future_equity = equity[max_dd_idx:]
    
    if len(future_equity[future_equity >= peak_value]) > 0:
        recovery_idx = future_equity[future_equity >= peak_value].index[0]
    
    print("\n📉 ANALISI DRAWDOWN:")
    print("="*60)
    print(f"Max Drawdown: {max_dd_value:.2f}%")
    print(f"Peak prima del DD: {peak_idx.date()}")
    print(f"Bottom del DD: {max_dd_idx.date()}")
    
    if recovery_idx:
        print(f"Recovery: {recovery_idx.date()}")
        dd_duration = (max_dd_idx - peak_idx).days
        recovery_duration = (recovery_idx - max_dd_idx).days
        total_duration = (recovery_idx - peak_idx).days
        print(f"\nDurata DD: {dd_duration} giorni")
        print(f"Durata Recovery: {recovery_duration} giorni")
        print(f"Durata totale: {total_duration} giorni")
    else:
        print(f"Recovery: Non ancora recuperato!")
    
    print("="*60)
    
    # Plot drawdown
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(results.index, 0, drawdown_pct, color='red', alpha=0.3)
    ax.plot(results.index, drawdown_pct, color='red', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Evidenzia il max drawdown
    ax.scatter([max_dd_idx], [max_dd_value], color='darkred', s=200, zorder=5,
              label=f'Max DD: {max_dd_value:.2f}%')
    
    ax.set_title('Drawdown Analysis Over Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def moving_average_crossover_strategy(ticker, start_date, end_date, 
                                       fast_period=50, slow_period=200,
                                       initial_capital=10000):
    """
    Implementa una strategia Moving Average Crossover.
    
    Parameters:
    - ticker: simbolo azione
    - start_date, end_date: periodo analisi
    - fast_period: periodo media mobile veloce (default 50)
    - slow_period: periodo media mobile lenta (default 200)
    - initial_capital: capitale iniziale in $
    
    Returns:
    - DataFrame con segnali e performance
    - Dizionario con metriche di performance
    """
    
    # Scarica dati
    print(f"📥 Scaricando dati {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, ignore_tz=True)
    
    # Fix MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calcola medie mobili
    df['MA_Fast'] = df['Close'].rolling(window=fast_period).mean()
    df['MA_Slow'] = df['Close'].rolling(window=slow_period).mean()
    
    # Genera segnali di trading
    df['Signal'] = 0  # 0 = nessun segnale
    df['Signal'][fast_period:] = np.where(
        df['MA_Fast'][fast_period:] > df['MA_Slow'][fast_period:], 1, 0
    )
    
    # Identifica i punti di entrata/uscita (quando il segnale cambia)
    df['Position'] = df['Signal'].diff()
    # Position: 1 = BUY signal, -1 = SELL signal, 0 = hold
    
    # Rimuovi le prime righe con NaN
    df = df.dropna()
    
    # Calcola rendimenti della strategia
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    
    # Calcola equity curve (valore portafoglio nel tempo)
    df['Buy_Hold_Equity'] = initial_capital * (1 + df['Returns']).cumprod()
    df['Strategy_Equity'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    
    print(f"✅ Strategia calcolata su {len(df)} giorni")
    print(f"📅 Periodo: {df.index[0].date()} → {df.index[-1].date()}")
    
    return df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def calculate_rsi(data, period=14):
    """
    Calcola RSI (Relative Strength Index).
    
    Parameters:
    - data: Series dei prezzi di chiusura
    - period: periodo per il calcolo (default 14)
    
    Returns:
    - Series con valori RSI
    """
    # Calcola i cambiamenti di prezzo
    delta = data.diff()
    
    # Separa gain e loss
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calcola media mobile esponenziale (EMA) di gain e loss
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    # Calcola RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calcola RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def rsi_mean_reversion_strategy(ticker, start_date, end_date, 
                                  rsi_period=14, 
                                  oversold=30, 
                                  overbought=70,
                                  initial_capital=10000):
    """
    Implementa strategia RSI Mean Reversion.
    
    Parameters:
    - ticker: simbolo azione
    - start_date, end_date: periodo analisi
    - rsi_period: periodo per calcolo RSI (default 14)
    - oversold: soglia oversold per BUY (default 30)
    - overbought: soglia overbought per SELL (default 70)
    - initial_capital: capitale iniziale
    
    Returns:
    - DataFrame con segnali e performance
    """
    
    # Scarica dati
    print(f"📥 Scaricando dati {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, ignore_tz=True)
    
    # Fix MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calcola RSI
    df['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
    
    # Genera segnali
    df['Signal'] = 0  # 0 = hold, 1 = long position
    
    # BUY quando RSI < oversold (asset sottovalutato)
    df.loc[df['RSI'] < oversold, 'Signal'] = 1
    
    # SELL quando RSI > overbought (asset sopravalutato)
    df.loc[df['RSI'] > overbought, 'Signal'] = 0
    
    # Forward fill: mantieni la posizione finché non arriva segnale opposto
    df['Signal'] = df['Signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Identifica i punti di entrata/uscita
    df['Position'] = df['Signal'].diff()
    
    # Calcola rendimenti
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    
    # Equity curve
    df['Buy_Hold_Equity'] = initial_capital * (1 + df['Returns']).cumprod()
    df['Strategy_Equity'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    
    # Rimuovi NaN
    df = df.dropna()
    
    print(f"✅ Strategia calcolata su {len(df)} giorni")
    print(f"📅 Periodo: {df.index[0].date()} → {df.index[-1].date()}")
    
    return df

def analyze_rsi_trades(results, initial_capital=10000):
    """Analizza trade-by-trade della strategia RSI."""
    
    buy_signals = results[results['Position'] == 1].copy()
    sell_signals = results[results['Position'] == -1].copy()
    
    trades_list = []
    
    for i in range(min(len(buy_signals), len(sell_signals))):
        entry_date = buy_signals.index[i]
        entry_price = buy_signals['Close'].iloc[i]
        entry_rsi = buy_signals['RSI'].iloc[i]
        
        future_sells = sell_signals[sell_signals.index > entry_date]
        if len(future_sells) > 0:
            exit_date = future_sells.index[0]
            exit_price = future_sells['Close'].iloc[0]
            exit_rsi = future_sells['RSI'].iloc[0]
            
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_dollar = (exit_price - entry_price) * (initial_capital / entry_price)
            holding_days = (exit_date - entry_date).days
            
            trade = {
                'Trade_Num': i + 1,
                'Entry_Date': entry_date,
                'Entry_Price': entry_price,
                'Entry_RSI': entry_rsi,
                'Exit_Date': exit_date,
                'Exit_Price': exit_price,
                'Exit_RSI': exit_rsi,
                'PnL_%': pnl_pct,
                'PnL_$': pnl_dollar,
                'Holding_Days': holding_days,
                'Result': 'WIN' if pnl_pct > 0 else 'LOSS'
            }
            trades_list.append(trade)
    
    return pd.DataFrame(trades_list)
