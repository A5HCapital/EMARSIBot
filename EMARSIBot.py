import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Calculate Indicators
def calculate_indicators(data, ema_short_period, ema_long_period):
    data['EMA_Short'] = data['Close'].ewm(span=ema_short_period, adjust=False).mean()
    data['EMA_Long'] = data['Close'].ewm(span=ema_long_period, adjust=False).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['Volume_Change'] = data['Volume'].pct_change()
    return data

# Backtest Function
def backtest(data, initial_balance, rsi_threshold):
    balance = initial_balance
    position = 0
    data['Portfolio Value'] = initial_balance
    data['Signal'] = 0
    trades = []

    for i in range(1, len(data)):
        # Buy condition
        if (
            data['EMA_Short'].iloc[i] > data['EMA_Long'].iloc[i] and
            data['EMA_Short'].iloc[i - 1] <= data['EMA_Long'].iloc[i - 1] and
            data['Volume_Change'].iloc[i] > 0 and
            data['RSI'].iloc[i] > rsi_threshold and
            position == 0  # Ensures no position is currently held (scalar check)
        ):
            entry_date = data.index[i]
            entry_price = float(data['Close'].iloc[i])  # Ensure scalar value
            position = balance / entry_price
            balance = 0
            data.loc[data.index[i], 'Signal'] = 1
            trades.append({"Type": "Buy", "Date": entry_date, "Price": entry_price})

        # Sell condition
        elif (
            data['EMA_Short'].iloc[i] < data['EMA_Long'].iloc[i] and
            data['EMA_Short'].iloc[i - 1] >= data['EMA_Long'].iloc[i - 1] and
            data['Volume_Change'].iloc[i] > 0 and
            data['RSI'].iloc[i] < rsi_threshold and
            position > 0  # Ensures a position is held (scalar check)
        ):
            exit_date = data.index[i]
            exit_price = float(data['Close'].iloc[i])  # Ensure scalar value
            balance = position * exit_price
            trades[-1].update({"Exit Date": exit_date, "Exit Price": exit_price})
            trades[-1].update({"P/L": balance - initial_balance, "P/L %": (balance - initial_balance) / initial_balance * 100})
            position = 0
            data.loc[data.index[i], 'Signal'] = -1

        # Update portfolio value
        data.loc[data.index[i], 'Portfolio Value'] = balance + (position * float(data['Close'].iloc[i]))  # Ensure scalar calculation

    data['Peak'] = data['Portfolio Value'].cummax()
    data['Drawdown'] = data['Portfolio Value'] - data['Peak']
    data['Drawdown %'] = (data['Drawdown'] / data['Peak']) * 100
    max_drawdown = data['Drawdown %'].min()

    for trade in trades:
        trade["Max Drawdown %"] = max_drawdown

    return data, trades


# Streamlit UI
st.title("Swing Trading Backtest App")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
initial_balance = st.number_input("Initial Balance ($)", value=10000, min_value=1)
ema_short_period = st.number_input("Short EMA Period", value=5, min_value=1)
ema_long_period = st.number_input("Long EMA Period", value=20, min_value=1)
rsi_threshold = st.number_input("RSI Threshold", value=50, min_value=1, max_value=100)

if st.button("Run Backtest"):
    st.write("Fetching stock data...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if not data.empty:
        st.write("Calculating indicators...")
        data = calculate_indicators(data, ema_short_period, ema_long_period)
        
        st.write("Running backtest...")
        results, trades = backtest(data, initial_balance, rsi_threshold)
        
        st.write("Backtest Complete!")
        st.write(f"**Final Portfolio Value:** ${results['Portfolio Value'].iloc[-1]:,.2f}")
        st.write(f"**Max Drawdown:** {results['Drawdown %'].min():.2f}%")
        
        st.write("Equity Curve:")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results.index, results['Portfolio Value'], label="Portfolio Value")
        ax.plot(results.index, results['Close'], label="Stock Price", alpha=0.5)
        ax.legend()
        st.pyplot(fig)

        # Export trades to CSV
        trades_df = pd.DataFrame(trades)
        csv = trades_df.to_csv(index=False)
        st.download_button(label="Download Trades CSV", data=csv, file_name=f"{ticker}_trade_details.csv", mime="text/csv")
        
        # Export results with portfolio value
        csv_results = results.to_csv(index=True)
        st.download_button(label="Download Full Results CSV", data=csv_results, file_name=f"{ticker}_backtest_results.csv", mime="text/csv")
    else:
        st.error("No data found for the given ticker and date range.")
