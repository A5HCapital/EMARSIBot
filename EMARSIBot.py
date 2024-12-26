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

    # Dynamically identify the ticker (second level of the MultiIndex for price columns)
    ticker = data.columns.get_level_values(1).unique()[0]

    for i in range(1, len(data)):
        # Buy condition
        if (
            data['EMA_Short'].iloc[i] > data['EMA_Long'].iloc[i] and
            data['EMA_Short'].iloc[i - 1] <= data['EMA_Long'].iloc[i - 1] and
            data['Volume_Change'].iloc[i] > 0 and
            data['RSI'].iloc[i] > rsi_threshold and
            position == 0
        ):
            entry_date = data.index[i]
            entry_price = float(data[('Close', ticker)].iloc[i])  # Dynamically access 'Close' price
            position = balance / entry_price
            balance = 0
            data.loc[data.index[i], 'Signal'] = 1
            trades.append({"Type": "Buy", "Entry Date": entry_date, "Entry Price": entry_price})

        # Sell condition
        elif (
            data['EMA_Short'].iloc[i] < data['EMA_Long'].iloc[i] and
            data['EMA_Short'].iloc[i - 1] >= data['EMA_Long'].iloc[i - 1] and
            data['Volume_Change'].iloc[i] > 0 and
            data['RSI'].iloc[i] < rsi_threshold and
            position > 0
        ):
            exit_date = data.index[i]
            exit_price = float(data[('Close', ticker)].iloc[i])  # Dynamically access 'Close' price
            balance = position * exit_price
            trades[-1].update({
                "Exit Date": exit_date,
                "Exit Price": exit_price,
                "P/L": balance - initial_balance,
                "P/L %": (balance - initial_balance) / initial_balance * 100
            })
            position = 0
            data.loc[data.index[i], 'Signal'] = -1

        # Update portfolio value
        data.loc[data.index[i], 'Portfolio Value'] = balance + (position * float(data[('Close', ticker)].iloc[i]))

    data['Peak'] = data['Portfolio Value'].cummax()
    data['Drawdown'] = data['Portfolio Value'] - data['Peak']
    data['Drawdown %'] = (data['Drawdown'] / data['Peak']) * 100
    max_drawdown = data['Drawdown %'].min()

    for trade in trades:
        trade["Max Drawdown %"] = max_drawdown

    # Create trades DataFrame
    trade_df = pd.DataFrame(trades)

    # Flatten the MultiIndex in `data`
    data.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in data.columns]
    data = data.reset_index()  # Reset the index, making 'Date' a column
    data.rename(columns={'index': 'Date'}, inplace=True)  # Rename the index column to 'Date'

    # Perform the merge
    combined = pd.merge(data, trade_df, how='left', left_on='Date', right_on='Entry Date')

    return combined








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
        results = backtest(data, initial_balance, rsi_threshold)
        
        st.write("Backtest Complete!")
        st.write(f"**Final Portfolio Value:** ${results['Portfolio Value'].iloc[-1]:,.2f}")
        st.write(f"**Max Drawdown:** {results['Drawdown %'].min():.2f}%")
        
        # Dynamically get the flattened column name for `Close`
        close_col = [col for col in results.columns if col.startswith("Close")][0]

        st.write("Equity Curve:")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results['Date'], results['Portfolio Value'], label="Portfolio Value")
        ax.plot(results['Date'], results[close_col], label="Stock Price", alpha=0.5)
        ax.legend()
        st.pyplot(fig)

        # Export combined results to CSV
        csv = results.to_csv(index=False)
        st.download_button(label="Download Combined Results CSV", data=csv, file_name=f"{ticker}_backtest_combined_results.csv", mime="text/csv")
    else:
        st.error("No data found for the given ticker and date range.")




