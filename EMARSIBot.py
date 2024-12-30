import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Calculate Indicators
def calculate_indicators(data, ema_short_period, ema_long_period):
    # Flatten MultiIndex if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]

    # Identify Close and Volume columns dynamically
    close_col = [col for col in data.columns if 'Close' in col][0]
    volume_col = [col for col in data.columns if 'Volume' in col]
    if not volume_col:
        st.error("The 'Volume' column is missing. Ensure you are fetching valid data.")
        st.stop()
    volume_col = volume_col[0]

    # Calculate Indicators using the identified Close column
    data['EMA_Short'] = data[close_col].ewm(span=ema_short_period, adjust=False).mean()
    data['EMA_Long'] = data[close_col].ewm(span=ema_long_period, adjust=False).mean()
    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['Volume_Change'] = data[volume_col].pct_change()
    return data

# Backtest Function
def backtest(data, initial_balance, rsi_threshold):
    # Flatten MultiIndex if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]

    # Identify the Close column dynamically
    close_col = [col for col in data.columns if 'Close' in col][0]

    balance = initial_balance
    position = 0
    shares = 0
    trades = []

    for i in range(1, len(data)):
        # Extract scalar values
        current_price = data[close_col].iloc[i]
        ema_short = data['EMA_Short'].iloc[i]
        ema_long = data['EMA_Long'].iloc[i]
        rsi = data['RSI'].iloc[i]

        # Entry Condition
        if ema_short > ema_long and rsi > rsi_threshold and position == 0:
            trade_value = balance * 0.05
            shares = trade_value // current_price
            position = shares * current_price
            balance -= position
            trades.append({
                "Entry Date": data.index[i],
                "Entry Price": current_price,
                "Shares Bought": shares
            })

        # Exit Condition
        elif position > 0:
            unrealized_pl = ((current_price - trades[-1]["Entry Price"]) / trades[-1]["Entry Price"]) * 100
            if unrealized_pl >= 20 or unrealized_pl <= -10:
                balance += shares * current_price
                trades[-1].update({
                    "Exit Date": data.index[i],
                    "Exit Price": current_price,
                    "Shares Sold": shares,
                    "P/L": (current_price - trades[-1]["Entry Price"]) * shares,
                    "P/L %": unrealized_pl
                })
                position = 0
                shares = 0

    # Add Portfolio Value to DataFrame
    data['Portfolio Value'] = balance
    return pd.DataFrame(trades)

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
        # Flatten MultiIndex columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns]

        # Debugging: Check column names
        st.write("Columns in data (before identifying Close/Volume columns):", data.columns)

        # Identify the Close column
        if not any('Close' in col for col in data.columns):
            st.error("The 'Close' column is missing. Ensure you are fetching valid data.")
            st.stop()

        st.write("Calculating indicators...")
        data = calculate_indicators(data, ema_short_period, ema_long_period)
        
        st.write("Running backtest...")
        results = backtest(data, initial_balance, rsi_threshold)
        
        if not results.empty:
            st.write(f"**Final Balance:** ${results['P/L'].sum() + initial_balance:.2f}")
            st.write(f"**Total Trades:** {len(results)}")
            
            # Plot Portfolio Value and Stock Price
            st.subheader("Performance Chart")
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot Portfolio Value
            ax1.plot(data.index, data['Portfolio Value'], label="Portfolio Value", color="green")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Portfolio Value ($)", color="green")
            ax1.tick_params(axis="y", labelcolor="green")

            # Plot Stock Price
            close_col = [col for col in data.columns if 'Close' in col][0]
            ax2 = ax1.twinx()
            ax2.plot(data.index, data[close_col], label="Stock Price", color="blue", alpha=0.6)
            ax2.set_ylabel("Stock Price ($)", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")

            fig.tight_layout()
            st.pyplot(fig)

            # Export Results to CSV
            st.subheader("Transaction Details")
            st.dataframe(results)
            st.download_button(
                label="Download Transactions CSV",
                data=results.to_csv(index=False),
                file_name=f"{ticker}_backtest_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("No trades were executed based on the conditions.")
    else:
        st.error("No data found for the given ticker and date range.")
