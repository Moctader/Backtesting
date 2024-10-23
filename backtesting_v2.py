import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import ffn
import matplotlib.pyplot as plt

# 1. Data Preparation

ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Prepare features and target
data['Next_High'] = data['High'].shift(-1)  # Target variable: next day's high
data['Return'] = data['Close'].pct_change()  # Daily returns
data['Volatility'] = data['Return'].rolling(window=5).std()  # 5-day rolling volatility
data['High_Low_Range'] = (data['High'] - data['Low']) / data['Low']  # Intraday high-low range
data['Prev_Close_Rel_High'] = (data['Close'].shift(1) - data['High']) / data['High']  # Previous close relative to high

# Drop NaN rows
data.dropna(inplace=True)

# Normalize the features
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility', 'High_Low_Range', 'Prev_Close_Rel_High']])

# Normalize the target
target_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_target = target_scaler.fit_transform(data[['Next_High']])

# Create dataset function for LSTM
def create_dataset(features, target, time_step=1):
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        X.append(features[i:(i + time_step), :])
        y.append(target[i + time_step])
    return np.array(X), np.array(y)

# Define time step for LSTM model
time_step = 60
X, y = create_dataset(scaled_features, scaled_target, time_step)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 2. Build and Train the LSTM Model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, callbacks=[early_stop], verbose=1)

# Make predictions for the test set
predictions = model.predict(X_test)
predictions = target_scaler.inverse_transform(predictions)

# 3. Trading Strategy Implementation

# Initialize parameters
initial_cash = 100000  # Starting cash
position = 0  # Initially no position
cash = initial_cash
portfolio_value = []
trades = []

# Strategy parameters
buy_threshold = 0.02  # Buy if predicted high is 2% higher than current price
position_size_factor = 0.1  # Allocate 10% of capital to stock
trading_days = len(X_test)  # Backtest over the test period

# Backtest
for i in range(trading_days):
    current_price = data['Close'].iloc[train_size + i]
    current_price = current_price.values[0] if isinstance(current_price, pd.Series) else current_price
    predicted_high = predictions[i][0]
    daily_portfolio_value = cash  # Start with cash

    # Debugging: Print current price and predicted high
    print(f"Day {i}: Current Price = {current_price:.2f}, Predicted High = {predicted_high:.2f}")

    # Buy condition: predicted high is 2% higher than current price and no current position
    if position == 0 and predicted_high > current_price * (1 + buy_threshold):
        amount_to_invest = position_size_factor * cash  # Invest 10% of cash
        shares_to_buy = amount_to_invest // current_price
        cash -= shares_to_buy * current_price
        position += shares_to_buy
        buy_price = current_price  # Store the buy price
        print(f"Bought {shares_to_buy} shares at {current_price:.2f}")

    # Sell condition: current price reaches or exceeds predicted high
    if position > 0 and current_price >= predicted_high:
        cash += position * current_price
        profit = (current_price - buy_price) * position  # Calculate profit based on buy price
        trades.append(profit)  # Store the result of the trade
        print(f"Sold {position} shares at {current_price:.2f}, Profit: {profit:.2f}")
        position = 0  # Reset position after selling

    # Update portfolio value
    daily_portfolio_value = cash + position * current_price
    portfolio_value.append(daily_portfolio_value)

    # Update portfolio value
    daily_portfolio_value = cash + position * current_price
    portfolio_value.append(daily_portfolio_value)

# 4. Performance Metrics: Sharpe Ratio, Max Drawdown

# Convert portfolio value to pandas Series for performance analysis
portfolio_series = pd.Series(portfolio_value, index=pd.date_range(start=data.index[-len(portfolio_value)], periods=len(portfolio_value), freq='B'))

# Calculate performance metrics
sharpe_ratio = ffn.calc_sharpe(portfolio_series.pct_change(), rf=0.0, nperiods=252)
max_drawdown = ffn.calc_max_drawdown(portfolio_series)


# Calculate hit ratio
winning_trades = [trade for trade in trades if trade > 0]
hit_ratio = (len(winning_trades) / len(trades)) * 100 if trades else 0

print(f"Final portfolio value: ${portfolio_value[-1]:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Hit Ratio: {hit_ratio:.2f}%")

# Plot portfolio value over time
plt.plot(portfolio_series.index, portfolio_value, label='Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value (USD)')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.show()

# Display performance stats
perf = portfolio_series.calc_stats()
perf.display()