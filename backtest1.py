# Import necessary libraries
import ffn
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


# Fetch stock data from Yahoo Finance
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
data.columns=data.columns.droplevel(1)  

# Remove rows with NaN or infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Adj Close'])

# Preprocess data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

# Create dataset function for LSTM
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time step for LSTM model
time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[early_stop], verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to get actual prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Combine train and test predictions
predicted_prices = np.concatenate((train_predict, test_predict), axis=0)
predicted_prices_df = pd.DataFrame(predicted_prices, index=data.index[time_step + 1:], columns=['Predicted Price'])

# Calculate rolling performance metrics using ffn
window = 252  # 1 year of trading days

predicted_prices_df['Returns'] = predicted_prices_df['Predicted Price'].pct_change()
predicted_prices_df['Sharpe'] = predicted_prices_df['Returns'].rolling(window).apply(lambda x: ffn.calc_sharpe(x.dropna()), raw=False)
predicted_prices_df['Sortino'] = predicted_prices_df['Returns'].rolling(window).apply(lambda x: ffn.calc_sortino_ratio(x.dropna()), raw=False)
predicted_prices_df['CAGR'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: ffn.calc_cagr(x.dropna()), raw=False)
predicted_prices_df['Total Return'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: ffn.calc_total_return(x.dropna()), raw=False)
predicted_prices_df['Max Drawdown'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: ffn.calc_max_drawdown(x.dropna()), raw=False)
predicted_prices_df['YTD'] = predicted_prices_df['Predicted Price'].pct_change().cumsum()

# Generate trading signals based on performance metrics
def generate_signals(df):
    signals = []
    for i in range(len(df)):
        if df['Sharpe'].iloc[i] > 0.2 and df['Sortino'].iloc[i] > 0.2 and df['CAGR'].iloc[i] >= 0.0 and df['YTD'].iloc[i] >= 0.0:
            signals.append('Buy')
        elif df['Total Return'].iloc[i] < 0 or df['Max Drawdown'].iloc[i] > 0.1:
            signals.append('Sell')
        else:
            signals.append('Exit')
    return signals

# Apply the performance metrics strategy to generate signals
predicted_prices_df['Signal'] = generate_signals(predicted_prices_df)

# Define the SimpleStrategy class
class SimpleStrategy:
    def apply_strategy(self, times, prices, signals):
        data = pd.DataFrame(index=times)
        data['Price'] = prices
        data['Signal'] = signals

        positions = []
        position = 0
        for i in range(len(data)):
            signal = data['Signal'].iloc[i]
            if signal == 'Buy':
                if position >= 0:
                    position += 1
                else:
                    position = 1
            elif signal == 'Sell':
                if position <= 0:
                    position -= 1
                else:
                    position = -1
            elif signal == 'Exit':
                position = 0
            positions.append(position)
        data['Position'] = positions

        data['Price_Change'] = data['Price'].pct_change()
        data['Position_shifted'] = data['Position'].shift(1)
        data['Position_shifted'].fillna(0, inplace=True)
        data['Strategy_Returns'] = data['Position_shifted'] * data['Price_Change']
        return data

# Apply the SimpleStrategy to the predicted prices
simple_strategy = SimpleStrategy()
strategy_data = simple_strategy.apply_strategy(predicted_prices_df.index, predicted_prices_df['Predicted Price'], predicted_prices_df['Signal'])

# Print the strategy data
print(strategy_data)

# Calculate hit ratio
# Determine the actual price movements
predicted_prices_df['Actual Movement'] = np.sign(predicted_prices_df['Predicted Price'].diff())

# Map signals to numerical values
signal_map = {'Buy': 1, 'Sell': -1, 'Exit': 0}
predicted_prices_df['Signal Numeric'] = predicted_prices_df['Signal'].map(signal_map)

# Compare predicted signals with actual movements
predicted_prices_df['Hit'] = predicted_prices_df['Signal Numeric'] == predicted_prices_df['Actual Movement']

# Calculate hit ratio
hit_ratio = predicted_prices_df['Hit'].mean()
print(f"Hit Ratio: {hit_ratio:.2f}")

# Normalize Max Drawdown to the range of Adj Close prices
min_price = data['Adj Close'].min()
max_price = data['Adj Close'].max()
predicted_prices_df['Normalized Max Drawdown'] = (predicted_prices_df['Max Drawdown'] - predicted_prices_df['Max Drawdown'].min()) / (predicted_prices_df['Max Drawdown'].max() - predicted_prices_df['Max Drawdown'].min()) * (max_price - min_price) + min_price

# Plot stock prices with trading signals and Max Drawdown
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot actual and predicted stock prices on the primary y-axis
ax1.plot(data['Adj Close'], label='Actual Prices', color='blue')
ax1.plot(predicted_prices_df.index, predicted_prices_df['Predicted Price'], label='Predicted Prices', color='orange')

# Highlight buy/sell signals
buy_signals = strategy_data[strategy_data['Signal'] == 'Buy']
sell_signals = strategy_data[strategy_data['Signal'] == 'Sell']
ax1.scatter(buy_signals.index, buy_signals['Price'], marker='^', color='green', label='Buy Signal', s=100)
ax1.scatter(sell_signals.index, sell_signals['Price'], marker='v', color='red', label='Sell Signal', s=100)

# Secondary y-axis for Normalized Max Drawdown
# ax2 = ax1.twinx()
# ax2.plot(predicted_prices_df.index, predicted_prices_df['Normalized Max Drawdown'], label='Normalized Max Drawdown', color='purple', linestyle='--')
# ax2.set_ylabel('Normalized Max Drawdown', color='purple')
# ax2.tick_params(axis='y', labelcolor='purple')

# Set title and labels
ax1.set_title('AAPL Stock Price with Trading Signals and Normalized Max Drawdown')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Add legends for both y-axes
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
fig.tight_layout()

# Show the plot
plt.show()