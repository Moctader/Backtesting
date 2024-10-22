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
data.columns = data.columns.droplevel(1)

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
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64, callbacks=[early_stop], verbose=1)

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

# Calculate moving average deviation
def moving_average_deviation(data, window=3):
    moving_avg = data.rolling(window=window).mean()
    deviation = data - moving_avg
    return deviation

class MovingAverageSignalGenerator:
    def generate_signals(self, y_pred, window=3, upper_threshold=0.000051, lower_threshold=-0.000051):
        deviation = moving_average_deviation(pd.Series(y_pred.flatten()), window)
        signals = []
        for dev in deviation:
            if dev > upper_threshold:
                signals.append('Buy')
            elif dev < lower_threshold:
                signals.append('Sell')
            else:
                signals.append('Exit')
        return signals

# Define the SimpleStrategy class
class SimpleStrategy:
    def __init__(self, y_test_actual, y_pred):
        self.y_test_actual = y_test_actual
        self.y_pred = y_pred

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

    def plot_trading_signals(self, strategy_data, y_test_actual_series, y_pred_series):
        buy_signals = strategy_data[strategy_data['Signal'] == 'Buy']
        sell_signals = strategy_data[strategy_data['Signal'] == 'Sell']
        exit_signals = strategy_data[strategy_data['Signal'] == 'Exit']

        plt.figure(figsize=(12, 6))
        y_test_actual_series.plot(label='Actual', color='blue')
        y_pred_series.plot(label='Predicted', color='orange')

        plt.scatter(buy_signals.index, buy_signals['Price'], marker='^', color='green', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['Price'], marker='v', color='red', label='Sell Signal')
        plt.scatter(exit_signals.index, exit_signals['Price'], marker='o', color='blue', label='Exit Signal')

        plt.legend()
        plt.title('Trading Signals')
        plt.show()

    def calculate_performance(self, strategy_data):
        initial_capital = 10000
        strategy_data['Strategy_Returns'].fillna(0, inplace=True)
        strategy_data['Cumulative_Returns'] = (1 + strategy_data['Strategy_Returns']).cumprod()
        strategy_data['Equity'] = initial_capital * strategy_data['Cumulative_Returns']
        return strategy_data

# Create signals and apply the strategy
signal_generator = MovingAverageSignalGenerator()
signals = signal_generator.generate_signals(predicted_prices_df['Predicted Price'].values)

strategy = SimpleStrategy(y_test_actual, test_predict)
strategy_data = strategy.apply_strategy(predicted_prices_df.index, predicted_prices_df['Predicted Price'], signals)
print(strategy_data)

# Plot trading signals
strategy.plot_trading_signals(
    strategy_data,
    y_test_actual_series=pd.Series(
        y_test_actual.flatten(), 
        index=predicted_prices_df.index[-len(y_test_actual):]
    ),
    y_pred_series=pd.Series(
        predicted_prices_df['Predicted Price'].values[-len(y_test_actual):], 
        index=predicted_prices_df.index[-len(y_test_actual):]
    )
)

# Calculate performance metrics using ffn
window = 252  # 1 year of trading days
predicted_prices_df['Returns'] = predicted_prices_df['Predicted Price'].pct_change()

predicted_prices_df['Sharpe'] = predicted_prices_df['Returns'].rolling(window).apply(lambda x: ffn.calc_sharpe(x.dropna()), raw=False)
predicted_prices_df['Sortino'] = predicted_prices_df['Returns'].rolling(window).apply(lambda x: ffn.calc_sortino_ratio(x.dropna()), raw=False)
predicted_prices_df['CAGR'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: ffn.calc_cagr(x.dropna()), raw=False)
predicted_prices_df['Total Return'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: ffn.calc_total_return(x.dropna()), raw=False)
predicted_prices_df['Max Drawdown'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: ffn.calc_max_drawdown(x.dropna()), raw=False)
predicted_prices_df['YTD'] = predicted_prices_df['Predicted Price'].pct_change().cumsum()

# Calculate hit ratio
def calculate_hit_ratio(strategy_data):
    total_trades = len(strategy_data['Strategy_Returns'].dropna())
    winning_trades = (strategy_data['Strategy_Returns'] > 0).sum()
    hit_ratio = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    return hit_ratio

# Example usage
hit_ratio = calculate_hit_ratio(strategy_data)
print(f"Hit Ratio: {hit_ratio:.2f}%")