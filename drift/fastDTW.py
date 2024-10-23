import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Load your EUR/USD historical data
data = pd.read_csv('./EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(10000)
data1 = pd.read_csv('./EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').tail(2000)
data = data['close']
data1 = data1['close']

# Function to calculate log returns
def calculate_log_returns(prices):
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()

# Calculate log returns for both datasets
log_returns_data = calculate_log_returns(data)
log_returns_data1 = calculate_log_returns(data1)

# Convert log returns to 1-D numpy arrays
log_returns_data = log_returns_data.to_numpy().flatten()
log_returns_data1 = log_returns_data1.to_numpy().flatten()

# Print the first few log returns for verification
print("Log Returns for data:")
print(log_returns_data[:5])

print("\nLog Returns for data1:")
print(log_returns_data1[:5])

# Plot the log returns
plt.figure(figsize=(14, 7))
plt.plot(log_returns_data, label='Log Returns for data')
plt.plot(log_returns_data1, label='Log Returns for data1')
plt.xlabel('Time')
plt.ylabel('Log Return')
plt.title('Log Returns Over Time')
plt.legend()
plt.show()

# Calculate the dynamic time warping distance between the two log return series
distance, path = fastdtw(log_returns_data.reshape(-1, 1), log_returns_data1.reshape(-1, 1), dist=euclidean)

# Print the DTW distance
print(f"DTW Distance: {distance}")
