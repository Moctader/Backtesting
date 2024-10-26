import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Load your EUR/USD historical data
data = pd.read_csv('../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(14)
data1 = pd.read_csv('../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').tail(15)
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

# Calculate the dynamic time warping distance between the two log return series
distance, path = fastdtw(log_returns_data.reshape(-1, 1), log_returns_data1.reshape(-1, 1), dist=euclidean)

# Print the DTW distance
print(f"DTW Distance: {distance}")

# Plot the log returns
plt.figure(figsize=(14, 7))
plt.plot(log_returns_data, label='Log Returns for data', color='blue')
plt.plot(log_returns_data1, label='Log Returns for data1', color='orange')
plt.title('Log Returns with DTW Warping Path')
plt.xlabel('Time Index')
plt.ylabel('Log Return')

# Overlay the warping path
for (map_x, map_y) in path:
    plt.plot([map_x, map_y], [log_returns_data[map_x], log_returns_data1[map_y]], color='red', alpha=0.5)

plt.legend()
plt.show()
