import pandas as pd
import torch

# Step 1: Create a pandas.Series of minute-based timestamps
timestamps = pd.date_range(start="2023-01-01 12:00", periods=5, freq="T")
timestamp_series = pd.Series(timestamps)
print("Original Timestamp Series:")
print(timestamp_series)

# Step 2: Convert pandas.Timestamp series to torch.Tensor (epoch times)
# Convert timestamps to epoch seconds (int64) for compatibility with Torch
epoch_times = timestamp_series.astype('int64') // 10**9  # Convert to seconds
torch_tensor = torch.tensor(epoch_times.values, dtype=torch.float32)
print("\nTorch Tensor (Epoch Times):")
print(torch_tensor)

# Step 3: Convert back from torch.Tensor to pandas.Timestamp series
# Convert epoch times (seconds) back to pandas.Timestamp
converted_back_times = pd.to_datetime(torch_tensor.numpy(), unit="s")
converted_back_series = pd.Series(converted_back_times)
print("\nConverted Back to Timestamp Series:")
print(converted_back_series)