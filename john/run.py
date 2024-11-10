import torch
import torch.nn as nn
import pandas as pd
#from cool_strategy import cool_strategy
from naive_strategy import naive_strategy
from conservative_buy_sell import conservative_buy_sell_strategy
from weighted_buy_sell import weighted_buy_sell_strategy
from different_size_windows import different_size_windows_buy_sell_strategy
from trend_based_Strategy import trend_based_Strategy   
from conditional_buy_sell_strategy import conditional_buy_sell_strategy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the model architecture
class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get output from the last LSTM time step
        return out

# Define model parameters
input_size = 9  # Adjust according to your data
hidden_size = 50
num_layers = 1
output_size = 1

# Initialize and load model
model = MyLSTMModel(input_size, hidden_size, num_layers, output_size)
model_path = "/Users/moctader/TrustworthyAI/backtesting/Backtesting/john/lstm_model.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()  # Set to evaluation mode

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp').head(100)
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))
    data['Next_High'] = data['high'].shift(-1)
    data['Return'] = data['close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=5).std()
    data['High_Low_Range'] = (data['high'] - data['low']) / data['low']
    data['Prev_Close_Rel_High'] = (data['close'].shift(1) - data['high']) / data['high']
    data.dropna(inplace=True)
    return data

data = load_data('/Users/moctader/TrustworthyAI/backtesting/EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(50)

# Select features and preprocess data
features = data[['open', 'high', 'low', 'close', 'volume', 'Return', 'Volatility', 'High_Low_Range', 'Prev_Close_Rel_High']].values

# Initialize the scaler (assuming the same scaler used during training)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(data[['Next_High']].dropna())

# Make predictions for each row
predictions = []
with torch.no_grad():
    for i in range(len(features)):
        feature_tensor = torch.tensor(features[i:i+1], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, input_size)
        prediction = model(feature_tensor)  # Output shape: (1, output_size)
        prediction_value = prediction.item()  # Get a scalar value if output_size=1
        predictions.append(prediction_value)

# Inverse transform the predictions to real values
predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Add predictions to the dataset
data['Prediction'] = np.nan
data.iloc[:len(predictions), data.columns.get_loc('Prediction')] = predictions

# Save the new dataset to a file
new_dataset_path = '/Users/moctader/TrustworthyAI/backtesting/EODHD_EURUSD_HISTORICAL_2019_2024_1min_with_predictions.csv'
data.to_csv(new_dataset_path)

# Load the new dataset
new_dataset = pd.read_csv(new_dataset_path)
print(len(new_dataset))

# Initialize strategy with the model
buy_threshold = 0.01
sell_threshold = 0.02

#strat = conservative_buy_sell_strategy(1000, 0, buy_threshold, sell_threshold)
#strat=weighted_buy_sell_strategy(1000, 0)
#strat=different_size_windows_buy_sell_strategy(starting_capital=1000, starting_stocks=0, small_window=3, medium_window=5, large_window=7)
#strat=trend_based_Strategy(starting_capital=1000, starting_stocks=0, window_size=10) 
strat=conditional_buy_sell_strategy(starting_capital=1000, starting_stocks=0)

# Make decisions based on predictions
for i in range(len(new_dataset)):
    strat.make_decision(new_dataset['close'][i], new_dataset['Prediction'][i])

# Create log to track the strategy's actions
strat.create_log()