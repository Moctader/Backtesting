Data Preparation:
    Download historical stock data for a given ticker (AAPL) using yfinance.
    Create new features and target variable for the model.
    Normalize the features and target variable.
    Create a dataset function for the LSTM model.
    Split the data into training and testing sets.


Build and Train the LSTM Model:
    Define and compile an LSTM model using Keras.
    Train the model on the training data with early stopping.

Trading Strategy Implementation:
    Initialize parameters for the trading strategy.
    Define buy and sell conditions based on the model's predictions.
    Simulate trading over the test period and track portfolio value.

Performance Metrics:
    Calculate performance metrics such as Sharpe Ratio and Max Drawdown.
    Calculate the hit ratio of the trades.
    Plot the portfolio value over time.
    Display performance statistics.

