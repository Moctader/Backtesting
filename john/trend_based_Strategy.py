from base_strategy import base_strategy

class trend_based_Strategy(base_strategy):
    def __init__(self, starting_capital: int, starting_stocks: int, window_size: int):
        # WHAT MAKES THIS STRATEGY UNIQUE?
        strategy_name: str = 'trend_based_Strategy'
        batch_window: int = window_size

        # Store the window size
        self.window_size = window_size

        # FINALLY, CALL THE PARENT CONSTRUCTOR
        super().__init__(starting_capital, starting_stocks, batch_window, strategy_name)

    # DEFINE THE STRATEGY SPECIFIC BUY CONDITION
    def buy(self, latest_values: list[float], predicted_values: list[float]):
        # Buy if the trend is up (each subsequent predicted value is greater than the previous one)
        if all(predicted_values[i] > predicted_values[i - 1] for i in range(1, self.window_size)):
            return True, 'Trend is up (each subsequent predicted value is greater than the previous one)'
        else:
            return False, 'Trend is not up'

    # DEFINE THE STRATEGY SPECIFIC SELL CONDITION
    def sell(self, latest_values: list[float], predicted_values: list[float]):
        # Sell if the trend is down (each subsequent predicted value is less than the previous one)
        if all(predicted_values[i] < predicted_values[i - 1] for i in range(1, self.window_size)):
            return True, 'Trend is down (each subsequent predicted value is less than the previous one)'
        else:
            return False, 'Trend is not down'