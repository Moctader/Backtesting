from base_strategy import base_strategy

class different_size_windows_buy_sell_strategy(base_strategy):
    def __init__(self, starting_capital: int, starting_stocks: int, small_window: int, medium_window: int, large_window: int):
        # WHAT MAKES THIS STRATEGY UNIQUE?
        strategy_name: str = 'different_size_windows_buy_sell_strategy'
        batch_window: int = max(small_window, medium_window, large_window)

        # Store the window sizes
        self.small_window = small_window
        self.medium_window = medium_window
        self.large_window = large_window

        # FINALLY, CALL THE PARENT CONSTRUCTOR
        super().__init__(starting_capital, starting_stocks, batch_window, strategy_name)

    # DEFINE THE STRATEGY SPECIFIC BUY CONDITION
    def buy(self, latest_values: list[float], predicted_values: list[float]):
        # Calculate weighted averages for different window sizes
        latest_small_avg = self.weighted_average(latest_values[-self.small_window:])
        predicted_small_avg = self.weighted_average(predicted_values[-self.small_window:])
        latest_medium_avg = self.weighted_average(latest_values[-self.medium_window:])
        predicted_medium_avg = self.weighted_average(predicted_values[-self.medium_window:])
        latest_large_avg = self.weighted_average(latest_values[-self.large_window:])
        predicted_large_avg = self.weighted_average(predicted_values[-self.large_window:])

        # Buy if the predicted weighted average is higher than the latest weighted average for all window sizes
        if (predicted_small_avg > latest_small_avg and
            predicted_medium_avg > latest_medium_avg and
            predicted_large_avg > latest_large_avg):
            return True, 'Predicted weighted average is higher than the latest weighted average for all window sizes'
        else:
            return False, 'Predicted weighted average is not higher than the latest weighted average for all window sizes'

    # DEFINE THE STRATEGY SPECIFIC SELL CONDITION
    def sell(self, latest_values: list[float], predicted_values: list[float]):
        # Calculate weighted averages for different window sizes
        latest_small_avg = self.weighted_average(latest_values[-self.small_window:])
        predicted_small_avg = self.weighted_average(predicted_values[-self.small_window:])
        latest_medium_avg = self.weighted_average(latest_values[-self.medium_window:])
        predicted_medium_avg = self.weighted_average(predicted_values[-self.medium_window:])
        latest_large_avg = self.weighted_average(latest_values[-self.large_window:])
        predicted_large_avg = self.weighted_average(predicted_values[-self.large_window:])

        # Sell if the predicted weighted average is lower than the latest weighted average for all window sizes
        if (predicted_small_avg < latest_small_avg and
            predicted_medium_avg < latest_medium_avg and
            predicted_large_avg < latest_large_avg):
            return True, 'Predicted weighted average is lower than the latest weighted average for all window sizes'
        else:
            return False, 'Predicted weighted average is not lower than the latest weighted average for all window sizes'

    # Helper method to calculate weighted average
    def weighted_average(self, values: list[float]):
        weights = range(1, len(values) + 1)
        weighted_avg = sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
        return weighted_avg