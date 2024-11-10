from base_strategy import base_strategy

class conditional_buy_sell_strategy(base_strategy):
    def __init__(self, starting_capital: int, starting_stocks: int):
        # WHAT MAKES THIS STRATEGY UNIQUE?
        strategy_name: str = 'conditional_buy_sell_strategy'
        batch_window: int = 3

        # FINALLY, CALL THE PARENT CONSTRUCTOR
        super().__init__(starting_capital, starting_stocks, batch_window, strategy_name)

    # DEFINE THE STRATEGY SPECIFIC BUY CONDITION
    def buy(self, latest_values: list[float], predicted_values: list[float]):
        # Calculate weighted average of latest values and predicted values
        latest_weighted_avg = self.weighted_average(latest_values)
        predicted_weighted_avg = self.weighted_average(predicted_values)

        # Check the trend of the predicted values
        trend_up = all(predicted_values[i] > predicted_values[i - 1] for i in range(1, len(predicted_values)))

        # Conservative condition: Buy if the predicted weighted average is higher than the latest weighted average and the trend is up
        if predicted_weighted_avg > latest_weighted_avg and trend_up:
            # Additional condition: Ensure the capital is more than half of the starting capital and the stock quantity is lower than it was earlier
            if self._capital > self._init_capital / 2 and self._stock_count < self._init_stock_count:
                return True, 'Predicted weighted average is higher than the latest weighted average, the trend is up, and the capital is more than half of the starting capital with lower stock quantity'
            else:
                return False, 'Capital is not more than half of the starting capital or stock quantity is not lower than it was earlier'
        else:
            return False, 'Predicted weighted average is not higher than the latest weighted average or the trend is not up'

    # DEFINE THE STRATEGY SPECIFIC SELL CONDITION
    def sell(self, latest_values: list[float], predicted_values: list[float]):
        # Calculate weighted average of latest values and predicted values
        latest_weighted_avg = self.weighted_average(latest_values)
        predicted_weighted_avg = self.weighted_average(predicted_values)

        # Check the trend of the predicted values
        trend_down = all(predicted_values[i] < predicted_values[i - 1] for i in range(1, len(predicted_values)))

        # Conservative condition: Sell if the predicted weighted average is lower than the latest weighted average and the trend is down
        if predicted_weighted_avg < latest_weighted_avg and trend_down:
            # Additional condition: Ensure the capital is more than half of the starting capital and the stock quantity is lower than it was earlier
            if self._capital > self._init_capital / 2 and self._stock_count < self._init_stock_count:
                return True, 'Predicted weighted average is lower than the latest weighted average, the trend is down, and the capital is more than half of the starting capital with lower stock quantity'
            else:
                return False, 'Capital is not more than half of the starting capital or stock quantity is not lower than it was earlier'
        else:
            return False, 'Predicted weighted average is not lower than the latest weighted average or the trend is not down'

    # Helper method to calculate weighted average
    def weighted_average(self, values: list[float]):
        weights = range(1, len(values) + 1)
        weighted_avg = sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
        return weighted_avg