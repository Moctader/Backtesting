from base_strategy import base_strategy

class weighted_buy_sell_strategy(base_strategy):
    def __init__(self, starting_capital: int, starting_stocks: int):
        # WHAT MAKES THIS STRATEGY UNIQUE?
        strategy_name: str = 'weighted_buy_sell_strategy'
        batch_window: int = 3

        # FINALLY, CALL THE PARENT CONSTRUCTOR
        super().__init__(starting_capital, starting_stocks, batch_window, strategy_name)

    # DEFINE THE STRATEGY SPECIFIC BUY CONDITION
    def buy(self, latest_values: list[float], predicted_values: list[float]):
        # Calculate weighted average of latest values and predicted values
        latest_weighted_avg = self.weighted_average(latest_values)
        predicted_weighted_avg = self.weighted_average(predicted_values)

        # Conservative condition: Buy if the predicted weighted average is higher than the latest weighted average by a certain percentage
        if predicted_weighted_avg > latest_weighted_avg:
            return True, f'Predicted weighted average is higher than the latest weighted average'
        else:
            return False, f'Predicted weighted average is not higher than the latest weighted average'

    # DEFINE THE STRATEGY SPECIFIC SELL CONDITION
    def sell(self, latest_values: list[float], predicted_values: list[float]):
        # Calculate weighted average of latest values and predicted values
        latest_weighted_avg = self.weighted_average(latest_values)
        predicted_weighted_avg = self.weighted_average(predicted_values)

        # Conservative condition: Sell if the predicted weighted average is lower than the latest weighted average by a certain percentage
        if predicted_weighted_avg < latest_weighted_avg:
            return True, f'Predicted weighted average is lower than the latest weighted average'
        else:
            return False, f'Predicted weighted average is not lower than the latest weighted average'

    # Helper method to calculate weighted average
    def weighted_average(self, values: list[float]):
        weights = range(1, len(values) + 1)
        weighted_avg = sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
        return weighted_avg