from base_strategy import base_strategy

class conservative_buy_sell_strategy(base_strategy):
    def __init__(self, starting_capital: int, starting_stocks: int, buy_threshold: float, sell_threshold: float):
        # WHAT MAKES THIS STRATEGY UNIQUE?
        strategy_name: str = 'conservative_buy_sell_strategy'
        batch_window: int = 3

        # Store the thresholds
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # FINALLY, CALL THE PARENT CONSTRUCTOR
        super().__init__(starting_capital, starting_stocks, batch_window, strategy_name)

    # DEFINE THE STRATEGY SPECIFIC BUY CONDITION
    def buy(self, latest_values: list[float], predicted_values: list[float]):
        # Conservative condition: Buy if the predicted value is higher than the latest value by a certain percentage
        latest_value = latest_values[-1]
        predicted_value = predicted_values[-1]
        if predicted_value > latest_value * (1 + self.buy_threshold):
            return True, f'Predicted value is higher than the latest value by more than {self.buy_threshold * 100}%'
        else:
            return False, f'Predicted value is not higher than the latest value by more than {self.buy_threshold * 100}%'

    # DEFINE THE STRATEGY SPECIFIC SELL CONDITION
    def sell(self, latest_values: list[float], predicted_values: list[float]):
        # Conservative condition: Sell if the predicted value is lower than the latest value by a certain percentage
        latest_value = latest_values[-1]
        predicted_value = predicted_values[-1]
        if predicted_value < latest_value * (1 - self.sell_threshold):
            return True, f'Predicted value is lower than the latest value by more than {self.sell_threshold * 100}%'
        else:
            return False, f'Predicted value is not lower than the latest value by more than {self.sell_threshold * 100}%'