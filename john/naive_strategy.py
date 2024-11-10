from base_strategy import base_strategy

class naive_strategy(base_strategy):
    def __init__(self, starting_capital: int, starting_stocks: int):
        # WHAT MAKES THIS STRATEGY UNIQUE?
        strategy_name: str = 'naive_approach'
        batch_window: int = 3

        # FINALLY, CALL THE PARENT CONSTRUCTOR
        super().__init__(starting_capital, starting_stocks, batch_window, strategy_name)

    # DEFINE THE STRATEGY SPECIFIC BUY CONDITION
    # A naive approach might be to simply always return True with a reason
    def buy(self, latest_values: list[float], predicted_values: list[float]):
        # Naive condition: Buy if the predicted value is higher than the latest value
        if predicted_values[-1] > latest_values[-1]:
            return True, 'Predicted value is higher than the latest value'
        else:
            return False, 'Predicted value is not higher than the latest value'

    # DEFINE THE STRATEGY SPECIFIC SELL CONDITION
    def sell(self, latest_values: list[float], predicted_values: list[float]):
        # Naive condition: Sell if the predicted value is lower than the latest value
        if predicted_values[-1] < latest_values[-1]:
            return True, 'Predicted value is lower than the latest value'
        else:
            return False, 'Predicted value is not lower than the latest value'
