from abc import ABC, abstractmethod

class BaseSignal(ABC):
    def __init__(self, current_price, buy_signal, sell_signal, exit_signal=None):
        self.current_price = current_price
        self.buy_signal = eval(buy_signal["function"])
        self.sell_signal = eval(sell_signal["function"])
        self.exit_signal = eval(exit_signal["function"]) if exit_signal else None

    @abstractmethod
    def generate_buy_signal(self, predicted_high):
        pass

    @abstractmethod
    def generate_sell_signal(self, buy_price):
        pass

    def generate_exit_signal(self, buy_price, predicted_high):
        if self.exit_signal and buy_price is not None and predicted_high is not None:
            return self.exit_signal(self.current_price, buy_price, predicted_high)
        return False