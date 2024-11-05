import yaml
import logging
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
        pass




class BinarySignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price) if buy_price else False
    



class BinaryPlusExitSignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price) if buy_price else False

    def generate_exit_signal(self, buy_price, predicted_high):
        return self.exit_signal(self.current_price, buy_price, predicted_high) if buy_price and predicted_high else False
    


class MulticlassSignal(BaseSignal):
    def __init__(self, current_price, buy_signal, sell_signal, exit_signal, quartiles):
        super().__init__(current_price, buy_signal, sell_signal, exit_signal)
        self.quartiles = quartiles

    def generate_signal(self, predicted_high):
        if self.current_price < self.quartiles[0]:
            return "Buy"
        elif self.current_price > self.quartiles[2]:
            return "Sell"
        else:
            return "Hold"

    def generate_buy_signal(self, predicted_high):
        return self.generate_signal(predicted_high) == "Buy"

    def generate_sell_signal(self, buy_price):
        return self.generate_signal(buy_price) == "Sell"

    def generate_exit_signal(self, buy_price, predicted_high):
        return self.generate_signal(predicted_high) == "Hold"