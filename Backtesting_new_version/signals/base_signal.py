from abc import ABC, abstractmethod
import logging

class BaseSignal(ABC):
    def __init__(self, current_price, buy_signal, sell_signal, exit_signal=None):
        self.current_price = current_price
        self.buy_signal = eval(buy_signal["function"])
        self.sell_signal = eval(sell_signal["function"])
        self.exit_signal = eval(exit_signal["function"]) if exit_signal else None

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    @abstractmethod
    def generate_buy_signal(self, predicted_high):
        pass

    @abstractmethod
    def generate_sell_signal(self, buy_price):
        pass

    def generate_exit_signal(self, buy_price, predicted_high):
        return False