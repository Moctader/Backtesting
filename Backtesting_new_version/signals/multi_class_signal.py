from signals.base_signal import BaseSignal

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