from signals.base_signal import BaseSignal

class BinarySignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, predicted_high):
        return False if predicted_high is None else self.sell_signal(self.current_price, predicted_high)
