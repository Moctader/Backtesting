from signals.base_signal import BaseSignal

class BinarySignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price) if buy_price else False