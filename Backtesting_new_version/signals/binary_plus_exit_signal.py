from signals.base_signal import BaseSignal

class BinaryPlusExitSignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price) if buy_price else False

    def generate_exit_signal(self, buy_price, predicted_high):
        return self.exit_signal(self.current_price, buy_price, predicted_high) if buy_price and predicted_high else False
