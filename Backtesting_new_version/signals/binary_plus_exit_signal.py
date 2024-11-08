from signals.base_signal import BaseSignal

class BinaryPlusExitSignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, predicted_high):
        return self.sell_signal(self.current_price, predicted_high) if predicted_high else False

    def generate_exit_signal(self, predicted_high):
        if self.exit_signal and self.current_price is not None and predicted_high is not None and predicted_high is not None:
            return self.exit_signal(self.current_price, predicted_high)
        return False