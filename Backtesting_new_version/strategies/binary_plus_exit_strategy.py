from strategies.base_strategy import BaseStrategy
import logging

# MAYBE ADD THIS LOGGING INIT TO THE BASE_STRATEGY SO YOU DONT NEED TO INVOKE
# IT IN EVERY STRATEGY FILE?
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class BinaryPlusExitStrategy(BaseStrategy):
    def execute_trade(self, signal, future_timestamp, predicted_high, current_price):
        """Execute trades based on signals and manage positions."""
        self.current_price = current_price

        # Determine the signal strength
        if self.previous_signal == signal:
            self.signal_strength += 1
        else:
            self.signal_strength = 1
        self.previous_signal = signal

        buy_signal, sell_signal, exit_signal = self.generate_signals(signal, predicted_high)


        # Log the generated signals
        logging.debug(f"Buy Signal: {buy_signal}, Sell Signal: {sell_signal}, Exit Signal: {exit_signal}")

        # Buy decision
        if self.position == 0 and buy_signal:
            self.buy(future_timestamp, predicted_high)

        # Add to position if the signal is strong
        elif self.position > 0 and buy_signal:
            self.buy(future_timestamp, predicted_high)

        # Sell decision
        elif self.position > 0 and sell_signal:
            self.sell(future_timestamp, predicted_high)

        # Exit decision
        elif self.position > 0 and exit_signal:
            self.exit(future_timestamp, predicted_high)

        self.update_portfolio_value()