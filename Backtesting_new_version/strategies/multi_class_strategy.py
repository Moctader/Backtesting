from strategies.base_strategy import BaseStrategy
import logging

# MAYBE ADD THIS LOGGING INIT TO THE BASE_STRATEGY SO YOU DONT NEED TO INVOKE
# IT IN EVERY STRATEGY FILE?
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class MulticlassStrategy(BaseStrategy):
    def execute_trade(self, signal, future_timestamp, predicted_high, current_price):
        """Execute trades based on signals and manage positions."""
        self.current_price = current_price

        # Generate signals
        buy_signal, sell_signal, exit_signal = self.generate_signals(signal, predicted_high)

        # Log the generated signals
        logging.debug(f"Buy Signal: {buy_signal}, Sell Signal: {sell_signal}, Exit Signal: {exit_signal}")

        # Buy decision
        if self.position == 0 and buy_signal:
            logging.debug(f"Executing buy at {future_timestamp} with current price: {self.current_price}")
            self.buy(future_timestamp, predicted_high)

        # Sell decision
        elif self.position > 0 and sell_signal:
            logging.debug(f"Executing sell at {future_timestamp} with current price: {self.current_price}")
            self.sell(future_timestamp, predicted_high)

        # Exit decision
        elif self.position > 0 and exit_signal:
            logging.debug(f"Executing exit at {future_timestamp} with current price: {self.current_price}")
            self.exit(future_timestamp, predicted_high)

        # Buy to cover short position
        elif self.position < 0 and buy_signal:
            logging.debug(f"Executing buy to cover at {future_timestamp} with current price: {self.current_price}")
            self.buy_to_cover(future_timestamp, predicted_high)

        else:
            logging.debug(f"No trade executed at {future_timestamp} with current price: {self.current_price}")

        self.update_portfolio_value()