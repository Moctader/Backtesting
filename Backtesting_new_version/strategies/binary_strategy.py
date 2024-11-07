from strategies.base_strategy import BaseStrategy
import logging

# MAYBE ADD THIS LOGGING INIT TO THE BASE_STRATEGY SO YOU DONT NEED TO INVOKE
# IT IN EVERY STRATEGY FILE?
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class BinaryStrategy(BaseStrategy):
    def execute_trade(self, signal, future_timestamp, predicted_high, current_price):
        self.current_price = current_price
        #print(f"Current Price: {self.current_price}, Buy Price: {self.buy_price}, Predicted High: {predicted_high}")

        buy_signal, sell_signal = self.generate_signals(signal, predicted_high)
        #logging.debug(f"Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")

        # Case 1: No position and buy signal is active
        if self.position == 0 and buy_signal:
            # Execute buy if no open position and buy signal is active
            self.buy(future_timestamp, predicted_high)

        # Case 2: Currently in a buy position and a sell signal or exit signal is active
        elif self.position > 0 and (sell_signal):
            # Execute sell to exit the buy position
            self.sell(future_timestamp, predicted_high)  # Close the buy position
            if sell_signal:
                # Immediately open short position if sell signal is active
                self.initiate_short_position()

        # Case 3: Currently in a short position and a buy signal or exit signal is active
        elif self.position < 0 and (buy_signal):
            # Cover short position and re-buy if buy signal is active
            self.buy_to_cover(future_timestamp, predicted_high)  # Close the short position
            if buy_signal:
                # Immediately open a buy position if buy signal is active
                self.buy(future_timestamp, predicted_high)

        self.update_portfolio_value()

    