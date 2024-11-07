from strategies.base_strategy import BaseStrategy
import logging

# Initialize logging (consider moving this to BaseStrategy or the main entry point)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class BinaryStrategy(BaseStrategy):
    def execute_trade(self, signal, future_timestamp, predicted_high, current_price):
        self.current_price = current_price
        buy_signal, sell_signal = self.generate_signals(signal, predicted_high)
        
        logging.debug(f"buy_signal={buy_signal}, sell_signal={sell_signal}, position={self.position}")
        
        # Case 1: No position and buy signal is active - initiate a buy position
        if self.position == 0 and buy_signal:
            self.buy(future_timestamp, predicted_high)
            self.position = 1  # Set position to indicate a buy position
            logging.info("Bought position initiated.")

        # Case 2: Currently holding a buy position, and sell signal is active
        elif self.position > 0 and (sell_signal):
            self.sell(future_timestamp, predicted_high)
            self.position = 0  # Reset position to zero after selling
            logging.info("Sell position executed.")
            if sell_signal:
                self.initiate_short_position()
                logging.info("Short position initiated.")

        # Case 3: Currently in a short position and a buy signal is active
        elif self.position <= 0 and buy_signal:
            self.buy_to_cover(future_timestamp, predicted_high)
            self.position = 0  # Reset position to zero after buying to cover
            logging.info("Short position covered.")
            # Optionally start a new buy position
            if buy_signal:
                self.buy(future_timestamp, predicted_high)
                self.position = 1  # Set position to indicate a buy position
                logging.info("New buy position initiated after covering short.")
        # Update portfolio value
        self.update_portfolio_value()
