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

        buy_signal, sell_signal, exit_signal = self.generate_signals(signal, predicted_high)
        #logging.debug(f"Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")

        # Case 1: No position and buy signal is active
        if self.position == 0 and buy_signal:
            # Execute buy if no open position and buy signal is active
            self.buy(future_timestamp, predicted_high)

        # Case 2: Currently in a buy position and a sell signal or exit signal is active
        elif self.position > 0 and (sell_signal ):
            # Execute sell to exit the buy position
            self.sell(future_timestamp, predicted_high)  # Close the buy position
            if sell_signal:
                # Immediately open short position if sell signal is active
                self.initiate_short_position()

        # Case 3: Currently in a short position and a buy signal or exit signal is active
        elif self.position < 0 and (buy_signal ):
            # Cover short position and re-buy if buy signal is active
            self.buy_to_cover(future_timestamp, predicted_high)  # Close the short position
            if buy_signal:
                # Immediately open a buy position if buy signal is active
                self.buy(future_timestamp, predicted_high)

        self.update_portfolio_value()
        #logging.debug(f"Position: {self.position}, Cash: {self.cash}, Total Trades: {self.total_trades}")

    def buy(self, future_timestamp, predicted_high):
        amount_to_invest = self.position_size_factor * self.cash
        shares_to_buy = amount_to_invest // self.current_price
        self.cash -= shares_to_buy * self.current_price
        self.position += shares_to_buy
        self.buy_price = self.current_price
        self.total_trades += 1  # Update total trades count
        self.make_decision(future_timestamp, predicted_high, "buy", self.current_price)  # Log buy event
        #logging.debug(f"Buy: Position: {self.position}, Cash: {self.cash}, Total Trades: {self.total_trades}")

    def sell(self, future_timestamp, predicted_high):
        trade_value = self.position * self.current_price
        self.cash += trade_value
        profit_or_loss = trade_value - (self.position * self.buy_price)
        self.track_trade_outcome(profit_or_loss)
        self.position = 0
        self.buy_price = None  # Reset buy_price after selling
        self.total_trades += 1  # Update total trades count
        self.make_decision(future_timestamp, predicted_high, "sell", self.current_price)  # Log sell event
        #logging.debug(f"Sell: Position: {self.position}, Cash: {self.cash}, Total Trades: {self.total_trades}")

    def exit(self, future_timestamp, predicted_high):
        """Execute an exit decision."""
        trade_value = self.position * self.current_price
        self.cash += trade_value
        self.total_trades += 1
        self.position = 0
        self.buy_price = None  # Reset buy_price after exiting
        self.make_decision(future_timestamp, predicted_high, "exit", self.current_price)
        #logging.debug(f"Exit: Position: {self.position}, Cash: {self.cash}, Total Trades: {self.total_trades}")

    def buy_to_cover(self, future_timestamp, predicted_high):
        trade_value = abs(self.position) * self.current_price
        self.cash -= trade_value
        profit_or_loss = -trade_value
        self.track_trade_outcome(profit_or_loss)
        self.position = 0
        self.buy_price = None  # Reset buy_price after buying to cover
        self.total_trades += 1  # Update total trades count
        self.make_decision(future_timestamp, predicted_high, "buy_to_cover", self.current_price)  # Log buy to cover event
        #logging.debug(f"Buy to Cover: Position: {self.position}, Cash: {self.cash}, Total Trades: {self.total_trades}")

    def update_portfolio_value(self):
        portfolio_value = self.cash + self.position * self.current_price
        self.portfolio_value.append(portfolio_value)
        #logging.debug(f"Updated portfolio value: {portfolio_value}")

    def initiate_short_position(self):
        amount_to_invest = self.position_size_factor * self.cash
        shares_to_short = amount_to_invest // self.current_price
        self.cash += shares_to_short * self.current_price  # Short proceeds added to cash
        self.position -= shares_to_short
        #logging.debug(f"Initiate Short Position: Position: {self.position}, Cash: {self.cash}, Total Trades: {self.total_trades}")

    def track_trade_outcome(self, profit_or_loss):
        """Track the outcome of a trade."""
        if profit_or_loss > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.total_trades += 1
        #logging.debug(f"Track Trade Outcome: Winning Trades: {self.winning_trades}, Losing Trades: {self.losing_trades}, Total Trades: {self.total_trades}")