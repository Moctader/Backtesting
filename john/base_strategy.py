import json, yaml, time

class base_strategy:
    def __init__(self, starting_capital: int, starting_stocks: int, batch_window: int, strategy_name: str):
        self._capital: int = starting_capital
        self._stock_count: int = starting_stocks
        self._batch_window: int = batch_window
        self._strategy_name: str = strategy_name

        # SAVE INITIAL VALUES FOR LOGGING PURPOSES
        self._init_capital: int = starting_capital
        self._init_stock_count: int = starting_stocks
        self._init_stock_value: float = None 
        self._final_stock_value: float = None

        # BATCH WINDOWS FOR DECISIONMAKING
        self._value_batch: list[float] = []
        self._prediction_batch: list[float] = []

        # TRACK MADE DECISIONS -- DECISION TAG => REASON
        self._decisions: list[dict] = []

    ################################################################################################
    ################################################################################################

    # PRETTYPRINT DICTS WHEN DEBUGGING
    def pprint(self, data: dict):
        if True: print(json.dumps(data, indent=4))

    ################################################################################################
    ################################################################################################

    # SAVE DECISION LOG AS YAML FILE
    def create_log(self):

        # GENERATE UNIX TIMESTAMP TO MAKE FILENAME UNIQUE
        now: int = int(time.time())
        filename: str = f'{now}_{self._strategy_name}.yaml'

        decision_count = {
            'buy': 0,
            'sell': 0,
            'hold': 0
        }

        # COUNT DECISIONS
        for item in self._decisions:
            decision_count[item['decision']] += 1
        # STITCH TOGETHER ALL LOGDATA
        audit_dict = {
            'args': {
                'strategy_name': self._strategy_name,
                'batch_window': self._batch_window,
            },
            'init': {
                'capital': float(self._init_capital),
                'stocks': float(self._init_stock_count),
                'stock_value': float(self._init_stock_value),
                'liquid': float(self._init_capital + (self._init_stock_count * self._init_stock_value))
            },
            'final': {
                'capital': float(self._capital),
                'stocks': self._stock_count,
                'stock_value': float(self._final_stock_value),
                'liquid': float(self._capital + (self._stock_count * self._final_stock_value))
            },
            'n_decisions': decision_count,
            'decision_sequence': self._decisions
        }
    
        # SAVE YAML FILE
        with open(filename, 'w') as file:
            yaml.dump(audit_dict, file, default_flow_style=False, sort_keys=False, indent=4)

    ################################################################################################
    ################################################################################################

    # LET STRATEGY MAKE DECISION
    def make_decision(self, latest_known_value: float, predicted_value: float):

        # SAVE FIRST STOCK VALUE FOR LOGGING PURPOSES
        if self._init_stock_value == None:
            self._init_stock_value = latest_known_value

        # ADD NEW VALUES TO CONTAINERS
        self._value_batch.append(latest_known_value)
        self._prediction_batch.append(predicted_value)
        self._final_stock_value = latest_known_value

        # CANT MAKE DECISION YET -- STILL COLLECTING INITIAL BATCHES
        if len(self._prediction_batch) < self._batch_window:
            return self.pprint({
                'warning': f'no decision yet, collectin initial batch window (n={self._batch_window})'
            })

        # TRUNCATE CONTAINERS TO KEEP SEQUENTIAL ROLLING WINDOW
        if len(self._prediction_batch) > self._batch_window:
            self._prediction_batch = self._prediction_batch[1:]
            self._value_batch = self._value_batch[1:]

    ################################################################################################
    ################################################################################################

        # VERIFY WHETHER WE WANT TO BUY OR SELL 
        buy_decision, buy_reason = self.buy(self._value_batch, self._prediction_batch)
        sell_decision, sell_reason = self.sell(self._value_batch, self._prediction_batch)

        # PREVENT SIMULTANOUS BUYING AND SELLING
        if buy_decision and sell_decision:
            return self.pprint({
                'error': 'strategy wants to buy AND sell for same prediction',
                'buy_reason': buy_reason,
                'sell_reason': sell_reason,
            })

    ################################################################################################
    ################################################################################################

        # REGISTER BUY
        if buy_decision:

            # BLOCK BUYS WHEN WE DONT HAVE ENOUGH CAPITAL
            if self._capital < latest_known_value:
                return self.pprint({
                    'error': 'buy blocked due to lack of capital'
                })
            
            # CREATE WATERMARK FOR LOGGING
            buy_watermark = { 'decision': 'buy', 'reason': buy_reason}

            # OTHERWISE, REGISTER DECISION AND UPDATE STATE
            self._decisions.append(buy_watermark)
            self._capital -= latest_known_value
            self._stock_count += 1

            return self.pprint(buy_watermark)

    ################################################################################################
    ################################################################################################

        # REGISTER SELL
        if sell_decision:

            # BLOCK SELLS WHEN WE DONT HAVE ENOUGH STOCKS
            if self._stock_count == 0:
                return self.pprint({
                    'error': 'sell prevented due to lack of owned stocks'
                })

            # CREATE WATERMARK FOR LOGGING
            sell_watermark = { 'decision': 'sell', 'reason': sell_reason }

            # OTHERWISE, REGISTER DECISION AND UPDATE STATE
            self._decisions.append(sell_watermark)
            self._capital += latest_known_value
            self._stock_count -= 1

            return self.pprint(sell_watermark)
        
    ################################################################################################
    ################################################################################################

        # OTHERWISE, DO NOTHING
        hold_watermark = { 'decision': 'hold' }
        self._decisions.append(hold_watermark)
        self.pprint(hold_watermark)

    ################################################################################################
    ################################################################################################

    # RETURNS TRUE/FALSE, AND A HUMAN-READABLE REASON
    def buy(self, latest_values: list, predicted_values: list):
        raise NotImplementedError()
    
    # RETURNS TRUE/FALSE, AND A HUMAN-READABLE REASON
    def sell(self, latest_values: list, predicted_values: list):
        raise NotImplementedError()