import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ffn
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional
from abc import ABC, abstractmethod



class BaseSignal(ABC):
    def __init__(self, current_price, buy_signal, sell_signal, exit_signal):
        self.current_price = current_price
        self.buy_signal = eval(buy_signal["function"])
        self.sell_signal = eval(sell_signal["function"])
        self.exit_signal = eval(exit_signal["function"])

    @abstractmethod
    def generate_buy_signal(self, predicted_high):
        pass

    @abstractmethod
    def generate_sell_signal(self, buy_price):
        pass

    @abstractmethod
    def generate_exit_signal(self, buy_price, predicted_high):
        pass


class Signal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price) if buy_price else False

    def generate_exit_signal(self, buy_price, predicted_high):
        return self.exit_signal(self.current_price, buy_price, predicted_high) if buy_price else False
    

class BinarySignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price) if buy_price else False

    def generate_exit_signal(self, buy_price, predicted_high):
        return self.exit_signal(self.current_price, buy_price, predicted_high) if buy_price else False  
    

class BinaryPlusExitSignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price) if buy_price else False

    def generate_exit_signal(self, buy_price, predicted_high):
        return self.exit_signal(self.current_price, buy_price, predicted_high) if buy_price else False
    


class MulticlassSignal(BaseSignal):
    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price) if buy_price else False

    def generate_exit_signal(self, buy_price, predicted_high):
        return self.exit_signal(self.current_price, buy_price, predicted_high) if buy_price else False