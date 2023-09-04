import pandas as pd
from stable_baselines3.common.logger import configure

import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from Envs.env_stocktrading import StockTradingEnv

check_and_make_directories([TRAINED_MODEL_DIR])

def preparations():

    train = pd.read_csv('training_data_complete.csv')
    SENTIMENT = ['stocktwitsPosts','stocktwitsLikes','stocktwitsImpressions','stocktwitsSentiment', 'random']