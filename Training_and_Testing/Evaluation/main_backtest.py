import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from Envs.env_stocktrading import StockTradingEnv



def prep(train):
    # Sort the DataFrame by the 'date' column
    train = train.sort_values(by='date')

    # Create a mapping of unique dates to their corresponding index
    date_mapping = {date: idx for idx, date in enumerate(train['date'].unique())}

    # Map the 'date' column to the date_mapping to get the new index
    train['index'] = train['date'].map(date_mapping)

    # Set the new index for the DataFrame
    train.set_index('index', inplace=True, drop=True)
    train = train.fillna(0)
    return train
def directory(hourly, sentiment, smoothed, normalized):
    if hourly:
        if smoothed:
            path = 'data_testing/hourly/hourly_smoothed_testing_data.csv'
            turbulence_threshold = 1000
            if sentiment:
                SENTIMENT = ['stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment',
                             'random']
                if normalized:
                    model_path = 'model/hourly/normalized/smoothedSentiment'
                    save_path = "results_data/hourly/normalized/smoothedSentiment"
                else:
                    model_path = 'model/hourly/notNormalized/smoothedSentiment'
                    save_path = 'results_data/hourly/notNormalized/smoothedSentiment'

            else:
                SENTIMENT = ['random']
                if normalized:
                    model_path = 'model/hourly/normalized/smoothedNoSentiment'
                    save_path = 'results_data/hourly/normalized/smoothedNoSentiment'
                else:
                    model_path = 'model/hourly/notNormalized/smoothedNoSentiment'
                    save_path = 'results_data/hourly/notNormalized/smoothedNoSentiment'

        else:
            path = 'data_testing/hourly/hourly_testing_data.csv'
            turbulence_threshold = 700
            if sentiment:
                SENTIMENT = ['stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment',
                             'random']
                if normalized:
                    model_path = 'model/hourly/normalized/Sentiment'
                    save_path = 'results_data/hourly/normalized/Sentiment'
                else:
                    model_path = 'model/hourly/notNormalized/Sentiment'
                    save_path = 'results_data/hourly/notNormalized/Sentiment'

            else:
                SENTIMENT = ['random']
                if normalized:
                    model_path = 'model/hourly/normalized/noSentiment'
                    save_path = 'results_data/hourly/normalized/noSentiment'
                else:
                    model_path = 'model/hourly/notNormalized/noSentiment'
                    save_path = 'results_data/hourly/notNormalized/noSentiment'

    else:
        if smoothed:
            path = 'data_testing/daily/daily_smoothed_testing_data.csv'
            turbulence_threshold = 90
            if sentiment:
                SENTIMENT = ['stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment',
                             'random']
                if normalized:
                    model_path = 'model/daily/normalized/smoothedSentiment'
                    save_path = 'results_data/daily/normalized/smoothedSentiment'
                else:
                    model_path = 'model/daily/notNormalized/smoothedSentiment'
                    save_path = 'results_data/daily/notNormalized/smoothedSentiment'

            else:
                SENTIMENT = ['random']
                if normalized:
                    model_path = 'model/daily/normalized/smoothedNoSentiment'
                    save_path = 'results_data/daily/normalized/smoothedNoSentiment'
                else:
                    model_path = 'model/daily/notNormalized/smoothedNoSentiment'
                    save_path = 'results_data/daily/notNormalized/smoothedNoSentiment'
        else:
            path = 'data_testing/daily/daily_testing_data.csv'
            turbulence_threshold = 80
            if sentiment:
                SENTIMENT = ['stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment',
                             'random']
                if normalized:
                    model_path = 'model/daily/normalized/Sentiment'
                    save_path = 'results_data/daily/normalized/Sentiment'
                else:
                    model_path = 'model/daily/notNormalized/Sentiment'
                    save_path = 'results_data/daily/notNormalized/Sentiment'

            else:
                SENTIMENT = ['random']
                if normalized:
                    model_path = 'model/daily/normalized/noSentiment'
                    save_path = 'results_data/daily/normalized/noSentiment'
                else:
                    model_path = 'model/daily/notNormalized/noSentiment'
                    save_path = 'results_data/daily/notNormalized/noSentiment'
    return path, model_path, SENTIMENT, turbulence_threshold, save_path

def normalize_data(train):
    interm = train['close']

    df_num = train.select_dtypes(include='number')
    df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min()) * 200 + 20

    train[df_norm.columns] = df_norm
    train['close'] = interm
    return train

def main():
    # change the parameters according to the model to be tested
    hourly = True
    sentiment = False
    threshold_flag = True
    smoothed = False
    normalized = True
    transaction_cost = 0.01

    if_using_a2c = True
    if_using_ddpg = False
    if_using_ppo = True
    if_using_td3 = False
    if_using_sac = True




    path, model_path, SENTIMENT, turbulence_threshold, save = directory(hourly, sentiment, smoothed, normalized)

    trade = pd.read_csv(path)
    trade = prep(trade)

    if normalized:
        trade = normalize_data(trade)

    if 'vix' not in INDICATORS:
        INDICATORS.append('vix')


    if threshold_flag:
        model_path = model_path + '/threshold'
        save = save + '/threshold'
    else:
        model_path = model_path + '/noThreshold'
        save = save + '/noThreshold'

    if transaction_cost == 0.001:
        model_path = model_path + '/lowCost'
        save = save + '/lowCost'

    print(f'model path: {model_path}, save path: {save}')

    for seed in range(0, 3):

        model_name = f"seed{seed}"


        trained_a2c = A2C.load(model_path + f"/a2c/{model_name}") if if_using_a2c else None
        trained_ddpg = DDPG.load(model_path + f"/ddpg/{model_name}") if if_using_ddpg else None
        trained_ppo = PPO.load(model_path + f"/ppo/{model_name}") if if_using_ppo else None
        trained_td3 = TD3.load(model_path + f"/td3/{model_name}") if if_using_td3 else None
        trained_sac = SAC.load(model_path + f"/sac/{model_name}") if if_using_sac else None

        stock_dimension = len(trade.tic.unique())
        state_space = 1 + 4 * stock_dimension + len(INDICATORS) * stock_dimension + len(SENTIMENT) * stock_dimension
        print(f'State Space: {state_space}, Stock Dimension: {stock_dimension}')

        buy_cost_list = sell_cost_list = [transaction_cost] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "sentiment_list": SENTIMENT,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "seed": seed,
            "hourly": hourly,
        }
        if threshold_flag:
            e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=turbulence_threshold, **env_kwargs)
        else:
            e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
        print(INDICATORS)
        print(SENTIMENT)
        df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
            model=trained_a2c,
            environment=e_trade_gym) if if_using_a2c else (None, None)

        df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
            model=trained_ddpg,
            environment=e_trade_gym) if if_using_ddpg else (None, None)

        df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
            model=trained_ppo,
            environment=e_trade_gym) if if_using_ppo else (None, None)

        df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
            model=trained_td3,
            environment=e_trade_gym) if if_using_td3 else (None, None)

        df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
            model=trained_sac,
            environment=e_trade_gym) if if_using_sac else (None, None)

        df_result_a2c = (
            df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
            if if_using_a2c
            else None
        )
        df_result_ddpg = (
            df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
            if if_using_ddpg
            else None
        )
        df_result_ppo = (
            df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
            if if_using_ppo
            else None
        )
        df_result_td3 = (
            df_account_value_td3.set_index(df_account_value_td3.columns[0])
            if if_using_td3
            else None
        )
        df_result_sac = (
            df_account_value_sac.set_index(df_account_value_sac.columns[0])
            if if_using_sac
            else None
        )

        result = pd.DataFrame(
            {
                "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
                "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
                "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
                "td3": df_result_td3["account_value"] if if_using_td3 else None,
                "sac": df_result_sac["account_value"] if if_using_sac else None,
            }
        )



        result.to_csv(f"{save}/result{seed}.csv")
        #result.to_csv(f"{save}/result{seed}_norm{normalized}.csv")






main()