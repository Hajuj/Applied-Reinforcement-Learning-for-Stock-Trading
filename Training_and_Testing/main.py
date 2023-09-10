import os

import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from Envs.env_stocktrading import StockTradingEnv
def directory(hourly, sentiment, smoothed):
    if hourly:
        if smoothed:
            path = r"C:\Users\potap\OneDrive\Desktop\hourly\hourly_smoothed_training_data.csv"#'Data/Hourly/hourly_smoothed_training_data.csv'
            turbulence_threshold = 1000
            if sentiment:
                model_path = 'Evaluation/model/hourly/smoothedSentimentNorm'
                log_path = 'Logger/Hourly/smoothedSentimentNorm'
                SENTIMENT = ['stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment',
                             'random']
            else:
                model_path = 'Evaluation/model/hourly/smoothedNoSentiment'
                log_path = 'Logger/Hourly/smoothedNoSentiment'
                SENTIMENT = ['random']
        else:
            path = 'Data/Hourly/hourly_training_data.csv'
            turbulence_threshold = 700
            if sentiment:
                model_path = 'Evaluation/model/hourly/Sentiment'
                log_path = 'Logger/Hourly/Sentiment'
                SENTIMENT = ['stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment',
                             'random']
            else:
                model_path = 'Evaluation/model/hourly/noSentiment'
                log_path = 'Logger/Hourly/noSentiment'
                SENTIMENT = ['random']

    else:
        if smoothed:
            path = 'Data/Daily/daily_smoothed_training_data.csv'
            turbulence_threshold = 90
            if sentiment:
                model_path = 'Evaluation/model/daily/smoothedSentiment'
                log_path = 'Logger/Daily/smoothedSentiment'
                SENTIMENT = ['stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment',
                             'random']
            else:
                model_path = 'Evaluation/model/daily/smoothedNoSentiment'
                log_path = 'Logger/Daily/smoothedNoSentiment'
                SENTIMENT = ['random']
        else:
            path = 'Data/Daily/daily_training_data.csv'
            turbulence_threshold = 80
            if sentiment:
                model_path = 'Evaluation/model/daily/Sentiment'
                log_path = 'Logger/Daily/Sentiment'
                SENTIMENT = ['stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment',
                             'random']
            else:
                model_path = 'Evaluation/model/daily/noSentiment'
                log_path = 'Logger/Daily/noSentimentSentiment'
                SENTIMENT = ['random']
    return path, model_path, log_path,SENTIMENT, turbulence_threshold
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
    train = train.sort_values(['date', 'tic'], ascending=[True, True])
    return train

def normalize_data(train):
    interm = train['close']
    df_num = train.select_dtypes(include='number')
    df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min()) * 200 + 20

    train[df_norm.columns] = df_norm
    train['close'] = interm
    return train


def main():
    # Change these parameters
    # Set the corresponding values to 'True' for the algorithms that you want to use
    if_using_a2c = True
    if_using_ddpg = False
    if_using_ppo = False
    if_using_td3 = False
    if_using_sac = True

    hourly = True
    sentiment = True
    smoothed = True

    for seed in range(3):


        # Do not change (except for agent parameters)
        path, model_path, log_path, SENTIMENT, turbulence_threshold = directory(hourly, sentiment, smoothed)
        check_and_make_directories([model_path])

        train = pd.read_csv(path)
        train = prep(train)
        train = normalize_data(train)
        if 'vix' not in INDICATORS:
            INDICATORS.append('vix')
        print(INDICATORS)

        stock_dimension = len(train.tic.unique())
        state_space = 1 + 4 * stock_dimension + len(INDICATORS) * stock_dimension + len(SENTIMENT) * stock_dimension
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        buy_cost_list = sell_cost_list = [0.01] * stock_dimension
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
        e_train_gym = StockTradingEnv(df=train, turbulence_threshold=turbulence_threshold, **env_kwargs)

        env_train, _ = e_train_gym.get_sb_env()
        print(type(env_train))

        agent = DRLAgent(env=env_train)

        # A2C
        agent = DRLAgent(env=env_train)
        model_a2c = agent.get_model("a2c")

        if if_using_a2c:
            # set up logger
            tmp_path = log_path + f'/a2c/seed{seed}'
            new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_a2c.set_logger(new_logger_a2c)

        trained_a2c = agent.train_model(model=model_a2c,
                                        tb_log_name='a2c',
                                        total_timesteps=50000) if if_using_a2c else None

        trained_a2c.save(model_path + f"/a2c/seed{seed}") if if_using_a2c else None

        # DDPG

        agent = DRLAgent(env=env_train)
        model_ddpg = agent.get_model("ddpg")

        if if_using_ddpg:
            # set up logger
            tmp_path = log_path + f'/ddpg/seed{seed}'
            new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_ddpg.set_logger(new_logger_ddpg)

        trained_ddpg = agent.train_model(model=model_ddpg,
                                         tb_log_name='ddpg',
                                         total_timesteps=200000) if if_using_ddpg else None

        trained_ddpg.save(model_path + f"/ddpg/seed{seed}") if if_using_ddpg else None


        # PPO
        agent = DRLAgent(env=env_train)
        PPO_PARAMS = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        }
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

        if if_using_ppo:
            # set up logger
            tmp_path = log_path + f'/ppo/seed{seed}'
            new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_ppo.set_logger(new_logger_ppo)

        trained_ppo = agent.train_model(model=model_ppo,
                                        tb_log_name='ppo',
                                        total_timesteps=50000) if if_using_ppo else None

        trained_ppo.save(model_path + f"/ppo/seed{seed}") if if_using_ppo else None

        # TD3
        agent = DRLAgent(env=env_train)
        TD3_PARAMS = {"batch_size": 100,
                      "buffer_size": 1000000,
                      "learning_rate": 0.001}

        model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

        if if_using_td3:
            # set up logger
            tmp_path = log_path + f'/td3/seed{seed}'
            new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_td3.set_logger(new_logger_td3)

        trained_td3 = agent.train_model(model=model_td3,
                                        tb_log_name='td3',
                                        total_timesteps=200000) if if_using_td3 else None

        trained_td3.save(model_path + f"/td3/seed{seed}") if if_using_td3 else None


        # SAC

        agent = DRLAgent(env=env_train)
        SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 100000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }

        model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

        if if_using_sac:
            # set up logger
            tmp_path = log_path + f'/sac/seed{seed}'
            new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_sac.set_logger(new_logger_sac)

        trained_sac = agent.train_model(model=model_sac,
                                        tb_log_name='sac',
                                        total_timesteps=50000) if if_using_sac else None

        trained_sac.save(model_path + f"/sac/seed{seed}") if if_using_sac else None






if __name__ == "__main__":
    main()