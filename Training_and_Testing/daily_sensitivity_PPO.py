import torch
import numpy as np
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from Envs.env_stocktrading import StockTradingEnv
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Suppress specific warning
import onnx

# Your existing code...

# Function to ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
# If you are not using the data generated from part 1 of this tutorial, make sure
# it has the columns and index in the form that could be make into the environment.
# Then you can comment and skip the following two lines.
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
    flagsentiment = 'Sentiment/'
    train = pd.read_csv(r"Data/Daily/daily_training_data.csv")
    train = prep(train)
    train = normalize_data(train)

    INDICATORS = ['macd',
                  'boll_ub',
                  'boll_lb',
                  'rsi_30',
                  'cci_30',
                  'dx_30',
                  'close_30_sma',
                  'close_60_sma',
                  'vix']

    SENTIMENT = ['random']#, 'stocktwitsPosts', 'stocktwitsLikes', 'stocktwitsImpressions', 'stocktwitsSentiment']
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 4 * stock_dimension + len(INDICATORS) * stock_dimension + len(SENTIMENT) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    turbulence_threshold = 1000


    all_seeds_results = []
    for seed in range(3):
        for agent_name in ['ppo']:
            buy_cost_list = sell_cost_list = [0.01] * stock_dimension
            num_stock_shares = [0] * stock_dimension

            env_kwargs = {
                "hmax": 100,
                "initial_amount": 50,
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
                "hourly": False
            }
            print(SENTIMENT)
            e_train_gym = StockTradingEnv(df=train, turbulence_threshold=turbulence_threshold, risk_indicator_col='vix',
                                          **env_kwargs)
            env_train, _ = e_train_gym.get_sb_env()

            saved_model_path = r"Evaluation/model/daily/normalized/NoSentiment/threshold/" + agent_name + '/seed' + str(
                seed) + '.zip'
            agent = DRLAgent(env=env_train)
            model = agent.get_model(agent_name)
            trained_a2c = agent.train_model(model=model,
                                            tb_log_name=agent_name,
                                            total_timesteps=50)
            loaded_agent = trained_a2c.load(saved_model_path)
            # Make sure your model is in evaluation mode
            loaded_agent.policy.eval()

            # generate dummy input
            observation_size = loaded_agent.observation_space.shape[0]
            print(observation_size)
            dummy_input = torch.randn(1, observation_size)

            # Export the model to ONNX format
            torch.onnx.export(loaded_agent.policy, dummy_input,
                              "Policy/policy_" + agent_name + '_seed_' + str(seed) + '.onnx')
            print('Alive1')
            onnx_model = onnx.load("Policy/policy_" + agent_name + '_seed_' + str(seed) + '.onnx')  # load onnx model
            print('Alive2')
            tf_rep = prepare(onnx_model)  # prepare tf representation
            print('Alive3')
            tf_rep.export_graph("Policy/tensorflowGraph_" + agent_name + '_seed_' + str(seed) + '.pb')  # export the model

            model_filename = "Policy/tensorflowGraph_" + agent_name + '_seed_' + str(seed) + '.pb'
            # Load the TensorFlow model
            model_loaded = tf.saved_model.load(model_filename)
            print('Alive4')
            model_loaded = model_loaded.signatures['serving_default']
            print('Alive5')

            num_stock_shares = 30 * [0]
            test = train.sort_values(['date', 'tic'],
                                     ascending=[True, True])

            all_days_grads = []
            for day in range(len((train.date.unique()))):  # len((train.date.unique()))
                data = test.loc[day, :]
                # print(day)

                print(SENTIMENT)
                state = (
                        [50]
                        + data.close.values.tolist()
                        + num_stock_shares
                        + sum(
                    (
                        data[tech].values.tolist()
                        for tech in INDICATORS
                    ),
                    [],
                )
                        + sum(
                    (
                        data[sent].values.tolist()
                        for sent in SENTIMENT
                    ),
                    [],
                )
                        + data.change.values.tolist()
                        + data.volume.values.tolist())
                print(len(state))
                # print(state)
                xs = tf.convert_to_tensor([state])
                with tf.GradientTape() as tape:
                    tape.watch(xs)
                    pred = model_loaded(xs)
                    grads = tape.gradient(pred, xs)
                grads = grads.numpy()
                abs_grads_sum = np.sum(np.abs(grads), axis=0)
                all_days_grads.append(abs_grads_sum)

            plot_overname = agent_name + ' Seed ' + str(seed)

            best_days = pd.DataFrame(all_days_grads)
            best_days.set_index(train.date.unique(), inplace=True)
            best_days['date'] = best_days.index
            best_days['date'] = pd.to_datetime(best_days['date']).dt.date


            df = best_days.copy()
            # Ensure the 'date' column is in datetime format
            df['date'] = pd.to_datetime(df['date'])

            # Drop the 'date' column and sum across rows
            df['total'] = df.drop('date', axis=1).sum(axis=1)

            # Group by 'date' and sum the 'total' column
            result = df.groupby('date')['total'].sum()

            # Reset the index to get day number
            result = result.reset_index()
            # result['day_number'] = (result['date'] - result['date'].min()).dt.days + 1
            # result.set_index('day_number', inplace=True)
            result.drop('date', axis=1, inplace=True)
            unique_dates = df['date'].unique()
            date_mapping = pd.DataFrame({
                'date': unique_dates
            }, index=range(0, len(unique_dates)))

            # Join the date to the result DataFrame based on index
            result = result.join(date_mapping)

            # Set the date as the index for better plotting
            result.set_index('date', inplace=True)
            all_seeds_results.append(result['total'])

        # Calculate mean and standard deviation after processing all seeds
        mean_results = pd.concat(all_seeds_results, axis=1).mean(axis=1)
        std_results = pd.concat(all_seeds_results, axis=1).std(axis=1)

    # Plotting
    plt.figure(figsize=(12, 6))
    mean_results.plot(linewidth=2, label='Mean', color='blue')
    plt.fill_between(mean_results.index, mean_results - std_results, mean_results + std_results, color='blue',
                     alpha=0.2, label='Standard Deviation')
    plt.title('Mean and Standard Deviation of Gradients Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sum of Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir1 = 'Sensitivity/Day_importance/daily/' + flagsentiment + 'mean_std'
    ensure_dir(output_dir1)
    plt.savefig(output_dir1 + '/mean_std.pdf')



if __name__ == "__main__":
    main()