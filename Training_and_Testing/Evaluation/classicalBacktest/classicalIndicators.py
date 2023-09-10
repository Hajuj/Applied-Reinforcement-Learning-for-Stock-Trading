import numpy as np
import pandas as pd



import matplotlib.pyplot as plt
import pandas_ta as pta
import warnings
warnings.filterwarnings("ignore")

def directory(hourly, sentiment, smoothed, normalized):
    if hourly:
        if smoothed:
            path = 'Evaluation/data_testing/hourly/hourly_smoothed_testing_data.csv'
            if sentiment:
                if normalized:
                    save = 'results_data/comparison'
                else:
                    save = 'results_data/hourly/smoothed/Sentiment'
            else:
                save = 'results_data/hourly/smoothed/noSentiment'
        else:
            path = 'Evaluation/data_testing/hourly/hourly_testing_data.csv'
            if sentiment:
                save = 'results_data/hourly/notSmoothed/Sentiment'
            else:
                save = 'results_data/hourly/notSmoothed/noSentiment'

    else:
        if smoothed:
            path = 'Evaluation/data_testing/daily/daily_smoothed_testing_data.csv'
            if sentiment:
                save = 'results_data/daily/Smoothed/Sentiment'
            else:
                save = 'results_data/daily/Smoothed/noSentiment'
        else:
            path = 'Evaluation/data_testing/daily/daily_testing_data.csv'
            if sentiment:
                save = 'results_data/daily/notSmoothed/Sentiment'
            else:
                save = 'results_data/daily/notSmoothed/noSentiment'
    return path, save

def main():
    hourly = True
    smoothed = True
    sentiment = True
    normalized = True


    #path, save = directory(hourly, sentiment, smoothed, normalized)
    path = '/Users/lolo/PycharmProjects/automatic-stock-trading-applied-reinforcement-learning-sommer-term-2023/' \
           'Training_and_Testing/Evaluation/data_testing/hourly/hourly_smoothed_testing_data.csv'
    train = pd.read_csv(path)
    capital = 1000000

    arr = []
    for tic in train["tic"].unique():
        df = train.loc[train['tic'] == tic]

        df.reset_index(drop=True)
        df.set_index('date', inplace=True)

        df["sma"] = df["close"].rolling(window=20).mean()
        df["std"] = df["close"].rolling(window=20).std()
        df["upper_band"] = df["sma"] + (df["std"] * 2)
        df["lower_band"] = df["sma"] - (df["std"] * 2)

        df["Position"] = None
        df["Position"][20:] = np.where(df["close"][20:] < df["lower_band"][20:], 1, 0)
        df["Position"][20:] = np.where(df["close"][20:] > df["upper_band"][20:], -1, df["Position"][20:])
        df["Bollinger_Returns"] = 1 + (df["close"].pct_change() * df["Position"].shift(1))

        x = df["Bollinger_Returns"].cumprod().to_numpy()
        arr.append(x)
    bollinger = np.mean(arr, axis=0)

    arr = []
    ch = []
    for tic in train["tic"].unique():

        df = train.loc[train['tic'] == tic]

        df.reset_index(drop=True)
        df.set_index('date', inplace=True)

        df["ema12"] = df["close"].ewm(span=12).mean()
        df["ema26"] = df["close"].ewm(span=26).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        df["regime"] = pd.NA
        for i in range(0, len(df) - 1):
            if df["macd_histogram"][i] >= 0:
                df["regime"][i + 1] = 1
            else:
                df["regime"][i + 1] = 0

        df["change"] = df["close"].pct_change() + 1
        df["MACD_returns"] = 1.0
        for i in range(1, len(df)):
            if df["regime"][i] == 1:
                df["MACD_returns"][i] = df["change"][i]

        x = df["MACD_returns"].cumprod().to_numpy()
        c = df["change"].cumprod().to_numpy()
        arr.append(x)
        ch.append(c)
    macd = np.mean(arr, axis=0)
    buyandhold = np.mean(ch, axis=0)

    arr = []
    for tic in train["tic"].unique():

        df = train.loc[train['tic'] == tic]

        df.reset_index(drop=True)
        df.set_index('date', inplace=True)

        df["rsi"] = pta.rsi(df["close"], 2)
        df["rule"] = 0
        z = 1

        for i in range(0, len(df) - 1):
            if df["rsi"][i] <= 10 and z == 1:
                df["rule"][i + 1] = 1
                z = 0
            elif df["rsi"][i] >= 60 and z == 0:
                df["rule"][i] = -1
                z = 1
        for i in range(1, len(df)):
            if df["rule"][i - 1] == 1 and df["rule"][i] == 0:
                df["rule"][i] = 1
        df["change"] = df["close"].pct_change() + 1
        df["RSI_Returns"] = 1.0
        for i in range(0, len(df)):
            if df["rule"][i] != 0:
                df["RSI_Returns"][i] = df["change"][i]

        x = df["RSI_Returns"].cumprod().to_numpy()
        arr.append(x)
    rsi = np.mean(arr, axis=0)

    #plt.plot(buyandhold * capital, label="Buy & Hold")
    #plt.plot(macd * capital, label="MACD")
    #plt.plot(rsi * capital, label="RSI")
    #plt.plot(bollinger * capital, label="Bollinger Bands")

    buyandhold_df = buyandhold*capital
    macd_df = macd * capital
    rsi_df = rsi*capital
    bollinger_df = bollinger*capital

    # ToDo: In Dataframe bringen, s.d. es das gleiche Format hat wie die RL Agenten






main()