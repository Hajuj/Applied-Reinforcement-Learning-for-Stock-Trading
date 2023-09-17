import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def directory(hourly, sentiment, smoothed, normalized, threshold, cost):
    if hourly:
        if normalized:
            if smoothed:
                if sentiment:
                    if threshold:
                        if cost:
                            path = 'hourly/normalized/smoothedSentiment/threshold/lowCost'
                        else:
                            path = 'hourly/normalized/smoothedSentiment/threshold'
                    else:
                        if cost:
                            path = 'hourly/normalized/smoothedSentiment/noThreshold/lowCost'
                        else:
                            path = 'hourly/normalized/smoothedSentiment/noThreshold'
                else:
                    if threshold:
                        if cost:
                            path = 'hourly/normalized/smoothedNoSentiment/threshold/lowCost'
                        else:
                            path = 'hourly/normalized/smoothedNoSentiment/threshold'
                    else:
                        if cost:
                            path = 'hourly/normalized/smoothedNoSentiment/noThreshold/lowCost'
                        else:
                            path = 'hourly/normalized/smoothedNoSentiment/noThreshold'
            else:
                if sentiment:
                    if threshold:
                        if cost:
                            path = 'hourly/normalized/Sentiment/threshold/lowCost'
                        else:
                            path = 'hourly/normalized/Sentiment/threshold'
                    else:
                        if cost:
                            path = 'hourly/normalized/Sentiment/noThreshold/lowCost'
                        else:
                            path = 'hourly/normalized/Sentiment/noThreshold'
                else:
                    if threshold:
                        if cost:
                            path = 'hourly/normalized/noSentiment/threshold/lowCost'
                        else:
                            path = 'hourly/normalized/noSentiment/threshold'
                    else:
                        if cost:
                            path = 'hourly/normalized/noSentiment/noThreshold/lowCost'
                        else:
                            path = 'hourly/normalized/noSentiment/noThreshold'
        else:
            if smoothed:
                if sentiment:
                    if threshold:
                        if cost:
                            path = 'hourly/notNormalized/smoothedSentiment/threshold/lowCost'
                        else:
                            path = 'hourly/notNormalized/smoothedSentiment/threshold'
                    else:
                        if cost:
                            path = 'hourly/notNormalized/smoothedSentiment/noThreshold/lowCost'
                        else:
                            path = 'hourly/notNormalized/smoothedSentiment/noThreshold'
                else:
                    if threshold:
                        if cost:
                            path = 'hourly/notNormalized/smoothedNoSentiment/threshold/lowCost'
                        else:
                            path = 'hourly/notNormalized/smoothedNoSentiment/threshold'
                    else:
                        if cost:
                            path = 'hourly/notNormalized/smoothedNoSentiment/noThreshold/lowCost'
                        else:
                            path = 'hourly/notNormalized/smoothedNoSentiment/noThreshold'
            else:
                if sentiment:
                    if threshold:
                        if cost:
                            path = 'hourly/notNormalized/Sentiment/threshold/lowCost'
                        else:
                            path = 'hourly/notNormalized/Sentiment/threshold'
                    else:
                        if cost:
                            path = 'hourly/notNormalized/Sentiment/noThreshold/lowCost'
                        else:
                            path = 'hourly/notNormalized/Sentiment/noThreshold'
                else:
                    if threshold:
                        if cost:
                            path = 'hourly/notNormalized/noSentiment/threshold/lowCost'
                        else:
                            path = 'hourly/notNormalized/noSentiment/threshold'
                    else:
                        if cost:
                            path = 'hourly/notNormalized/noSentiment/noThreshold/lowCost'
                        else:
                            path = 'hourly/notNormalized/noSentiment/noThreshold'
    else:
        if normalized:
            if smoothed:
                if sentiment:
                    if threshold:
                        if cost:
                            path = 'daily/normalized/smoothedSentiment/threshold/lowCost'
                        else:
                            path = 'daily/normalized/smoothedSentiment/threshold'
                    else:
                        if cost:
                            path = 'daily/normalized/smoothedSentiment/noThreshold/lowCost'
                        else:
                            path = 'daily/normalized/smoothedSentiment/noThreshold'
                else:
                    if threshold:
                        if cost:
                            path = 'daily/normalized/smoothedNoSentiment/threshold/lowCost'
                        else:
                            path = 'daily/normalized/smoothedNoSentiment/threshold'
                    else:
                        if cost:
                            path = 'daily/normalized/smoothedNoSentiment/noThreshold/lowCost'
                        else:
                            path = 'daily/normalized/smoothedNoSentiment/noThreshold'
            else:
                if sentiment:
                    if threshold:
                        if cost:
                            path = 'daily/normalized/Sentiment/threshold/lowCost'
                        else:
                            path = 'daily/normalized/Sentiment/threshold'
                    else:
                        if cost:
                            path = 'daily/normalized/Sentiment/noThreshold/lowCost'
                        else:
                            path = 'daily/normalized/Sentiment/noThreshold'
                else:
                    if threshold:
                        if cost:
                            path = 'daily/normalized/noSentiment/threshold/lowCost'
                        else:
                            path = 'daily/normalized/noSentiment/threshold'
                    else:
                        if cost:
                            path = 'daily/normalized/noSentiment/noThreshold/lowCost'
                        else:
                            path = 'daily/normalized/noSentiment/noThreshold'
        else:
            if smoothed:
                if sentiment:
                    if threshold:
                        if cost:
                            path = 'daily/notNormalized/smoothedSentiment/threshold/lowCost'
                        else:
                            path = 'daily/notNormalized/smoothedSentiment/threshold'
                    else:
                        if cost:
                            path = 'daily/notNormalized/smoothedSentiment/noThreshold/lowCost'
                        else:
                            path = 'daily/notNormalized/smoothedSentiment/noThreshold'
                else:
                    if threshold:
                        if cost:
                            path = 'daily/notNormalized/smoothedNoSentiment/threshold/lowCost'
                        else:
                            path = 'daily/notNormalized/smoothedNoSentiment/threshold'
                    else:
                        if cost:
                            path = 'daily/notNormalized/smoothedNoSentiment/noThreshold/lowCost'
                        else:
                            path = 'daily/notNormalized/smoothedNoSentiment/noThreshold'
            else:
                if sentiment:
                    if threshold:
                        if cost:
                            path = 'daily/notNormalized/Sentiment/threshold/lowCost'
                        else:
                            path = 'daily/notNormalized/Sentiment/threshold'
                    else:
                        if cost:
                            path = 'daily/notNormalized/Sentiment/noThreshold/lowCost'
                        else:
                            path = 'daily/notNormalized/Sentiment/noThreshold'
                else:
                    if threshold:
                        if cost:
                            path = 'daily/notNormalized/noSentiment/threshold/lowCost'
                        else:
                            path = 'daily/notNormalized/noSentiment/threshold'
                    else:
                        if cost:
                            path = 'daily/notNormalized/noSentiment/noThreshold/lowCost'
                        else:
                            path = 'daily/notNormalized/noSentiment/noThreshold'

    return path


def main():
    # Params
    hourly = True
    sentiment = False
    smoothed = True
    normalized = True
    threshold = True
    low_cost = False

    comp = 'normalized'
    # comp = 'notNormalized'
    # comp = 'smoothed'
    # comp = 'notSmoothed'
    # comp = 'threshold'
    # comp = 'noThreshold'
    # comp = 'lowCost'
    # comp = 'highCost'

    agent = 'sac'

    # agents
    if_a2c = False
    if_ddpg = False
    if_ppo = False
    if_td3 = False
    if_sac = True

    path = directory(hourly, sentiment, smoothed, normalized, threshold, low_cost)

    df_RL = pd.concat(
        map(
            pd.read_csv,
            [
                f'{path}/result0.csv',
                f'{path}/result1.csv',
                f'{path}/result2.csv',

            ],
        ),
        ignore_index=True,
    )
    dataframes = []

    if if_a2c:
        a2c = df_RL.copy()
        a2c.drop(['ddpg', 'ppo', 'sac', 'td3'], axis=1, inplace=True)
        a2c['agent'] = f'a2c - {comp}'
        a2c.rename(columns={'a2c': 'account value'}, inplace=True)
        a2c.to_csv(f"a2c_combined_{comp}.csv")
        dataframes.append(a2c)
    if if_ddpg:
        ddpg = df_RL.copy()
        ddpg.drop(['a2c', 'ppo', 'sac', 'td3'], axis=1, inplace=True)
        ddpg['agent'] = f'ddpg - {comp}'
        ddpg.rename(columns={'ddpg': 'account value'}, inplace=True)
        ddpg.to_csv(f"ddpg_combined_{comp}.csv")
        dataframes.append(ddpg)
    if if_ppo:
        ppo = df_RL.copy()
        ppo.drop(['ddpg', 'a2c', 'sac', 'td3'], axis=1, inplace=True)
        ppo['agent'] = f'ppo - {comp}'
        ppo.rename(columns={'ppo': 'account value'}, inplace=True)
        ppo.to_csv(f"ppo_combined_{comp}.csv")
        dataframes.append(ppo)
    if if_td3:
        td3 = df_RL.copy()
        td3.drop(['ddpg', 'ppo', 'sac', 'a2c'], axis=1, inplace=True)
        td3['agent'] = f'td3 - {comp}'
        td3.rename(columns={'td3': 'account value'}, inplace=True)
        td3.to_csv(f"td3_combined_{comp}.csv")
        dataframes.append(td3)
    if if_sac:
        sac = df_RL.copy()
        sac.drop(['ddpg', 'ppo', 'a2c', 'td3'], axis=1, inplace=True)
        sac['agent'] = f'sac - {comp}'
        sac.rename(columns={'sac': 'account value'}, inplace=True)
        sac.to_csv(f"sac_combined_{comp}.csv")
        dataframes.append(sac)

    # Plot
    results = pd.concat(
        dataframes,
        ignore_index=True
    )
    # results.to_csv(f"data_concat_{comp}_{agent}.csv")

    # global
    plt.figure(figsize=(15, 6))
    sns.set_style("white")
    plt.grid(axis='y')

    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=False)

    # dedicated plot
    sns.set(font_scale=1.4)
    fig2 = sns.lineplot(
        data=results,
        x="date",
        y="account value",
        hue="agent",
        palette="colorblind")
    fig2.set(xticklabels=[])
    fig2.tick_params(bottom=False)
    fig2.set(xlabel="Date")
    fig2.set(ylabel="Account Value")

    # ticks
    data = pd.to_datetime(results["date"]).copy().dt.strftime('%d-%b-%Y')

    # param to control spacing of labels
    n_param = 50

    for i in range(len(data)):
        if (i + 1) % n_param:
            data[i] = ""

    plt.xticks(results.date, data.values, rotation=45)

    # dedicated plot
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("Agent_performance.pdf")

    # results = pd.read_csv(f"data_concat_{comp}_{agent}.csv")

main()
