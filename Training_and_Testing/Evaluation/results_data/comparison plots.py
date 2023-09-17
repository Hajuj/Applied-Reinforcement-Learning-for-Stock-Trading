import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    comparison = 'hourlySentimentTechnical'
    agent = 'a2c_ddpg_buy_hold'
    result1 = pd.read_csv('a2c_combined_hourlySentiment.csv')
    # result2 = pd.read_csv('a2c_combined_hourlyNoSentiment.csv')
    # result3 = pd.read_csv('ppo_combined_hourlySentiment.csv')
    # result4 = pd.read_csv('ppo_combined_hourlyNoSentiment.csv')
    # result5 = pd.read_csv('sac_combined_hourlySentiment.csv')
    # result6 = pd.read_csv('sac_combined_hourlyNoSentiment.csv')
    # result7 = pd.read_csv('td3_combined_hourlySentiment.csv')
    # result8 = pd.read_csv('td3_combined_hourlyNoSentiment.csv')
    result9 = pd.read_csv('ddpg_combined_hourlySentiment.csv')
    # result10 = pd.read_csv('ddpg_combined_hourlyNoSentiment.csv')
    result11 = pd.read_csv('technical_indicators_hourly.csv')
    result_ind = result11.loc[result11["agent"] == "Buy_and_hold"]
    results = pd.concat([result1, result9, result_ind], ignore_index=True)

    # Ermitteln Sie die Anzahl der Datenpunkte
    num_data_points = len(results)

    # Wählen Sie aus, wie viele Ticks Sie anzeigen möchten (z.B. jeden fünften)
    num_ticks = 15

    # Berechnen Sie die x-Koordinaten der Ticks
    tick_positions = np.arange(0, num_data_points, num_data_points // num_ticks)

    # Plot
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
    # plt.savefig(f"Compare_agent_performance_{comparison}_{agent}.pdf")
    plt.savefig(f"Compare_agent_performance_{comparison}_{agent}.pdf")

main()
