import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MultipleLocator


def main():
    comparison = 'normalized'
    agent = 'sac'
    result1 = pd.read_csv('sac_combined_normalized.csv')
    result2 = pd.read_csv('sac_combined_notNormalized.csv')
    results = pd.concat([result1, result2], ignore_index=True)

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
    plt.savefig(f"Agent_performance_{comparison}_{agent}.pdf")

main()
