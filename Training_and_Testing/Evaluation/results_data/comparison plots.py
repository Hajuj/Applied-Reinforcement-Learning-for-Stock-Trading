import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    comparison = 'normalized'
    agent = 'sac'
    result1 = pd.read_csv('data_concat_normalized_sac.csv')
    result2 = pd.read_csv('data_concat_notNormalized_sac.csv')
    results = pd.concat([result1, result2], ignore_index=True)

    # Ermitteln Sie die Anzahl der Datenpunkte
    num_data_points = len(results)

    # Wählen Sie aus, wie viele Ticks Sie anzeigen möchten (z.B. jeden fünften)
    num_ticks = 15

    # Berechnen Sie die x-Koordinaten der Ticks
    tick_positions = np.arange(0, num_data_points, num_data_points // num_ticks)



    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 6))
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)
    fig = sns.lineplot(
        data=results,
        x="date",
        y="account value",
        hue="agent",
        palette="colorblind",
    )
    # Anzahl der x-Achsenticks reduzieren, um nur jeden fünften zu beschriften
    #plt.xticks(tick_positions, results["date"].iloc[tick_positions], rotation=30)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"Agent_performance_{comparison}_{agent}.pdf")

main()