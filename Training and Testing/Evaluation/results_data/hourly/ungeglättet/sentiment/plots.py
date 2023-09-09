import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

a2c = pd.concat(
    map(
        pd.read_csv,
        [
            "resulta2c0.csv",
            "resulta2c1.csv",
            "resulta2c2.csv",

        ],
    ),
    ignore_index=True,
)
a2c.drop(['ddpg', 'ppo', 'sac', 'td3'], axis=1, inplace=True)
a2c['agent'] = 'a2c'
a2c.rename(columns={'a2c': 'account value'}, inplace=True)

a2c.to_csv("a2c_combined.csv")

ddpg = pd.concat(
    map(
        pd.read_csv,
        [
            "resultddpg0.csv",
            "resultddpg1.csv",
            "resultddpg2.csv",

        ],
    ),
    ignore_index=True,
)
ddpg.drop(['a2c', 'ppo', 'sac', 'td3'], axis=1, inplace=True)
ddpg['agent'] = 'ddpg'
ddpg.rename(columns={'ddpg': 'account value'}, inplace=True)

ddpg.to_csv("ddpg_combined.csv")

ppo = pd.concat(
    map(
        pd.read_csv,
        [
            "resultppo0.csv",
            "resultppo1.csv",
            "resultppo2.csv",

        ],
    ),
    ignore_index=True,
)
ppo.drop(['ddpg', 'a2c', 'sac', 'td3'], axis=1, inplace=True)
ppo['agent']='ppo'
ppo.rename(columns={'ppo': 'account value'}, inplace=True)

ppo.to_csv("ppo_combined.csv")

td3 = pd.concat(
    map(
        pd.read_csv,
        [
            "resulttd30.csv",
            "resulttd31.csv",
            "resulttd32.csv",

        ],
    ),
    ignore_index=True,
)
td3.drop(['ddpg', 'ppo', 'sac', 'a2c'], axis=1, inplace=True)
td3['agent'] = 'td3'
td3.rename(columns={'td3': 'account value'}, inplace=True)

td3.to_csv("td3_combined.csv")

sac = pd.concat(
    map(
        pd.read_csv,
        [
            "resultsac0.csv",
            "resultsac1.csv",
            "resultsac2.csv",

        ],
    ),
    ignore_index=True,
)
sac.drop(['ddpg', 'ppo', 'a2c', 'td3'], axis=1, inplace=True)
sac['agent'] = 'sac'
sac.rename(columns={'sac': 'account value'}, inplace=True)

sac.to_csv("sac_combined.csv")


results = pd.concat(
    [
        a2c,
        ddpg,
        ppo,
        td3,
        sac
    ],
    ignore_index=True
)
results.to_csv("data_concat.csv")


sns.set(font_scale=1.4)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
plt.rc("text", usetex=True)
fig2 = sns.lineplot(
    data=results,
    x="date",
    y="account value",
    hue="agent",
    palette="colorblind",
)
# Anzahl der x-Achsenticks reduzieren, um nur jeden fünften zu beschriften
plt.xticks(range(0, len(results), 20))
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("Agent_performance.pdf")
