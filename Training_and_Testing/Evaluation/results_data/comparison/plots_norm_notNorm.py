import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

a2c = pd.concat(
    map(
        pd.read_csv,
        [
            "resulta2c0_normFalse.csv",
            "resulta2c1_normFalse.csv",
            "resulta2c2_normFalse.csv",

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
            "resultddpg0_normFalse.csv",
            "resultddpg1_normFalse.csv",
            "resultddpg2_normFalse.csv",

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
            "resultppo0_normFalse.csv",
            "resultppo1_normFalse.csv",
            "resultppo2_normFalse.csv",

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
            "resulttd30_normFalse.csv",
            "resulttd31_normFalse.csv",
            "resulttd32_normFalse.csv",

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
            "resultsac0_normFalse.csv",
            "resultsac1_normFalse.csv",
            "resultsac2_normFalse.csv",

        ],
    ),
    ignore_index=True,
)
sac.drop(['ddpg', 'ppo', 'a2c', 'td3'], axis=1, inplace=True)
sac['agent'] = 'sac'
sac.rename(columns={'sac': 'account value'}, inplace=True)

sac.to_csv("sac_combined.csv")

a2c_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "resulta2c0_normTrue.csv",
            "resulta2c1_normTrue.csv",
            "resulta2c2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
a2c.drop(['ddpg', 'ppo', 'sac', 'td3'], axis=1, inplace=True)
a2c['agent'] = 'a2c normalized'
a2c.rename(columns={'a2c': 'account value'}, inplace=True)

a2c.to_csv("a2c_combined.csv")

ddpg_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "resultddpg0_normTrue.csv",
            "resultddpg1_normTrue.csv",
            "resultddpg2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
ddpg.drop(['a2c', 'ppo', 'sac', 'td3'], axis=1, inplace=True)
ddpg['agent'] = 'ddpg normalized'
ddpg.rename(columns={'ddpg': 'account value'}, inplace=True)

ddpg.to_csv("ddpg_combined.csv")

ppo_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "resultppo0_normTrue.csv",
            "resultppo1_normTrue.csv",
            "resultppo2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
ppo.drop(['ddpg', 'a2c', 'sac', 'td3'], axis=1, inplace=True)
ppo['agent']='ppo normalized'
ppo.rename(columns={'ppo': 'account value'}, inplace=True)

ppo.to_csv("ppo_combined.csv")

td3_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "resulttd30_normTrue.csv",
            "resulttd31_normTrue.csv",
            "resulttd32_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
td3.drop(['ddpg', 'ppo', 'sac', 'a2c'], axis=1, inplace=True)
td3['agent'] = 'td3 normalized'
td3.rename(columns={'td3': 'account value'}, inplace=True)

td3.to_csv("td3_combined.csv")

sac_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "resultsac0_normTrue.csv",
            "resultsac1_normTrue.csv",
            "resultsac2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
sac.drop(['ddpg', 'ppo', 'a2c', 'td3'], axis=1, inplace=True)
sac['agent'] = 'sac normalized'
sac.rename(columns={'sac': 'account value'}, inplace=True)

sac.to_csv("sac_combined.csv")


results = pd.concat(
    [
        a2c,
        ddpg,
        ppo,
        td3,
        sac,
        a2c_norm,
        ddpg_norm,
        ppo_norm,
        td3_norm,
        sac_norm
    ],
    ignore_index=True
)
results.to_csv("data_concat.csv")


sns.set(font_scale=1.4)
sns.set_style("whitegrid")
plt.figure(figsize=(25, 6))
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
#plt.xticks(range(0, len(results), 20))
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("Agent_performance.pdf")
