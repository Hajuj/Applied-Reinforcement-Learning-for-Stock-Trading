import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

a2c = pd.concat(
    map(
        pd.read_csv,
        [
            "result0_normFalse.csv",
            "result1_normFalse.csv",
            "result2_normFalse.csv",

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
            "result0_normFalse.csv",
            "result1_normFalse.csv",
            "result2_normFalse.csv",

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
            "result0_normFalse.csv",
            "result1_normFalse.csv",
            "result2_normFalse.csv",

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
            "result0_normFalse.csv",
            "result1_normFalse.csv",
            "result2_normFalse.csv",

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
            "result0_normFalse.csv",
            "result1_normFalse.csv",
            "result2_normFalse.csv",

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
            "result0_normTrue.csv",
            "result1_normTrue.csv",
            "result2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
a2c_norm.drop(['ddpg', 'ppo', 'sac', 'td3'], axis=1, inplace=True)
a2c_norm['agent'] = 'a2c normalized'
a2c_norm.rename(columns={'a2c': 'account value'}, inplace=True)

a2c_norm.to_csv("a2c_combined.csv")

ddpg_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "result0_normTrue.csv",
            "result1_normTrue.csv",
            "result2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
ddpg_norm.drop(['a2c', 'ppo', 'sac', 'td3'], axis=1, inplace=True)
ddpg_norm['agent'] = 'ddpg normalized'
ddpg_norm.rename(columns={'ddpg': 'account value'}, inplace=True)

ddpg_norm.to_csv("ddpg_combined.csv")

ppo_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "result0_normTrue.csv",
            "result1_normTrue.csv",
            "result2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
ppo_norm.drop(['ddpg', 'a2c', 'sac', 'td3'], axis=1, inplace=True)
ppo_norm['agent']='ppo normalized'
ppo_norm.rename(columns={'ppo': 'account value'}, inplace=True)

ppo_norm.to_csv("ppo_combined.csv")

td3_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "result0_normTrue.csv",
            "result1_normTrue.csv",
            "result2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
td3_norm.drop(['ddpg', 'ppo', 'sac', 'a2c'], axis=1, inplace=True)
td3_norm['agent'] = 'td3 normalized'
td3_norm.rename(columns={'td3': 'account value'}, inplace=True)

td3_norm.to_csv("td3_combined.csv")

sac_norm = pd.concat(
    map(
        pd.read_csv,
        [
            "result0_normTrue.csv",
            "result1_normTrue.csv",
            "result2_normTrue.csv",

        ],
    ),
    ignore_index=True,
)
sac_norm.drop(['ddpg', 'ppo', 'a2c', 'td3'], axis=1, inplace=True)
sac_norm['agent'] = 'sac normalized'
sac_norm.rename(columns={'sac': 'account value'}, inplace=True)

sac_norm.to_csv("sac_combined.csv")


"""results = pd.concat(
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
results.to_csv("data_concat.csv")"""

results = pd.concat(
    [
        ppo_norm
    ],
    ignore_index=True
)



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
plt.savefig("Agent_performance_ppo_single_norm.pdf")
