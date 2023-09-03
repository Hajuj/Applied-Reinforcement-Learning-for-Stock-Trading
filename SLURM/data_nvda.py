import pandas as pd
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

ticker = "NVDA"

# Define the relative path
relative_path = os.path.join('stock_data', f'{ticker}.csv')

# Read data
test = pd.read_csv(os.path.abspath(relative_path))

# Sort by two columns and calculate the daily log returns for normalized data
test = test.sort_values(['Time'], ascending=True)

# Zeitstempel kopieren und leere Tage auffüllen
test['Time'] = test['Time'].astype('datetime64[ns]')
index = pd.date_range(start=test["Time"].min(), end=test["Time"].max(), freq="60T")
df_test = pd.DataFrame()
df_test = df_test.reindex(index, fill_value=np.nan).interpolate()

# Time steps numerisch darstellen
df_test["t_steps"] = range(len(index))

test['Time'] = test['Time'].astype('datetime64[ns]')

# Wieder auf alte time steps (jetzt numerisch) reduzieren
final = pd.merge(test, df_test, how='left', left_on=['Time'], right_on=df_test.index)
outlier_thresh = 0.9

# Treat y as position, and that y-dot is an unobserved state
# state vector [y, y_dot]
# transition_matrix =  [[1, dt], [0, 1]]

observation_matrix = np.asarray([[1, 0]])

# observation time steps:
t = final["t_steps"]

# dt between observations:
dt = [np.mean(np.diff(t))] + list(np.diff(t))
transition_matrices = np.asarray([[[1, each_dt], [0, 1]]
                                  for each_dt in dt])
# observations
y = np.transpose(np.asarray([test['Close']]))

y = np.ma.array(y)

leave_1_out_cov = []

for i in range(len(y)):
    y_masked = np.ma.array(copy.deepcopy(y))
    y_masked[i] = np.ma.masked

    kf1 = KalmanFilter(transition_matrices=transition_matrices,
                       observation_matrices=observation_matrix)

    kf1 = kf1.em(y_masked)

    leave_1_out_cov.append(kf1.observation_covariance[0, 0])

# Find indexes that contributed excessively to observation covariance
outliers = (leave_1_out_cov / np.mean(leave_1_out_cov)) < outlier_thresh

for i in range(len(outliers)):
    if outliers[i]:
        y[i] = np.ma.masked

kf1 = KalmanFilter(transition_matrices=transition_matrices,
                   observation_matrices=observation_matrix)

kf1 = kf1.em(y)

(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(y)

plt.figure()
plt.plot(t.to_numpy(), y, 'go-', label="Observations")
plt.plot(t.to_numpy(), smoothed_state_means[:, 0].flatten(), 'b--', label="Value Estimate")
plt.legend(loc="upper left")
plt.xlabel("Time (s)")
plt.ylabel("Value (unit)")

plt.savefig(ticker + ".png")

test['Close'] = smoothed_state_means[:, 0]

folder_path = os.path.join('stock_data', f'{ticker}_smoothed.csv')

# Save the modified DataFrame back to the CSV file
test.to_csv(folder_path, index=False)
