import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd


def gaussian(rise_time, amplitude, rise_peaktime, stddev, offset):
    return amplitude * np.exp(-(((rise_time - rise_peaktime) ** 2) / (2 * (stddev ** 2)))) + offset


def exp_dec(fall_time, amplitude, fall_peaktime, stddev, offset):
    return amplitude * np.exp(-((fall_time - fall_peaktime) / stddev)) + offset


data_set_path = '/home/sedmdev/ANT_Fitting/1118060051368/Data/1118060051368_avg_data.dat'
data = pd.read_csv(data_set_path, usecols=(0, 1, 2), header=None)
print(data)

fig, ax = plt.subplots(1)
fig.set_size_inches(12, 9)
ax.set_title(f'ANT Curve Fitting')

ax.set(xlabel='Modified Julian Day [MJD]', ylabel='Flux [Jy]')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.errorbar(data[0],
            data[1],
            yerr=data[2],
            linestyle='none',
            marker='s',
            ms=3,
            color='black'
            )

plt.show()
