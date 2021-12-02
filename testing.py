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
#data = data.sort_values(by=0, ascending=True, ignore_index=True)

data_flux_sorted = data.sort_values(by=1, ascending=True, ignore_index=True)
print(len(data_flux_sorted))
peak_prct = 20
peak_detections = int(np.round(len(data) * (peak_prct / 100)))
peak_prct_values = np.median(data_flux_sorted[1][len(data) - peak_detections:])
print(peak_prct_values)

peak_data_dict = {"index": data[1].idxmax(), "time": data[0][data[1].idxmax()], "amplitude": data[1][data[1].idxmax()]}
#peak_data_dict = {"index": n_highest_index, "time": data[0][n_highest_index], "amplitude": data[1][n_highest_index]}

offset_prct = 20
num_offset_detections = int(np.round(len(data) * (offset_prct / 100)))

gauss_data_dict = {"t_0": data[0][0],
                   "t_f": peak_data_dict["time"],
                   "t_g": np.std(data[0][0:peak_data_dict["index"]]),
                   "r_g": np.mean(data[1][0:num_offset_detections]),
                   "A_g": peak_data_dict["amplitude"] - np.mean(data[1][0:num_offset_detections]),
                   }

expdec_data_dict = {"t_0": peak_data_dict["time"],
                    "t_f": data[0][len(data) - 1],
                    "t_e": np.std(data[0][peak_data_dict["index"]:]),
                    "r_e": np.mean(data[1][len(data) - num_offset_detections:]),
                    "A_e": peak_data_dict["amplitude"] - np.mean(data[1][len(data) - num_offset_detections:]),
                    }

print(f'Number of Detections in Data Set: {len(data)}')
print(f'Data Starting Time: {data[0][0]}')
print(f'Data Ending Time: {data[0][len(data)-1]}\n')

print(f'Flux Offset Percentage: {offset_prct}%')
print(f'Number of Offset Detections: {num_offset_detections}\n')

print(f'Peak Index: {peak_data_dict["index"]}')
print(f'Peak Time: {peak_data_dict["time"]}')
print(f'Peak Amplitude: {peak_data_dict["amplitude"]}\n')

full_range = []
gauss_range = np.linspace(gauss_data_dict["t_0"], gauss_data_dict["t_f"], 1000)
print(f'Length of Gauss Range: {len(gauss_range)}')
print(f'Shape of Gauss Array: {gauss_range.shape}')
print(f'Start Day in Gauss Range: {gauss_range.min()}')
print(f'Max Day in Gauss Range: {gauss_range.max()}\n')

expdec_range = np.linspace(expdec_data_dict["t_0"], expdec_data_dict["t_f"], 1000)
print(f'Length of Expdec Range: {len(expdec_range)}')
print(f'Shape of Expdec Array: {expdec_range.shape}')
print(f'Start Day in Expdec Range: {expdec_range.min()}')
print(f'Max Day in Expdec Range: {expdec_range.max()}\n')


for i in range(len(gauss_range)):
    full_range.append(gauss_range[i])
for i in range(len(expdec_range)):
    full_range.append(expdec_range[i])
full_range = np.array(full_range)
full_range = np.unique(full_range)

print(f'Length of Full Range: {len(full_range)}')
print(f'Start Day in Full Range: {full_range.min()}')
print(f'End Day in Full Range: {full_range.max()}')
fit_inflection_position = None
for i in range(len(full_range)):
    if full_range[i] == peak_data_dict["time"]:
        fit_inflection_position = i
print(f'Fit Inflection Position: {fit_inflection_position}')

fit_values = []
for i in range(len(full_range)):
    if i <= fit_inflection_position:
        fit_values.append(gaussian(rise_time=full_range[i],
                                   amplitude=gauss_data_dict["A_g"],
                                   rise_peaktime=peak_data_dict["time"],
                                   stddev=gauss_data_dict["t_g"],
                                   offset=gauss_data_dict["r_g"]))
    if i > fit_inflection_position:
        fit_values.append(exp_dec(fall_time=full_range[i],
                                  amplitude=expdec_data_dict["A_e"],
                                  fall_peaktime=peak_data_dict["time"],
                                  stddev=expdec_data_dict["t_e"],
                                  offset=expdec_data_dict["r_e"]))


