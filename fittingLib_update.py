import time
import timeit
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import getpass


def get_datetime():
    # DATE STRINGS
    month = time.strftime("%m")
    day = time.strftime("%d")
    year = time.strftime("%Y")

    # TIME STRINGS
    hour = time.strftime("%I")
    min = time.strftime("%M")
    sec = time.strftime("%S")
    am_pm = time.strftime("%p")

    date_stamp = f'{month}.{day}.{year}'
    time_stamp = f'{hour}:{min}:{sec} {am_pm}'
    return date_stamp, time_stamp


def sci_not(val):
    formatted_val = "{:.2e}".format(val)
    return formatted_val


def gaussian(rise_time, amplitude, rise_peaktime, stddev, offset):
    return amplitude * np.exp(-(((rise_time-rise_peaktime)**2)/(2 * (stddev ** 2)))) + offset


def exp_dec(fall_time, amplitude, fall_peaktime, stddev, offset):
    return amplitude * np.exp(-((fall_time-fall_peaktime)/stddev)) + offset


class FittingLibrary():
    def __init__(self, pause=0.5,
                 user='default',
                 poly_degree=5,
                 sigma_coefficient=5,
                 offset_prct=20,
                 flux_error=.000005):

        # Checks to see if a directory for all fitting files exists, if not then it makes one in the users home folder
        path = f'/home/{getpass.getuser()}/ANT_Fitting'
        if not os.path.exists(path):
            os.mkdir(path)

        self.fig_size = [12, 8]

        self.pause_time = pause
        self.log_file = None
        self.user = user
        self.flux_error = flux_error

        self.data_sets = os.listdir(os.path.abspath('/home/sedmdev/Research/ant_fitting/CRTS_Test_Data'))
        self.filename = None
        self.plot_title = None
        self.home_dir = os.path.abspath(path)
        self.current_dir = None

        self.raw_data = None
        self.raw_data_length = None
        self.raw_data_peak_idx = None
        self.raw_data_peak_list = None
        self.raw_data_time_range_list = None

        self.mag_data = None
        self.mag_data_length = None
        self.mag_data_peak_idx = None
        self.mag_data_peak_list = None
        self.mag_data_time_range_list = None

        self.flux_data = None
        self.flux_data_length = None
        self.flux_data_peak_idx = None
        self.flux_data_peak_list = None
        self.flux_data_time_range_list = None

        self.poly_degree = poly_degree
        self.sigma_coefficient = sigma_coefficient
        self.polytrend = None
        self.polytrend_sigma = None

        self.sigma_idx = None
        self.sigma_excluded = None
        self.sigma_retained = None

        self.sigma_clip_data = None
        self.sigma_clip_data_length = None
        self.sigma_clip_data_peak_idx = None
        self.sigma_clip_data_peak_list = None
        self.sigma_clip_data_time_range_list = None

        # Variables for the Sigma Clipped and Averaged Data Frame
        self.avg_data = None
        self.avg_data_length = None
        self.avg_data_peak_idx = None
        self.avg_data_peak_list = None
        self.avg_data_time_range_list = None

        # Variables For Determining Fit Parameters (NOTE: THESE ALL DEPEND ON THE AVERAGED DATA SET!)
        # General:
        self.offset_prct = (offset_prct / 100)
        self.num_offset_detections = None

        # Gaussian:


        # Exponential Decay:


        # Combining the Fit:


    def import_data(self, file):
        self.filename = file
        self.plot_title = f'{self.filename[:-4]}'

        dir_path = f'{self.home_dir}/{self.filename[:-4]}'
        # Checks to see if a directory for this data set exists, if it doesn't then it creates one
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)  # Makes the data set directory
            self.current_dir = os.path.abspath(dir_path)
            os.mkdir(f'{self.current_dir}/Plots')  # Makes a "Plots" subdirectory
            os.mkdir(f'{self.current_dir}/Data')  # Makes a "Data" subdirectory
        self.current_dir = os.path.abspath(dir_path)

        # Finds the data set based on the filename provided and creates a dataframe
        data_path = os.path.abspath('/home/sedmdev/Research/ant_fitter/CRTS_Test_Data')
        data_set_path = os.path.join(data_path, file)
        print(data_set_path)
        data = pd.read_csv(data_set_path, usecols=(0, 1, 2), delim_whitespace=True, header=None)

        # Creates a new dataframe for the sorted magnitude data
        mag_data = data.sort_values(by=0, ascending=True, ignore_index=True)

        # Creates a new dataframe for the sorted data that has been converted from magnitude to flux
        # Also sets the error value to be used for the flux data
        flux_data = data.sort_values(by=0, ascending=True, ignore_index=True)
        flux_data[1] = flux_data[1].apply(lambda x: 3631.0 * (10.0 ** (-0.4 * x)))
        flux_data[2] = self.flux_error  # This is a placeholder

        self.raw_data = data
        self.raw_data_length = len(self.raw_data)
        self.raw_data_peak_idx = self.raw_data[1].idxmin()
        self.raw_data_peak_list = [self.raw_data[0][self.raw_data_peak_idx],
                                   self.raw_data[1][self.raw_data_peak_idx]]

        self.mag_data = mag_data
        self.mag_data_length = len(self.mag_data)
        self.mag_data_peak_idx = self.mag_data[1].idxmin()
        self.mag_data_peak_list = [self.mag_data[0][self.mag_data_peak_idx],
                                   self.mag_data[1][self.mag_data_peak_idx]]

        self.flux_data = flux_data
        self.flux_data_length = len(self.flux_data)
        self.flux_data_peak_idx = self.flux_data[1].idxmax()
        self.flux_data_peak_list = [self.flux_data[0][self.flux_data_peak_idx],
                                    self.flux_data[1][self.flux_data_peak_idx]]

        # Saves two new data frames. One for the sorted magnitude data, and one for the sorted flux data, saves to
        # the "Data" subdirectory
        self.mag_data.to_csv(f'{self.current_dir}/Data/{self.plot_title}_sorted_mag.dat',
                             index=False,
                             header=False,
                             )

        self.flux_data.to_csv(f'{self.current_dir}/Data/{self.plot_title}_sorted_flux.dat',
                              index=False,
                              header=False,
                              )

        # Writes out basic info taken from the import
        self.log_file = open(f'{self.current_dir}/{self.plot_title}_log.txt', 'w')
        self.log_file.write(f'RUN INFORMATION\n'
                            f'Source ID: {file}\n'
                            f'Source Path: {data_set_path}\n'
                            f'Date: {get_datetime()[0]} @ {get_datetime()[1]}\n'
                            f'User: {self.user}\n'
                            f'\n'
                            )

        self.log_file.write(f'RAW DATA INFORMATION\n'
                            f' > Total Detections: {self.raw_data_length}\n'
                            f' > Peak Index: {self.raw_data_peak_idx}\n'
                            f' > Peak Time (tp,raw) ~ {sci_not(self.raw_data_peak_list[0])} MJD\n'
                            f' > Peak Amplitude (Ap,raw) ~ {sci_not(self.raw_data_peak_list[1])} Mag\n'
                            f'\n'
                            )

        self.log_file.write(f'MAGNITUDE DATA INFORMATION\n'
                            f' > Total Detections: {self.mag_data_length}\n'
                            f' > Peak Index: {self.mag_data_peak_idx}\n'
                            f' > Peak Time (tp,mag) ~ {sci_not(self.mag_data_peak_list[0])} MJD\n'
                            f' > Peak Amplitude (Ap,mag) ~ {sci_not(self.mag_data_peak_list[1])} Mag\n'
                            f'\n'
                            )

        self.log_file.write(f'FlUX DATA INFORMATION\n'
                            f' > Total Detections: {self.flux_data_length}\n'
                            f' > Peak Index: {self.flux_data_peak_idx}\n'
                            f' > Peak Time (tp,flux) ~ {sci_not(self.flux_data_peak_list[0])} MJD\n'
                            f' > Peak Amplitude (Ap,flux) ~ {sci_not(self.flux_data_peak_list[1])} Jy\n'
                            f'\n'
                            )

    def plot_raw(self, show=True, save=True):
        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.fig_size[0], self.fig_size[1])
        ax.set_title(f'{self.plot_title} Light Curve [Raw]')
        window_name = f'{self.plot_title}_raw_magnitude_light_curve'
        fig.canvas.manager.set_window_title(window_name)

        ax.set(xlabel='Modified Julian Day [MJD]', ylabel='Magnitude')
        ax.invert_yaxis()

        # Plots the Light Curve:
        ax.errorbar(self.raw_data[0],
                    self.raw_data[1],
                    yerr=self.raw_data[2],
                    linestyle='none',
                    marker='s',
                    ms=3,
                    color='black'
                    )

        # Plots the Peak Location:
        ax.errorbar(self.raw_data[0][self.raw_data_peak_idx],
                    self.raw_data[1][self.raw_data_peak_idx],
                    yerr=self.raw_data[2][self.raw_data_peak_idx],
                    linestyle='none',
                    marker='s',
                    ms=5,
                    color='red'
                    )

        if save:
            plt.savefig(f'{self.current_dir}/Plots/{window_name}.png')

        if show:
            plt.pause(self.pause_time)
            plt.show(block=False)
            plt.close()

        plt.close()

    def plot_mag(self, show=True, save=True):
        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.fig_size[0], self.fig_size[1])
        ax.set_title(f'{self.plot_title} Light Curve [Magnitude]')
        window_name = f'{self.plot_title}_magnitude_light_curve'
        fig.canvas.manager.set_window_title(window_name)

        ax.set(xlabel='Modified Julian Day [MJD]', ylabel='Magnitude')
        ax.invert_yaxis()
        # Plots the Light Curve
        ax.errorbar(self.mag_data[0],
                    self.mag_data[1],
                    yerr=self.mag_data[2],
                    linestyle='none',
                    marker='s',
                    ms=3,
                    color='black'
                    )

        # Plots the Peak Location:
        ax.errorbar(self.mag_data[0][self.mag_data_peak_idx],
                    self.mag_data[1][self.mag_data_peak_idx],
                    yerr=self.mag_data[2][self.mag_data_peak_idx],
                    linestyle='none',
                    marker='s',
                    ms=5,
                    color='red'
                    )

        if save:
            plt.savefig(f'{self.current_dir}/Plots/{window_name}.png')

        if show:
            plt.pause(self.pause_time)
            plt.show(block=False)
            plt.close()

        plt.close()

    def plot_flux(self, show=True, save=True):
        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.fig_size[0], self.fig_size[1])
        ax.set_title(f'{self.plot_title} Light Curve [Flux]')
        window_name = f'{self.plot_title}_flux_light_curve'
        fig.canvas.manager.set_window_title(window_name)

        ax.set(xlabel='Modified Julian Day [MJD]', ylabel='Flux [Jy]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.errorbar(self.flux_data[0],
                    self.flux_data[1],
                    yerr=self.flux_data[2],
                    linestyle='none',
                    marker='s',
                    ms=3,
                    color='black'
                    )

        # Plots the Peak Location:
        ax.errorbar(self.flux_data[0][self.flux_data_peak_idx],
                    self.flux_data[1][self.flux_data_peak_idx],
                    yerr=self.flux_data[2][self.flux_data_peak_idx],
                    linestyle='none',
                    marker='s',
                    ms=5,
                    color='red'
                    )

        if save:
            plt.savefig(f'{self.current_dir}/Plots/{window_name}.png')

        if show:
            plt.pause(self.pause_time)
            plt.show(block=False)
            plt.close()

        plt.close()

    def polyfit_sigma_clipping(self):
        poly_format = {0: 'g(x) = (a)',
                       1: 'g(x) = (ax) + b',
                       2: 'g(x) = (ax^2) + (bx) + c',
                       3: 'g(x) = (ax^3) + (bx^2) + (cx) + d',
                       4: 'g(x) = (ax^4) + (bx^3) + (cx^2) + (dx) + e',
                       5: 'g(x) = (ax^5) + (bx^4) + (cx^3) + (dx^2) + (ex) + f'
                       }

        # Returns the coefiicients of the polynomial fit
        poly_coefficients = np.polyfit(self.flux_data[0], self.flux_data[1], self.poly_degree)
        poly_coefficients_vars = ['a', 'b', 'c', 'd', 'e', 'f']

        self.polytrend = np.polyval(poly_coefficients, self.flux_data[0])
        self.polytrend_sigma = self.sigma_coefficient * np.std(self.polytrend)

        self.sigma_idx = []
        for i in range(len(self.flux_data)):
            if (self.flux_data[1][i] - self.flux_data[2][i]) >= self.polytrend[i] + self.polytrend_sigma:
                self.sigma_idx.append(i)
            if (self.flux_data[1][i] + self.flux_data[2][i]) <= self.polytrend[i] - self.polytrend_sigma:
                self.sigma_idx.append(i)

        self.sigma_clip_data = self.flux_data.drop(labels=self.sigma_idx, axis=0, inplace=False).reset_index(drop=True)
        self.sigma_clip_data_length = len(self.sigma_clip_data)
        self.sigma_clip_data_peak_idx = self.sigma_clip_data[1].idxmax()
        self.sigma_clip_data_peak_list = [self.sigma_clip_data[0][self.sigma_clip_data_peak_idx],
                                          self.sigma_clip_data[1][self.sigma_clip_data_peak_idx]]

        self.sigma_clip_data.to_csv(f'{self.current_dir}/Data/{self.plot_title}_sigma_clipped.dat',
                                    index=False,
                                    header=False,
                                    )

        self.sigma_excluded = [len(self.sigma_idx), sci_not((len(self.sigma_idx) / self.flux_data_length) * 100.0)]
        self.sigma_retained = [len(self.sigma_clip_data),
                               sci_not((len(self.sigma_clip_data) / self.flux_data_length) * 100.0)]

        self.log_file.write(f'POLYNOMIAL FITTING INFORMATION\n'
                            f' > Degree: {self.poly_degree}\n'
                            f' > Polynomial Format: {poly_format[self.poly_degree]}\n'
                            f' > Calculated Sigma ~ {sci_not(self.polytrend_sigma)}\n'
                            f' > Coefficients (From Highest to Lowest Power):\n'
                            )
        for i in range(self.poly_degree + 1):
            self.log_file.write(f'    {poly_coefficients_vars[i]} ~ {sci_not(poly_coefficients[i])}\n')
        self.log_file.write(f'\n')

        self.log_file.write(f'SIGMA CLIPPING INFORMATION\n'
                            f' > Clipping Bound: (+/-) {self.sigma_coefficient} Sigma\n'
                            f' > Excluded {self.sigma_excluded[0]} of {self.flux_data_length} Detections ({self.sigma_excluded[1]} %)\n'
                            f' > Retained {self.sigma_retained[0]} of {self.flux_data_length} Detections ({self.sigma_retained[1]} %)\n'
                            f'\n'
                            )

        self.log_file.write(f'SIGMA CLIPPED DATA INFORMATION\n'
                            f' > Total Detections: {self.sigma_clip_data_length}\n'
                            f' > Peak Index: {self.sigma_clip_data_peak_idx}\n'
                            f' > Peak Time (tp,sig) ~ {sci_not(self.sigma_clip_data_peak_list[0])} MJD\n'
                            f' > Peak Amplitude (Ap,sig) ~ {sci_not(self.sigma_clip_data_peak_list[1])} Jy\n'
                            f'\n'
                            )

    def plot_sigma_clip(self, show=True, save=True):
        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.fig_size[0], self.fig_size[1])

        clipped_x = self.flux_data[0][self.sigma_idx]
        clipped_y = self.flux_data[1][self.sigma_idx]
        clipped_err = self.flux_data[2][self.sigma_idx]

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set(xlabel='Modified Julian Day [MJD]', ylabel='Flux [Jy]')
        ax.errorbar(self.sigma_clip_data[0],
                    self.sigma_clip_data[1],
                    yerr=self.sigma_clip_data[2],
                    linestyle='none',
                    marker='s',
                    ms=3,
                    color='black'
                    )

        ax.plot(self.flux_data[0],
                self.polytrend,
                linestyle='--',
                linewidth='1',
                color='black')

        if show:
            ax.set_title(f'{self.plot_title} Polynomial Fit')
            window_name = f'{self.plot_title}_polytrend'
            fig.canvas.manager.set_window_title(window_name)
            plt.pause(self.pause_time)
            plt.show(block=False)
        if save:
            plt.savefig(f'{self.current_dir}/Plots/{self.plot_title}_polytrend.png')

        ax.plot(self.flux_data[0],
                self.polytrend - self.polytrend_sigma,
                linestyle='--',
                linewidth='1',
                color='black')

        ax.plot(self.flux_data[0],
                self.polytrend + self.polytrend_sigma,
                linestyle='--',
                linewidth='1',
                color='black')

        ax.fill_between(self.flux_data[0],
                        self.polytrend - self.polytrend_sigma,
                        self.polytrend + self.polytrend_sigma,
                        color='whitesmoke')

        if show:
            ax.set_title(f'{self.plot_title} Sigma Clipping')
            window_name = f'{self.plot_title}_sigma_clipping'
            fig.canvas.manager.set_window_title(window_name)
            plt.pause(self.pause_time)
            plt.show(block=False)
        if save:
            plt.savefig(f'{self.current_dir}/Plots/{self.plot_title}_sigma_clipping.png')

        ax.errorbar(clipped_x,
                    clipped_y,
                    yerr=clipped_err,
                    linestyle='none',
                    marker='x',
                    ms=4,
                    color='red'
                    )

        if show:
            ax.set_title(f'{self.plot_title} Sigma Clipping [Show Excluded]')
            window_name = f'{self.plot_title}_sigma_clipping_show_clipped'
            fig.canvas.manager.set_window_title(window_name)
            plt.pause(self.pause_time)
            plt.show(block=False)
        if save:
            plt.savefig(f'{self.current_dir}/Plots/{self.plot_title}_sigma_clipping_show_clipped.png')

        plt.close()

    def get_average(self):
        unique_days_str = np.unique(self.sigma_clip_data[0].apply(lambda x: str(x)[0:5]))
        unique_days = []
        unique_fluxes_avg = []
        unique_errors_avg = []

        for i in range(len(unique_days_str)):
            flux_list_per_obs = []
            unique_days.append(int(unique_days_str[i]))
            unique_errors_avg.append(self.flux_error)
            for j in range(len(self.sigma_clip_data)):
                if unique_days_str[i] == str(self.sigma_clip_data[0][j])[0:5]:
                    flux_list_per_obs.append(self.sigma_clip_data[1][j])
            unique_fluxes_avg.append(np.average(flux_list_per_obs))

        self.avg_data = pd.DataFrame([unique_days, unique_fluxes_avg, unique_errors_avg]).T
        self.avg_data_length = len(self.avg_data)
        self.avg_data_peak_idx = self.avg_data[1].idxmax()
        self.avg_data_peak_list = [self.avg_data[0][self.avg_data_peak_idx],
                                   self.avg_data[1][self.avg_data_peak_idx]]

        self.avg_data.to_csv(f'{self.current_dir}/Data/{self.plot_title}_avg_data.dat', index=False,
                             header=False)

        self.log_file.write(f'AVERAGED DATA INFORMATION\n'
                            f' > Total Detections: {self.avg_data_length}\n'
                            f' > Peak Index: {self.avg_data_peak_idx}\n'
                            f' > Peak Time (tp) ~ {sci_not(self.avg_data_peak_list[0])} MJD\n'
                            f' > Peak Amplitude (Ap) ~ {sci_not(self.avg_data_peak_list[1])} Jy\n'
                            f'\n'
                            )
        return self.avg_data

    def plot_avg(self, show=True, save=True):
        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.fig_size[0], self.fig_size[1])
        ax.set_title(f'{self.plot_title} Light Curve [Averaged]')
        window_name = f'{self.plot_title}_avg_light_curve'
        fig.canvas.manager.set_window_title(window_name)

        ax.set(xlabel='Modified Julian Day [MJD]', ylabel='Flux [Jy]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.errorbar(self.avg_data[0],
                    self.avg_data[1],
                    yerr=self.avg_data[2],
                    linestyle='none',
                    marker='s',
                    ms=3,
                    color='black'
                    )

        # Plots the Peak Location:
        ax.errorbar(self.avg_data[0][self.avg_data_peak_idx],
                    self.avg_data[1][self.avg_data_peak_idx],
                    yerr=self.avg_data[2][self.avg_data_peak_idx],
                    linestyle='none',
                    marker='s',
                    ms=5,
                    color='red'
                    )

        if save:
            plt.savefig(f'{self.current_dir}/Plots/{window_name}.png')

        if show:
            plt.pause(self.pause_time)
            plt.show(block=False)
            plt.close()

        plt.close()

    def get_fit_parameters(self):
        n_highest_val = 2
        data_flux_sorted = self.avg_data.sort_values(by=1, ascending=True, ignore_index=True)
        n_highest_amplitude = data_flux_sorted[1][len(data_flux_sorted) - n_highest_val]
        n_highest_index = None
        for i in range(len(self.avg_data)):
            if self.avg_data[1][i] == n_highest_amplitude:
                n_highest_index = i

        # peak_data_dict = {"index": data[1].idxmax(), "time": data[0][data[1].idxmax()], "amplitude": data[1][data[1].idxmax()]}
        peak_data_dict = {"index": n_highest_index, "time": self.avg_data[0][n_highest_index],
                          "amplitude": self.avg_data[1][n_highest_index]}

        offset_prct = 20
        num_offset_detections = int(np.round(len(self.avg_data) * (offset_prct / 100)))

        gauss_data_dict = {"t_0": self.avg_data[0][0],
                           "t_f": peak_data_dict["time"],
                           "t_g": np.std(self.avg_data[0][0:peak_data_dict["index"]]),
                           "r_g": np.mean(self.avg_data[1][0:num_offset_detections]),
                           "A_g": peak_data_dict["amplitude"] - np.mean(self.avg_data[1][0:num_offset_detections]),
                           }

        expdec_data_dict = {"t_0": peak_data_dict["time"],
                            "t_f": self.avg_data[0][len(self.avg_data) - 1],
                            "t_e": np.std(self.avg_data[0][peak_data_dict["index"]:]),
                            "r_e": np.mean(self.avg_data[1][len(self.avg_data) - num_offset_detections:]),
                            "A_e": peak_data_dict["amplitude"] - np.mean(self.avg_data[1][len(self.avg_data) - num_offset_detections:]),
                            }

        print(f'Number of Detections in Data Set: {len(self.avg_data)}')
        print(f'Data Starting Time: {self.avg_data[0][0]}')
        print(f'Data Ending Time: {self.avg_data[0][len(self.avg_data) - 1]}\n')

        print(f'Flux Offset Percentage: {offset_prct}%')
        print(f'Number of Offset Detections: {num_offset_detections}\n')

        print(f'Peak Index: {peak_data_dict["index"]}')
        print(f'Peak Time: {peak_data_dict["time"]}')
        print(f'Peak Amplitude: {peak_data_dict["amplitude"]}\n')

        fig, ax = plt.subplots(1)
        fig.set_size_inches(12, 9)
        ax.set_title(f'ANT Curve Fitting')

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


    def plot_fit_parameters(self, show=True, save=True):
        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.fig_size[0], self.fig_size[1])
        ax.set_title(f'{self.plot_title} Light Curve [Fit Parameters]')
        window_name = f'{self.plot_title}_fit_light_curve'
        fig.canvas.manager.set_window_title(window_name)

        ax.set(xlabel='Modified Julian Day [MJD]', ylabel='Flux [Jy]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # plots just the data
        ax.errorbar(data[0],
                    data[1],
                    yerr=data[2],
                    linestyle='none',
                    marker='s',
                    ms=3,
                    color='black'
                    )

        # plots the lines defining the peak regions:
        ax.axhline(y=data[1][peak_data_dict["index"]],
                   xmin=0,
                   xmax=1,
                   color='black',
                   ls='--',
                   linewidth=0.5
                   )

        ax.axhline(y=0,
                   xmin=0,
                   xmax=1,
                   color='black',
                   ls='--',
                   linewidth=0.5
                   )

        ax.vlines(x=data[0][peak_data_dict["index"]],
                  ymin=0,
                  ymax=data[1][peak_data_dict["index"]],
                  color='black',
                  linewidth=0.5)

        # Plots the flux offset value for the gaussian
        ax.hlines(gauss_data_dict["r_g"],
                  data[0][0],
                  data[0][peak_data_dict["index"]],
                  color='black',
                  linestyles='--',
                  linewidth=0.5)

        # Plots the flux offset value for the exponential decay
        ax.hlines(expdec_data_dict["r_e"],
                  data[0][peak_data_dict["index"]],
                  data[0][len(data) - 1],
                  color='black',
                  linestyles='--',
                  linewidth=0.5)

        # Plots the fit data:
        ax.plot(full_range,
                fit_values,
                color='black',
                linestyle='--',
                linewidth=0.5)

        if save:
            plt.savefig(f'{self.current_dir}/Plots/{window_name}.png')

        if show:
            plt.pause(self.pause_time)
            plt.show(block=True)
            plt.close()

        plt.close()

    def fit_ant(self, path, toggle_show=True, toggle_save=True):
        self.import_data(file=path)
        self.plot_raw(save=toggle_save, show=toggle_show)
        self.plot_mag(save=toggle_save, show=toggle_show)
        self.plot_flux(save=toggle_save, show=toggle_show)
        self.polyfit_sigma_clipping()
        self.plot_sigma_clip(save=toggle_save, show=toggle_show)
        self.get_average()
        self.plot_avg(save=toggle_save, show=toggle_show)
        self.get_fit_parameters()
        self.plot_fit_parameters(save=toggle_save, show=toggle_show)
