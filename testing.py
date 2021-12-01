import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
data_set_path = '/home/sedmdev/ANT_Fitting/1118060051368/Data/1118060051368_avg_data.dat'
data = pd.read_csv(data_set_path, usecols=(0, 1, 2), header=None)
print(data)

