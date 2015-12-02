# Daniel Green, Gregory Green, 2014
# drgreen@fas.harvard.edu
# Human Evolutionary Biology
# Center for Astrophysics
# Harvard University
#
# Mineralization Model Re-Size:
# this code takes a larger mineralization model
# and produces images demonstrating mineral density
# increase over time, total density over time, or
# calculates final isotope distributions at full
# or partial resolution.
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import nlopt
from PIL import Image

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.misc import imresize
from blood_delta import calc_blood_step, calc_water_step2, calc_water_gaussian, calc_blood_gaussian, blood_delta, tooth_phosphate_reservoir
from blood_delta_experimentation import PO4_dissoln_reprecip
import scipy.special as spec

from blood_delta import calc_blood_gaussian
from scipy.optimize import curve_fit, minimize, leastsq

from time import time

def create_day_sample_arrays(data_tuple):
    '''
    This function effectively replicates the zip function.
    :param data_tuple: a list of tuples for (day, data) sets
    :return: two arrays, one for days, one for data.
    '''

    days, data = zip(*data_tuple)

    return days, data

def create_sample_tuple(days, data):
    '''
    Takes an array of days and array of data with nans,
    returns a list of tuples, with nan day and data sets
    removed.

    Inputs
    days:       1D array of days on which samples drawn.
    data:       1D array of sample values or nans for each day.
    Returns
    data_tuple: List of day-data tuples with nan elements removed.
    '''

    # Make a list of tuples from two arrays
    data_tuple_array = np.array(zip(days,data))
    # Delete NaN values
    data_tuple = data_tuple_array[~np.isnan(data_tuple_array).any(1)].tolist()

    return data_tuple

def load_sample_data():
    '''
    The place where I input or load data. Right now, this function
    returns an array with either sample days, or data from each animal
    in axis 0 (rows), and sampling days or data in axis 1 (columns).

    Inputs
    None.

    Outputs
    data_array:     2D array (sample days+ number of sheep by number of sample days)
    '''

    sample_days = np.array([30.00, 57.00, 73.00, 101.00, 130.00, 161.00, 169.00, 199.00, 201.00, 203.00, 203.00, 205.00, 207.00, 211.00, 211.00, 215.00, 219.00, 231.00, 239.00, 261.00, 271.00, 281.00, 283.00, 285.00, 289.00, 291.00, 297.00, 297.00, 309.00, 321.00, 357.00, 382.00, 410.00, 422.00, 452.00, 468.00, 482.00, 495.00])
    s951 = np.array([np.nan, -5.55, np.nan, -3.46, -3.99, -3.85, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    s949 = np.array([np.nan, np.nan, -4.88, -4.21, -3.86, -3.35, -3.00, -4.94, np.nan, np.nan, -5.02, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -3.57, np.nan, np.nan, -4.37, np.nan, -4.34, -4.67, np.nan, np.nan, np.nan, -5.22, np.nan, -4.72, -9.14, -6.02, np.nan, -6.16, -4.97, -5.04, -4.15, np.nan])
    s957 = np.array([np.nan, -5.78, np.nan, -4.08, -3.88, -3.36, np.nan, -4.28, np.nan, np.nan, -4.54, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -4.06, np.nan, np.nan, np.nan, np.nan, np.nan, -4.65, np.nan, -10.17, np.nan, np.nan, np.nan, -4.64, np.nan, np.nan, np.nan])
    s963 = np.array([np.nan, -6.03, np.nan, -3.72, -3.68, -3.43, np.nan, -4.74, np.nan, np.nan, -4.30, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -4.34, np.nan, np.nan, np.nan, np.nan, np.nan, -4.76, np.nan, -8.83, np.nan, np.nan, -6.87, -4.70, np.nan, -4.03, np.nan])
    s947 = np.array([np.nan, -5.69, -4.91, -3.95, -3.89, -3.00, -4.80, -4.17, -7.58, -9.50, -9.59, np.nan, -10.87, -11.96, -12.02, -12.26, -12.25, -11.75, -12.11, -12.36, -6.10, -4.43, -4.25, np.nan, -4.41, -4.25, -4.29, -4.03, -4.96, np.nan, -12.34, -7.28, np.nan, -8.01, -4.63, -5.22, -3.81, np.nan])
    s962 = np.array([np.nan, -5.71, -5.01, -4.07, -3.96, -4.53, -3.95, -4.96, -8.56, -10.34, -10.10, np.nan, -12.21, -13.14, -13.09, -13.49, -13.16, -12.93, -13.46, -13.29, -5.68, -4.87, -4.76, np.nan, -4.97, np.nan, -4.60, -4.65, -4.94, -5.45, -9.34, -5.56, np.nan, -6.55, -4.25, np.nan, -4.31, np.nan])
    s948 = np.array([np.nan, np.nan, -5.39, -4.27, -4.17, np.nan, -3.42, -4.15, -7.53, np.nan, -9.40, -10.15, np.nan, np.nan, -11.91, np.nan, -11.76, -4.40, -4.15, -4.57, -4.63, -4.60, -4.44, -7.70, np.nan, np.nan, np.nan, -4.64, -5.06, -5.53, -9.47, -3.80, np.nan, -7.95, np.nan, -4.66, -3.76, np.nan])
    s950 = np.array([np.nan, -6.07, -4.99, -3.59, -4.02, -3.67, -4.08, -4.85, -8.46, np.nan, -10.43, -11.33, np.nan, np.nan, np.nan, np.nan, -12.55, -4.53, -4.59, -4.70, -5.52, -5.01, np.nan, -4.89, np.nan, np.nan, np.nan, -4.56, -4.78, -4.61, np.nan, np.nan, np.nan, -6.87, -4.66, -5.94, -4.04, -5.25])
    s961 = np.array([np.nan, -6.20, -4.99, -4.21, np.nan, -3.64, -3.75, -4.55, -8.57, np.nan, -10.81, -11.89, np.nan, np.nan, -12.63, np.nan, -12.63, -4.56, -4.34, -4.75, -13.03, -13.22, -10.26, -8.64, np.nan, -5.83, np.nan, np.nan, -4.92, np.nan, -11.06, np.nan, np.nan, -7.01, -4.49, -5.55, -4.04, -4.30])
    s964 = np.array([np.nan, np.nan, np.nan, -4.23, -3.98, -4.14, -3.80, -4.42, -8.39, np.nan, -9.92, -11.17, np.nan, np.nan, -12.76, np.nan, -12.55, -4.60, -4.16, -4.45, -12.11, -12.87, -10.44, -8.81, np.nan, -6.13, np.nan, -4.87, -4.75, -6.06, -9.16, np.nan, np.nan, -5.57, -5.17, -5.61, -4.73, np.nan])
    bedford_water = np.array([-6.04, np.nan, -6.19, np.nan, -6.85, np.nan, -7.01, -6.61, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -6.32, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -5.94, -6.14, -5.93, np.nan, -7.35, np.nan, -6.83, -6.65, -6.98])
    montana_water = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -19.41, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -19.31, np.nan, np.nan, -19.38, -19.43, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    bedford_snow = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -17.63, np.nan, -13.66, -13.67, np.nan, np.nan, np.nan, np.nan])
    hadley_water = np.array([-8.83, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    data_array = np.concatenate([sample_days, s951, s949, s957, s963, s947, s962, s948, s950, s961, s964, bedford_water, montana_water, bedford_snow, hadley_water]).reshape(15, np.size(sample_days))

    return data_array

def plot_blood(days, blood):
    '''
    '''

    t_save = time()
    textstr = '949'

    data_tuple_all = []
    for i in xrange(blood[0]):
        data_tuple_all.append

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(days, blood, 'ro', linewidth=1.0)
    ax1.text(350, -20, textstr, fontsize=8)
    ax1.set_ylim(-30, 0)
    ax1.set_xlim(70, 500)

    fig.savefig('blood_test_image_{0}.svg'.format(t_save), dpi=300, bbox_inches='tight')
    plt.show()

def main():

    all_data = load_sample_data()
    sample_days = all_data[0]
    plot_blood(sample_days, all_data)

    return 0

if __name__ == '__main__':
    main()




