# Graphing growth data

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.special
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import spline
from numpy import linspace
import scipy.ndimage.interpolation as imginterp
import scipy.ndimage.filters as filters
from os.path import abspath, expanduser
import os, os.path
from datetime import datetime
import argparse, sys, warnings
from matplotlib.ticker import FuncFormatter
import time
import h5py
import emcee
from fit_positive_function import TMonotonicPointModel, TMCMC
from scipy.optimize import curve_fit, minimize, leastsq
from scipy.integrate import quad
from scipy import pi, sin
import scipy.special as spec
from lmfit import Model

def est_tooth_extension(x, amplitude, slope, offset):
    
    height_max = 40.5 # in millimeters
    extension_erf = (amplitude * spec.erf(slope * (x - offset))) + (height_max - amplitude)

    return extension_erf

def main():

    tooth_days = np.array([88., 92., 97., 100., 101., 101., 105., 124., 127., 140., 140., 159., 167., 174., 179., 202., 203., 222., 223., 251., 265., 285., 510., 510., 510., 510.])
    tooth_extension = np.array([2.22, 4.43, 6.23, 3.14, 7.16, 3.19, 6.74, 6.23, 9.61, 11.8, 12.3, 18.4, 16.9, 17.9, 18.9, 21, 24.59, 20.6, 26.23, 20.1, 30.60, 32.50, 41.79, 39.61, 39.39, 42.3])
    p0_var_ext = np.array([40, .0045, 0.])

    days = np.linspace(50, 550, 501)
    ext_variables, ext_pcov = curve_fit(est_tooth_extension, tooth_days, tooth_extension, p0_var_ext)
    print 'ext', ext_variables

    extension_erf = est_tooth_extension(days, ext_variables[0], ext_variables[1], ext_variables[2])
    print 'ext_erf', extension_erf.shape
    diff_extension_erf = np.diff(extension_erf) * 1000
    print 'diff_ext_erf', diff_extension_erf.shape

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ax2.plot(days[1::4], diff_extension_erf[::4], 'b.', label=r'$ \mathrm{extension} \ \Delta $', alpha=.5)
    ax2.set_ylim([0,300])
    ax2.set_xlim([50,550])
    ax.plot(tooth_days, tooth_extension, marker='o', linestyle='none', color='b', label=r'$ \mathrm{extension} \ \mathrm{(observed)} $')
    ax.plot(days, extension_erf, linestyle='-', color='b', label=r'$ \mathrm{extension,} \ \mathrm{optimized} $')
    ax.set_ylim([0,45])
    ax.set_xlim([50,550])
    plt.title('Enamel secretion and maturation progress over time')
    ax.set_xlabel('Days after birth')
    ax.set_ylabel('Progress from cusp tip in mm')
    ax2.set_ylabel('Secretion or maturation speed in um/day')
    #ax2.legend(loc='lower right', fancybox=True, framealpha=0.8)
    #ax.legend(loc='upper right', fancybox=True, framealpha=0.8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='center right')

    plt.show()

    return 0
if __name__ == '__main__':
    main()
