# Enamel standard regressions



import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import spline
from numpy import linspace
import scipy.ndimage.interpolation as imginterp
import scipy.ndimage.filters as filters
from os.path import abspath, expanduser
import argparse, sys
from pylab import *

fig = plt.figure()
x1 = [79.1293578, 83.71110092, 91.56541284, 104.6987156, 135.766496, 137.6644295, 135.8352373, 144.5023723]
y1 = [1.19, 1.26, 1.39, 1.65, 2.13, 2.24, 2.37, 3.09]
fit = polyfit(x1, y1, 1)
fit_fn1 = poly1d(fit)

x2 = [56.21962, 58.30071, 64.71261, 80.02926, 124.8818, 130.2464, 141.7281, 167.4438]
y2 = [1.19, 1.26, 1.39, 1.65, 2.13, 2.24, 2.37, 3.09]
fit = polyfit(x2, y2, 1)
fit_fn2 = poly1d(fit)

x3 = [60.24224884, 64.76666977, 73.59379767, 91.15387907, 155.0427284, 164.1166056,177.6297519, 211.3790076]
y3 = [1.19, 1.26, 1.39, 1.65, 2.13, 2.24, 2.37, 3.09]
fit = polyfit(x3, y3, 1)
fit_fn3 = poly1d(fit)

plt.plot(x1, y1, 'yo')
plt.plot(x1, fit_fn1(x1), 'y--', label=r'Mo, no filter, $R^2$=.86')
plt.plot(x2, y2, 'go')
plt.plot(x2, fit_fn2(x2), 'g--', label=r'W, Cu filter, $R^2$=.97')
plt.plot(x3, y3, 'bo')
plt.plot(x3, fit_fn3(x3), 'b--', label=r'Synchrotron, $R^2$=.97')
plt.title('1.a  Linking grayscales to density', fontsize=16)
plt.xlabel('Gray scale', fontsize=16)
plt.ylabel(r'Density in $g/cm^3$', fontsize=16)
plt.legend(loc='upper left')

plt.show()
fig.savefig('graydense3.png', dpi=500, edgecolor='none')

