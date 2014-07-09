#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  small_blood_to_toothmap.py
#  
#  Copyright 2014 Daniel Green, Greg Green
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#


import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import h5py
from bloodhist import calc_blood_hist
from PIL import Image
#from scipy.misc import imresize

def d18O_pixel(age, increase, blood_hist, x, y):

    #print '(x, y) = %d, %d' % (x, y)
    #print ''
    d_age = np.diff(age)
    add = np.array([1], dtype='uint16')
    d_age = np.insert(d_age, 0, add)
    increase_rate = increase[0:, x, y]
    increase_rate /= d_age
    daily_increase = np.repeat(increase_rate, d_age)
    daily_increase[np.isnan(daily_increase)] = 0.
    di_sum = np.sum(daily_increase)
    daily_increase /= di_sum
    daily_increase[daily_increase==0.] = np.nan
    stop = daily_increase.size
    d18O_addition = daily_increase * blood_hist[:stop]
    d18O_addition[np.isnan(d18O_addition)] = 0.

    d18O_total = np.sum(d18O_addition)

    return (d18O_total, daily_increase, d18O_addition)

def smoothListGaussian(list,degree=20):  

     window=degree*2-1  
     weight=np.array([1.0]*window)  
     weightGauss=[]  
     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  
     weight=np.array(weightGauss)*weight  
     smoothed=[0.0]*((list.size)-window)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)
     return smoothed

def imresize1(x, shape, method=Image.BILINEAR):
    assert len(x.shape) == 2
    
    im = Image.fromarray(x)
    im_resized = im.resize(shape, method)
    
    x_resized = np.array(im_resized.getdata()).reshape(shape[::-1]).T
    
    return x_resized

def z_calc(x_resized):

    # Below follows the actual isotope data from a real tooth
    data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 10.38, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 13.11, 12.70, 12.69, 12.23, 12.56, 11.53, 12.82, 12.36, 12.51, 10.69, 11.33, 13.33, 13.12, 13.21, 13.07, 13.76, 12.90, 14.63, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 12.31, 12.44, 12.12, 10.84, 12.85, 12.90, 13.13, 13.74, 13.18, 11.91, 12.53, 13.10, 12.28, 12.92, 10.95, 12.83, 13.20, 13.25, 12.10, 11.95, 12.08, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 10.19, 11.24, 10.55, 11.33, 12.09, 12.56, 13.71, 12.03, 10.78, 12.75, 12.67, 12.50, 12.48, 12.50, 11.96, 12.21, 12.28, 9.88, 11.85, 12.44, 11.07, 11.18, 10.68, 11.42, 12.39, 10.08])
    # Now the data is transformed.
    data_mean = np.mean(data)
    data = np.reshape(data, (5,34))
    data = np.flipud(data)
    dataT = data.T
    data[data==0] = np.nan

    #count data elements per column
    data_x_ct = []
    for i in dataT:
        data_x_ct.append(5. - np.count_nonzero(i))
    print data_x_ct

    #x_resized = x_resized + 100.
    idx = np.isnan(x_resized)
    x_resized[idx] = 0
    model_mean = np.mean(x_resized)

    factor = data_mean / model_mean
    #x_resized = factor*x_resized #!!!!!!!!!!!!!!!!!!!!!!!!
    two = np.mean(x_resized)
    x_resized[x_resized==0.] = np.nan    

    x_resized_small = np.delete(x_resized, np.s_[0:7], 1)
    data_small = np.delete(data, np.s_[0:7], 1)
    compare_s = np.absolute(data_small - x_resized_small)
    idx = np.isnan(compare_s)
    compare_s[idx] = 0.
    compare_s2 = np.square(compare_s)
    z_s_raw = np.sum(compare_s2)
    z_s = z_s_raw**.5

    compare = np.absolute(data - x_resized)

    idx = np.isnan(compare)
    compare[idx] = 0.
    compare2 = np.square(compare)
    z_raw = np.sum(compare2)
    z = z_raw**.5

    compare[compare==0.] = np.nan
    
    return (data, compare, z, x_resized, data_mean, model_mean, factor, two, z_s)

def main():

    f = h5py.File('simple_fit2.h5', 'r') #read in file
    dset1 = f['/age']
    age = dset1[:]
    dset2 = f['/img_mon']
    monotonic = dset2[:]
    dset3 = f['/img_mon_diff']
    increase = dset3[:]
    f.close()
    
    blood_hist, water_history, feed_history, air_history = calc_blood_hist()
 
    # increase.shape is a tuple containing
    # (# of time samples, # of x samples, # of y samples),
    # so we need elements 1 and 2 of the tuple (remember that there's also
    # a "zeroeth" element in the tuple).
    map_x, map_y = increase.shape[1:]

    print increase.shape
    print increase.shape[1:]

    return 0
    
    # Empty array of 8-Byte floats (pretty standard), and correct shape
    d18O_map = np.empty((map_x, map_y), dtype='f8')

    #d18O_total = d18O_pixel(age, increase, blood_hist, x, y)
    #print d_age, increase_x, daily_increase , d18O_addition, d18O_total

    
    # Loop over x-coordinate
    for x in xrange(map_x):
        # Loop over y-coordinate
        for y in xrange(map_y):
            d18O_total, daily_increase, d18O_addition = d18O_pixel(age, increase, blood_hist, x, y)
            if d18O_total == 0:
                d18O_total = np.nan
                
            # Store the total d18O in this pixel to the map
            d18O_map[x, y] = d18O_total

    d18O_map = d18O_map.T

    d18O_map = np.delete(d18O_map, np.s_[185:-1], 0)
    d18O_map = np.delete(d18O_map, np.s_[0:8], 0)

    shape = (34, 5)
    
    x_resized = imresize1(d18O_map, shape, method=Image.BILINEAR)
    x_resized = x_resized.T

    data, compare, z, x_resized, data_mean, model_mean, factor, two, z_s = z_calc(x_resized)
    
    d18O_total, daily_increase, d18O_addition = d18O_pixel(age, increase, blood_hist, 80, 15)

    print 'model mean =', model_mean
    print 'data mean =', data_mean
    print 'factor (data / model) =', factor
    print 'z score =', z
    print '2nd mean =', two
    

    
    # Show the map with matplotlib
    fig = plt.figure(dpi=300)#figsize=(6, 3), dpi=300, facecolor='w', edgecolor='k')

    #ax1 = fig.add_subplot(2,2,1)
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4)
    ax1.set_title('Modeled high-res d18O tooth map')
    cimg1 = ax1.imshow(d18O_map, origin='lower', aspect='auto', interpolation='none')
    cax1 = fig.colorbar(cimg1)
    
    #ax2 = fig.add_subplot(2,2,2)
    ax2 = plt.subplot2grid((4,4), (1,0), colspan=4)
    ax2.set_title('Modeled sample-res d18O tooth map')
    cimg2 = ax2.imshow(x_resized, origin='lower', aspect='auto', interpolation='none')
    cax2 = fig.colorbar(cimg2)

    #ax3 = fig.add_subplot(2,2,3)
    ax3 = plt.subplot2grid((4,4), (2,0), colspan=4)
    ax3.set_title('Data: sampled tooth d18O')
    cimg3 = ax3.imshow(data, origin='lower', aspect='auto', interpolation='none')
    cax3 = fig.colorbar(cimg3)

    #ax4 = fig.add_subplot(2,2,4)
    ax4 = plt.subplot2grid((4,4), (3,0), colspan=4)
    ax4.set_title('Magnitude divergence modeled - measured')
    cimg4 = ax4.imshow(compare, origin='lower', aspect='auto', interpolation='none')
    cax4 = fig.colorbar(cimg4)

    plt.tight_layout()
    #fig.savefig('complex003*, z = %.3f b.png' % z_s)
    
    return 0


if __name__ == '__main__':
    main()


