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
import matplotlib.pyplot as plt
import h5py
from bloodhist import calc_blood_hist

def d18O_pixel(age, increase, blood_hist, x, y):

    d_age = np.diff(age)
    increase_x = increase[1:, x, y]
    increase_x = increase_x / d_age
    daily_increase = np.repeat(increase_x, d_age)
    stop = daily_increase.size
    d18O_addition = daily_increase * blood_hist[:stop]   
    idx = np.isnan(d18O_addition)
    d18O_addition[idx] = 0.

    d18O_total = np.sum(d18O_addition)

    return d18O_total

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
    
    # Empty array of 8-Byte floats (pretty standard), and correct shape
    d18O_map = np.empty((map_x, map_y), dtype='f8')

    #d18O_total = d18O_pixel(age, increase, blood_hist, x, y)
    #print d_age, increase_x, daily_increase , d18O_addition, d18O_total

    # Loop over x-coordinate
    for x in xrange(map_x):
        # Loop over y-coordinate
        for y in xrange(map_y):
            d18O_total = d18O_pixel(age, increase, blood_hist, x, y)
                       
            # Store the total d18O in this pixel to the map
            d18O_map[x, y] = d18O_total

    '''
    shape_map = d18O_map.shape
    print 'shape_map', shape_map
    flat_map = d18O_map.ravel()
    print 'flat_map shape', flat_map.shape
    list = flat_map
    print 'list shape', list.shape
    smoothed = smoothListGaussian(list, degree=20)
    print 'smoothed type1', type(smoothed)
    smoothed = np.asarray(smoothed)
    minus = smoothed.shape
    print 'smoothed shape2', smoothed.shape
    added1 = flat_map.shape[0] - minus[0]
    print 'added', added1
    added2 = np.zeros(added1)
    print 'added2', added2
    smoothed = np.append(smoothed, added2)
    print 'smoothed shape3', smoothed.shape
    d18O_map = smoothed.reshape(shape_map)
    '''
    
    # Show the map with matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    cimg = ax.imshow(d18O_map.T, origin='lower', aspect='auto', interpolation='none')
    cax = fig.colorbar(cimg)
    
    plt.show()
    
    return 0

if __name__ == '__main__':
    main()


