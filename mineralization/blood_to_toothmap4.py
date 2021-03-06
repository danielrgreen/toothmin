#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  blood_to_toothmap.py
#  
#  Copyright 2013 Daniel Green, Greg Green
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

def calculate_means(pct_min_samples):

    mean_pct = np.mean(pct_min_samples, axis=1)
    low_pct = np.percentile(pct_min_samples, 4, axis=1)
    high_pct = np.percentile(pct_min_samples, 94, axis=1)

    return mean_pct, low_pct, high_pct

def d18O_pixel(age, increase, blood_hist, x, y):

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


def blood_pixel_mnzt(mean_pct, age, blood_hist):
    '''
    
    '''
    
    n_days = blood_hist.size
    n_tmp = max(age[-1], n_days) # should delete this? yes for teeth, no for experiment
    mnzt_rate = np.zeros(n_tmp, dtype='f8') # will be same as increase rate
    #d_age = np.diff(age)
    #add = np.array([1], dtype='uint16')
    #d_age = np.insert(d_age, 0, add)

    mean_pct[np.isnan(mean_pct)] = 0.

    #for p in pct_min_samples:
        #pct_min_samples = np.hstack([0., pct_min_samples])
    
    for k, (a1, a2) in enumerate(zip(age[:-1], age[1:])):
        rate = (mean_pct[k+1] - mean_pct[k]) / (a2 - a1)
        
        for a in xrange(a1, a2):
            mnzt_rate[a] = rate
        #mnzt_rate[:, a1:a2] = rate
            
    mnzt_rate = mnzt_rate[:n_days] # mnzt_rate is MCMC samples x total days in size

    print mnzt_rate.shape
    
    # Calculate isotope ratios per pixel
    
    tot_isotope = (  np.sum(blood_hist * mnzt_rate, axis=0)
                   + blood_hist[0] * mean_pct[0] )
    tot_mnzt = np.sum(mnzt_rate, axis=0) + mean_pct[0]

    #tmp_ret = np.empty(3, dtype='f8')
    #tmp_ret[0] = np.sum(mnzt_rate[1, :25])
    #tmp_ret[1] = np.sum(mnzt_rate[1, :50])
    #tmp_ret[2] = np.sum(mnzt_rate[1, :100])
    
    return tot_isotope / tot_mnzt #tmp_ret

    
def mnzt_all_pix(mean_pct, age_mask, ages, blood_hist):
    '''

    '''

    n_pix = mean_pct.shape[0]
    mnzt_pct = np.empty(n_pix, dtype='f8')

    for n in xrange(n_pix):
        samples = blood_pixel_mnzt(mean_pct[n],
                                   ages[age_mask[n]],
                                   blood_hist)
        #mnzt_pct[:, n] = np.percentile(samples, [5., 50., 95.])
    
    return samples


def main():

    # read in h5py file and its datasets
    
    f = h5py.File('final.h5', 'r') 
    dset1 = f['/age_mask']
    age_mask = dset1[:].astype(np.bool)
    dset2 = f['/locations']
    locations = dset2[:]
    dset3 = f['/pct_min_samples']
    pct_min_samples = dset3[:]
    dset4 = f['/ages']
    ages = dset4[:]
    f.close()

    # print dataset shapes
    
    print 'age_mask shape', age_mask.shape
    print 'locations', locations.shape
    print 'pct_min_samples shape', pct_min_samples.shape

    # calculate low, mean and high values for samples

    mean_pct, low_pct, high_pct = calculate_means(pct_min_samples)

    # reshape samples into tooth shaped grids

    #Nx, Ny = np.max(locations, axis=0) + 1
    #print Nx, Ny, (Nx * Ny)
    #n_pix = mean_pct.shape[0]
    #mean_pct = np.reshape(mean_pct, (Nx, Ny, 45))

    print mean_pct.shape

    

    age_expanded= np.einsum('ij,j->ij', age_mask, ages)
    
    Nx, Ny = np.max(locations, axis=0) + 1
    n_pix = locations.shape[0]

    '''
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    for i in xrange(11000, 11050):
        print locations[i]
        y = np.median(pct_min_samples[i, :, :], axis=0)
        idx = np.isfinite(y)
        x = ages[idx]
        y = y[idx]
        ax.plot(x, y)

    plt.show()
    '''
    
    '''
    fig = plt.figure()
    
    for i,t in enumerate(ages[:16]):
        print i
        
        ax = fig.add_subplot(4, 4, i+1)

        img = np.empty((Nx, Ny), dtype='f8')
        img[:,:] = np.nan

        
        t_idx = np.sum(age_expanded < t, axis=1) - 1
        idx = (t_idx >= 0)
        
        img[locations[idx,0], locations[idx,1]] = pct_min_samples[idx, 0, t_idx[idx]]

        vmin = np.min(img[np.isfinite(img)])
        vmax = np.max(img[np.isfinite(img)])

        print img.shape
        
        cimg = ax.imshow(img.T, aspect='auto', interpolation='nearest',
                                origin='lower', vmin=0., vmax=1.)

    cax = fig.colorbar(cimg)
    plt.show()
    
    '''
    blood_hist, water_history, feed_history, air_history = calc_blood_hist()
    
    #print blood_hist

    #blood_hist = np.zeros(100, dtype='f8')
    #blood_hist[50:51] = 1.
    
    mnzt_pct = mnzt_all_pix(mean_pct, age_mask, ages, blood_hist)

    print mnzt_pct.shape
    
    img = np.empty((Nx, Ny), dtype='f8')
    img[:] = np.nan

    for n in xrange(n_pix):
        x, y = locations[n]
        img[x, y] = mnzt_pct[n]

    vmin = np.min(img[np.isfinite(img)])
    vmax = np.max(img[np.isfinite(img)])
    print vmin, vmax
    
    fig = plt.figure()

    ax = fig.add_subplot(3, 1, 1)
    cimg = ax.imshow(img[0].T, aspect='auto', interpolation='nearest', origin='lower',
                        vmin=vmin, vmax=vmax)
    
    ax = fig.add_subplot(3, 1, 2)
    cimg = ax.imshow(img[1].T, aspect='auto', interpolation='nearest', origin='lower',
                        vmin=vmin, vmax=vmax)
    
    ax = fig.add_subplot(3, 1, 3)
    cimg = ax.imshow(img[2].T, aspect='auto', interpolation='nearest', origin='lower',
                        vmin=vmin, vmax=vmax)

    # Show density in each pixel
    img = np.empty((3, Nx, Ny), dtype='f8')
    img[:] = np.nan

    for n in xrange(n_pix):
        x, y = locations[n]
        img[:, x, y] = np.median(pct_min_samples[n, :, :3], axis=0)

    fig = plt.figure()

    for n in xrange(3):
        ax = fig.add_subplot(3, 1, n+1)
        ax.imshow(img[n].T, aspect='auto', interpolation='nearest', origin='lower',
                            vmin=0., vmax=1.)

    cax = fig.colorbar(cimg)
    plt.show()


    return 0

if __name__ == '__main__':
    main()



    
