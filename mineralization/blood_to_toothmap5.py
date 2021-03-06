#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  blood_to_toothmap5.py effort to integrate in
#  effort to integrate in 'small_blood_to_toothmap3.py'
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


def blood_pixel_mnzt(pct_min_samples, age, blood_hist):
    '''
    
    '''
    
    n_days = blood_hist.size
    n_samples = pct_min_samples.shape[0]
    
    n_tmp = max(age[-1], n_days)
    mnzt_rate = np.zeros((n_samples, n_tmp+1), dtype='f8')

    # turning pct_min_samples NaNs into 0s
    pct_min_samples[np.isnan(pct_min_samples)] = 0.

    # inserting a zero before the first pct_min_sample time value
    add_zeros = np.zeros((pct_min_samples.shape[0], pct_min_samples.shape[1]+1))
    add_zeros[:, :-1] = pct_min_samples
    pct_min_samples = add_zeros
    
    for k, (a1, a2) in enumerate(zip(age[:-1], age[1:])):
        rate = (pct_min_samples[:, k+1] - pct_min_samples[:, k]) / (a2 - a1)
        
        for a in xrange(a1, a2):
            mnzt_rate[:, a] = rate

    mnzt_rate = mnzt_rate[:, :n_days] # same as daily_increase, shape = nsamples x days
    di_sum = np.sum(mnzt_rate, axis=1) # shape = nsamples

    # first method of calculating isotope values per pixel   ###    
    #tot_isotope = (  np.sum(blood_hist * mnzt_rate, axis=1)
                   #+ blood_hist[0] * pct_min_samples[:, 0] )
    #tot_mnzt = np.sum(mnzt_rate, axis=1) + pct_min_samples[:, 0]
    #tot_isotope = tot_isotope / tot_mnzt

    # second method calculating isotope values per pixel   ###
    mnzt_rate = mnzt_rate / di_sum[:, None]
    mnzt_rate[mnzt_rate==0.] = np.nan
    d18O_addition = mnzt_rate * blood_hist[None, :]
    d18O_addition[np.isnan(d18O_addition)] = 0.
    tot_isotope = np.sum(d18O_addition, axis=1)

    #tmp_ret = np.empty(3, dtype='f8')
    #tmp_ret[0] = np.sum(mnzt_rate[1, :25])
    #tmp_ret[1] = np.sum(mnzt_rate[1, :50])
    #tmp_ret[2] = np.sum(mnzt_rate[1, :100])
    
    return tot_isotope

    
def mnzt_all_pix(pct_min_samples, age_mask, ages, blood_hist):
    '''

    '''

    n_pix = pct_min_samples.shape[0]
    mnzt_pct = np.empty((3, n_pix), dtype='f8')

    for n in xrange(n_pix):
        samples = blood_pixel_mnzt(pct_min_samples[n],
                                   ages[age_mask[n]],
                                   blood_hist)
        mnzt_pct[:, n] = np.percentile(samples, [5., 50., 95.])
    
    return mnzt_pct


def main():

    f = h5py.File('final.h5', 'r') #read in file
    for name in f:
        print name
    dset1 = f['/age_mask']
    age_mask = dset1[:].astype(np.bool)
    dset2 = f['/locations']
    locations = dset2[:]
    dset3 = f['/pct_min_samples']
    pct_min_samples = dset3[:]
    dset4 = f['/ages']
    ages = dset4[:]
    f.close()
    print 'age_mask shape', age_mask.shape
    print 'locations', locations.shape
    print 'pct_min_samples shape', pct_min_samples.shape

    age_expanded = np.einsum('ij,j->ij', age_mask, ages)
    
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

    print ages
    
    mnzt_pct = mnzt_all_pix(pct_min_samples, age_mask, ages, blood_hist)
    
    img = np.empty((3, Nx, Ny), dtype='f8')
    img[:] = np.nan

    for n in xrange(n_pix):
        x, y = locations[n]
        img[:, x, y] = mnzt_pct[:, n]

    vmin = np.min(img[np.isfinite(img)])
    vmax = np.max(img[np.isfinite(img)])
    print vmin, vmax
    
    fig = plt.figure()
    '''
    ax = fig.add_subplot(3, 1, 1)
    cimg1 = ax.imshow(img[0].T, aspect='auto', interpolation='nearest', origin='lower',
                        vmin=vmin, vmax=vmax)
    cax1 = fig.colorbar(cimg1)
    '''
    ax = fig.add_subplot(1, 1, 1)
    cimg2 = ax.imshow(img[1].T, aspect='equal', interpolation='nearest', origin='lower',
                        vmin=.8, vmax=vmax)
    cax2 = fig.colorbar(cimg2)
    '''
    ax = fig.add_subplot(3, 1, 3)
    cimg3 = ax.imshow(img[2].T, aspect='auto', interpolation='nearest', origin='lower',
                        vmin=vmin, vmax=vmax)
    cax3 = fig.colorbar(cimg3)
    '''
    fig.savefig('fig_20140815b.png', dpi=500, figsize=8, edgecolor='none')
    plt.show()

    '''
    # Show density in each pixel
    img = np.empty((3, Nx, Ny), dtype='f8')
    img[:] = np.nan

    for n in xrange(n_pix):
        x, y = locations[n]
        img[:, x, y] = np.median(pct_min_samples[n, :, :3], axis=0)
    
    fig = plt.figure()

    for n in xrange(3):
        ax = fig.add_subplot(3, 1, n+1)
        cimg4 = ax.imshow(img[n].T, aspect='auto', interpolation='nearest', origin='lower',
                            vmin=0., vmax=1.)
        cax4 = fig.colorbar(cimg4)

    plt.show()
    '''

    return 0

if __name__ == '__main__':
    main()



    
