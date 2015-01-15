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
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from PIL import Image

# DECLARE SHAPE OF REAL ISOTOPE DATA FROM TOOTH
iso_shape = (34, 5)

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

# Taken from small_blood_to_toothmap.py
def imresize1(x, iso_shape, method=Image.BILINEAR):
    '''
    '''
    assert len(x.shape) == 2
    
    im = Image.fromarray(x)
    im_resized = im.resize(iso_shape, method)
    
    x_resized = np.array(im_resized.getdata()).reshape(iso_shape[::-1]).T
    
    return x_resized

# Taken from complex_image_resize.py
def count_number(iso_data):
    '''
    '''
    iso_data = np.reshape(iso_data, (iso_shape[1],iso_shape[0]))
    iso_data = iso_data.T
    iso_data = np.fliplr(iso_data)
    iso_data_x_ct = iso_shape[1] - np.sum(np.isnan(iso_data), axis=1)

    return (iso_data, iso_data_x_ct)

# Taken from complex_image_resize.py
def resize(model_row, new_size):
    '''
    '''
    model_row = model_row[np.isfinite(model_row)]
    msize = model_row.size
    required = abs(iso_shape[1] - new_size)
    transform = np.repeat(model_row, new_size).reshape(new_size, msize)
    new_row = np.einsum('ij->i', transform) / msize
    if required > 0:
        new_row = np.append(new_row, np.zeros(required)) # ADDING ZEROS, COULD CAUSE PROBLEMS
    new_row[new_row==0.] = np.nan
    return new_row

# Taken from complex_image_resize.py
def complex_resize(model, iso_data_x_ct):
    '''
    '''
    model = np.reshape(model, (iso_shape[1], iso_shape[0]))
    model = model.T
    model = np.flipud(model)
    fill = []

    for ct, row in zip(iso_data_x_ct, model):
        fill.append(resize(row, ct))

    model_resized = np.array(fill)
    return model_resized

def compare1(remodeled, reduced, iso_data, iso_data_reduced):
    '''
    Inputs
    x_resized:    2D array modeled isotope output, scaled to sampling size
    reduced:           1D vector, collapsed version of x_resized
    iso_data:          2D array of sampled d18O from actual tooth
    iso_data_reduced:  1D vector collapsed verion of iso_data
    Outputs
    likelihood_2D:     Numerical score likelihood
    likelihood_1D:     Numerical score likelihood
    '''
    compare_vector = []
    for i,j in zip(iso_data_reduced, reduced):
        compare_vector.append(i-j)
    compare_vector = np.array(compare_vector)**2
    compare_vector /= .8
    compare_vector = np.ma.masked_array(compare_vector, np.isnan(compare_vector))
    likelihood_1D = np.sum(compare_vector)

    compare_array = []
    for i,j in zip(iso_data, remodeled):
        compare_array.append(i-j)
    compare_array = np.array(compare_array)**2
    compare_array /= .8
    compare_array = np.ma.masked_array(compare_array, np.isnan(compare_array))
    likelihood_2D = np.sum(compare_array)
    
    return likelihood_1D, likelihood_2D

# Taken from small_blood_to_toothmap.py
def z_calc(x_resized):
    '''
    '''
    # Below follows the actual isotope data from a real tooth
    iso_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 10.38, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 13.11, 12.70, 12.69, 12.23, 12.56, 11.53, 12.82, 12.36, 12.51, 10.69, 11.33, 13.33, 13.12, 13.21, 13.07, 13.76, 12.90, 14.63, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 12.31, 12.44, 12.12, 10.84, 12.85, 12.90, 13.13, 13.74, 13.18, 11.91, 12.53, 13.10, 12.28, 12.92, 10.95, 12.83, 13.20, 13.25, 12.10, 11.95, 12.08, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 10.19, 11.24, 10.55, 11.33, 12.09, 12.56, 13.71, 12.03, 10.78, 12.75, 12.67, 12.50, 12.48, 12.50, 11.96, 12.21, 12.28, 9.88, 11.85, 12.44, 11.07, 11.18, 10.68, 11.42, 12.39, 10.08])
    # Now the data is transformed.
    iso_data_mean = np.mean(iso_data)
    data = np.reshape(data, iso_shape)
    data = np.flipud(data)
    dataT = data.T
    data[data==0] = np.nan

    #count data elements per column
    iso_data_x_ct = []
    for i in dataT:
        iso_data_x_ct.append(5. - np.count_nonzero(i))

    idx = np.isnan(x_resized)
    x_resized[idx] = 0
    model_mean = np.mean(x_resized)

    factor = data_mean / model_mean
    two = np.mean(x_resized)
    x_resized[x_resized==0.] = np.nan    

    x_resized_small = np.delete(x_resized, np.s_[0:7], 1)
    iso_data_small = np.delete(iso_data, np.s_[0:7], 1)
    compare_s = np.absolute(iso_data_small - x_resized_small)
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
    
    return (iso_data, compare, z, x_resized, data_mean, model_mean, factor, two, z_s)


def main():

    #f = h5py.File('final_equalsize_dec2014.h5', 'r') #read in file
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

    age_expanded = np.einsum('ij,j->ij', age_mask, ages)
    
    Nx, Ny = np.max(locations, axis=0) + 1
    n_pix = locations.shape[0]

    blood_hist, water_history, feed_history, air_history = calc_blood_hist()
        
    mnzt_pct = mnzt_all_pix(pct_min_samples, age_mask, ages, blood_hist)
    
    img = np.empty((Nx, Ny), dtype='f8')
    img[:] = np.nan

    for n in xrange(n_pix):
        x, y = locations[n]
        img[x, y] = mnzt_pct[1, n]

    img = img.T
    #d18O_map = np.delete(d18O_map, np.s_[185:-1], 0)
    #img = np.delete(img, np.s_[0:8], 0)
    img = img[8:, :] + 18.
    img = gaussian_filter(img, (2,2), mode='nearest')
    
    x_resized = imresize1(img, iso_shape, method=Image.BILINEAR)

    #iso_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 10.38, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 13.11, 12.70, 12.69, 12.23, 12.56, 11.53, 12.82, 12.36, 12.51, 10.69, 11.33, 13.33, 13.12, 13.21, 13.07, 13.76, 12.90, 14.63, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 12.31, 12.44, 12.12, 10.84, 12.85, 12.90, 13.13, 13.74, 13.18, 11.91, 12.53, 13.10, 12.28, 12.92, 10.95, 12.83, 13.20, 13.25, 12.10, 11.95, 12.08, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 10.19, 11.24, 10.55, 11.33, 12.09, 12.56, 13.71, 12.03, 10.78, 12.75, 12.67, 12.50, 12.48, 12.50, 11.96, 12.21, 12.28, 9.88, 11.85, 12.44, 11.07, 11.18, 10.68, 11.42, 12.39, 10.08]) #old data
    iso_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 10.38, 13.29, 13.36, 12.85, 13.15, 12.35, 13.31, 12.89, 12.92, 13.35, 13.12, 13.21, 13.08, 13.30, 13.67, 12.45, 11.82, 11.32, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 13.00, 13.07, 13.11, 12.98, 13.20, 13.37, 13.24, 12.26, 12.61, 13.19, 12.50, 13.01, 12.75, 13.08, 12.97, 13.15, 12.52, 12.33, 12.08, 11.87, 11.07, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 12.86, 12.69, 12.95, 12.66, 12.89, 13.52, 12.47, 12.91, 12.95, 12.87, 12.41, 12.72, 12.82, 12.38, 12.44, 12.89, 11.03, 12.63, 12.99, 13.13, 12.43, 7.35, 12.10, 11.42, 12.39, 10.08])
    #iso_data = iso_data.reshape(iso_shape[0], iso_shape[1])
    iso_data[iso_data==0.] = np.nan
    iso_data, iso_data_x_ct = count_number(iso_data)
    temp_x_r = np.fliplr(x_resized.T)
    model_resized = complex_resize(temp_x_r.flatten(), iso_data_x_ct)
    remodeled = np.array(model_resized)
    iso_data = iso_data
    x_resized = x_resized.T
    iso_data = iso_data.T

    iso_data_reduced1 = np.ma.masked_array(iso_data, np.isnan(iso_data))
    iso_data_reduced2 = np.mean(iso_data_reduced1, axis=0)
    iso_data_reduced = iso_data_reduced2.filled(np.nan)

    remodeled = remodeled.T
    reduced1 = np.ma.masked_array(remodeled, np.isnan(remodeled))
    reduced2 = np.mean(reduced1, axis=0)
    reduced = reduced2.filled(np.nan)

    likelihood_1D, likelihood_2D = compare1(x_resized, reduced, iso_data, iso_data_reduced)
    
    print 'compare vector = ', likelihood_1D
    print 'compare array = ', likelihood_2D
    
    fig = plt.figure()

    ax1 = fig.add_subplot(6, 1, 1)
    cimg1 = ax1.imshow(img, aspect='equal', interpolation='nearest', origin='lower',
                        vmin=10, vmax=14, cmap=plt.get_cmap('bwr'))
    cax1 = fig.colorbar(cimg1)

    ax2 = fig.add_subplot(6, 1, 2)
    cimg2 = ax2.imshow(x_resized, aspect='equal', interpolation='nearest', origin='lower',
                        vmin=10, vmax=14, cmap=plt.get_cmap('bwr'))
    cax2 = fig.colorbar(cimg2)

    ax3 = fig.add_subplot(6, 1, 3)
    cimg3 = ax3.imshow(iso_data, aspect='equal', interpolation='nearest', origin='lower',
                        vmin=10, vmax=14, cmap=plt.get_cmap('bwr'))
    cax3 = fig.colorbar(cimg3)

    ax4 = fig.add_subplot(6, 1, 4)
    iso_data_reduced.shape = (iso_data_reduced.size, 1)
    cimg4 = ax4.imshow(iso_data_reduced.T, aspect='equal', interpolation='nearest', origin='lower',
                        vmin=10, vmax=14, cmap=plt.get_cmap('bwr'))
    cax4 = fig.colorbar(cimg4)

    ax5 = fig.add_subplot(6, 1, 5)
    cimg5 = ax5.imshow(remodeled, aspect='equal', interpolation='nearest', origin='lower',
                        vmin=10, vmax=14, cmap=plt.get_cmap('bwr'))
    cax5 = fig.colorbar(cimg5)

    ax6 = fig.add_subplot(6, 1, 6)
    reduced.shape = (reduced.size, 1)
    cimg6 = ax6.imshow(reduced.T, aspect='equal', interpolation='nearest', origin='lower',
                        vmin=10, vmax=14, cmap=plt.get_cmap('bwr'))
    cax6 = fig.colorbar(cimg6)

    fig.subplots_adjust(hspace=.5)
    fig.savefig('jan15_blood2toothmap6.pdf', dpi=300, figsize=8, format='pdf', edgecolor='none')
    plt.show()

    return 0

if __name__ == '__main__':
    main()



    
