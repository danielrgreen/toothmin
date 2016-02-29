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

trial_iteration = [0.]
score_prior_counter = []

class ToothModel:
    def __init__(self, fname=None):
        if fname == None:
            return
        
        self._load_tooth_model(fname)
        self._interp_missing_ages()

    def _load_tooth_model(self, fname):
        f = h5py.File(fname, 'r')
        dset1 = f['/age_mask']
        self.age_mask = dset1[:].astype(np.bool)
        dset2 = f['/locations']
        self.locations = dset2[:]
        dset3 = f['/pct_min_samples']
        self.pct_min_samples = dset3[:]
        dset4 = f['/ages']
        self.ages = dset4[:]
        f.close()
        
        self.age_expanded = np.einsum('ij,j->ij', self.age_mask, self.ages)
        
        self.n_pix, self.n_samples, self.n_ages = self.pct_min_samples.shape
    
    def _interp_missing_ages(self):
        '''
        Calculate mineralization percent on days that weren't sampled.
        '''
        pct_min_interp = np.empty((self.n_pix, self.n_samples, self.n_ages), dtype='f4')
        pct_min_interp[:] = np.nan
        
        for n in xrange(self.n_pix):
            idx = self.age_mask[n]
            samples = self.pct_min_samples[n, :, :np.sum(idx)]
            samples = np.swapaxes(samples, 0, 1)
            pct_min_interp[n, :, idx] = samples[:, :]
            
            x = self.ages[~idx]
            xp = self.ages[idx]

            for k in xrange(self.n_samples):
                pct_min_interp[n, k, ~idx] = np.interp(x, xp, samples[:,k], left=0.)

        self.pct_min_interp = pct_min_interp
    
    def _gen_rand_hist(self, pix_val):
        '''
        Returns a random history for each pixel.

        Output shape: (# of pixels, # of ages)
        '''
        
        idx0 = np.arange(self.n_pix)
        idx1 = np.random.randint(self.n_samples, size=self.n_pix)
        pix_val_rand = pix_val[idx0, idx1, :]

        return pix_val_rand

    def _pix2img(self, pix_val, mode='sample', interp=False):

        if mode == 'sample':
            pix_val = self._gen_rand_hist(pix_val)
        else:
            pix_val = np.percentile(pix_val, mode, axis=1)

        n_x = np.max(self.locations[:,0]) + 1
        n_y = np.max(self.locations[:,1]) + 1
        img = np.empty((n_x, n_y, pix_val.shape[1]), dtype='f8')
        img[:] = np.nan

        idx0 = self.locations[:,0]
        idx1 = self.locations[:,1]
        img[idx0, idx1, :] = pix_val[:,:]

        if interp:
            img_interp = interp1d(self.ages, img)
            return img_interp

        return img

    def _pix2imgu(self, isotope):
        print 'compressing to 2D model...'
        isotope = np.mean(isotope, axis=1)
        print isotope.shape
        print 'making isomap...'
        Nx, Ny = np.max(self.locations, axis=0) + 1
        n_pix = self.locations.shape[0]
        im = np.empty((Nx, Ny), dtype='f8')
        im[:] = np.nan
        im[self.locations[:,0], self.locations[:,1]] = isotope[:,-1]
        #for n in xrange(n_pix):
        #    x, y = self.locations[n]
        #    im[x, y, :] = isotope[n, -1] + 18 # works after 3 days, not at 2nd, 1st or 0th day

        #fig = plt.figure(dpi=100)
        #ax = plt.subplot(1,1,1)
        #cimg = ax.imshow(im.T, aspect='auto', interpolation='nearest', origin='lower', cmap=plt.get_cmap('bwr'))
        #cax = fig.colorbar(cimg)
        #plt.show()

        return im

    def gen_mnzt_image(self, mode='sample', interp=False):
        '''
        Returns an image of the tooth at each sampled age.
        If interp is True, then returns an interpolating function
        of the image as a function of time (in days).

        If mode == 'sample', then draws a random profile for each
        pixel, from the set of stored profiles.
        
        If mode is an integer, then returns the given percentile
        of the mineral density.
        '''
        
        return self._pix2img(self.pct_min_interp, mode=mode, interp=interp)

    def _resize2(self, array, shape):
        old_shape = array.shape

        array = np.repeat(array, shape[0], axis=0)
        array = np.repeat(array, shape[1], axis=1)

        array = np.reshape(array, (old_shape[0]*shape[0]*shape[1], old_shape[1]))
        array = np.mean(np.ma.masked_array(array, np.isnan(array)), axis=1)

        array = np.reshape(array, (old_shape[0], shape[0]*shape[1]))
        array = np.mean(np.ma.masked_array(array, np.isnan(array)), axis=0)

        array = np.reshape(array, shape)

        return array

    def _resize_w_nans(self, array, shape):
        arr_nan = np.isnan(array).astype('f8')

        arr_sm = self._resize2(array, shape)
        arr_sm_nan = self._resize2(arr_nan, shape)
        arr_sm[arr_sm_nan > 0.5] = np.nan

        return arr_sm

    def _resize(self, array, shape):
        '''
        Takes an array and broadcasts it into a new shape, ignoring NaNs.
        '''

        # Calculate shapes
        oldshape = array.shape
        axis0 = np.repeat(array, shape[0], axis=0)
        axis1 = np.repeat(axis0, shape[1], axis=1)

        # Resize
        axis1ravel = np.ravel(axis1)
        axis1stack = np.reshape(axis1ravel, (shape[0] * shape[1] * oldshape[0], oldshape[1]))
        m_axis1stack = np.ma.masked_array(axis1stack, np.isnan(axis1stack))
        axis1mean = np.mean(m_axis1stack, axis=1)
        axis2stack = np.reshape(axis1mean, (oldshape[0] * shape[0], shape[1])).T
        axis2reshape = np.reshape(axis2stack, (shape[0]*shape[1], oldshape[0]))

        m_axis2reshape = np.ma.masked_array(axis2reshape, np.isnan(axis2reshape))
        axis2reshape_mean = np.mean(m_axis2reshape, axis=1).reshape(shape[1], shape[0])
        new_array = axis2reshape_mean.T
        # Add back in NaNs, threshold > 50% NaN
        nan_map = np.zeros(oldshape)
        nan_map[np.isnan(array)] = 1.
        sm_nan_map = imresize(nan_map, shape, interp='bilinear', mode='F')
        new_array[sm_nan_map >= 0.5] = np.nan

        return new_array
    
    def downsample_model(self, shape, n_samples):
        from scipy.misc import imresize
        print 'downsampling model...'
        # (x, y, sample, age)
        img_sm = np.empty((shape[0], shape[1], n_samples, self.n_ages), dtype='f8') ######## 0,1
        
        for n in xrange(n_samples):
            img = self.gen_mnzt_image(mode='sample')

            for t in xrange(self.n_ages):
                #img_sm[:,:,n,t] = imresize(img[:,:,t], shape)
                img_sm[:,:,n,t] = self._resize(img[:,:,t], (shape[0], shape[1]))
                #img_sm[:,:,n,t] = np.fliplr((self._resize(img[:,:,t], (shape[0], shape[1]))).T) ######## 0,1 #
                #print 'first', img_sm[:,:,n,t].shape
                #img_sm[:,:,n,t] = img_sm[:,:,n,t].T
                #print 'second', img_sm[:,:,n,t].shape

        img_sm.shape = (shape[0]*shape[1], n_samples, self.n_ages) ########## 0,1 # no effect

        locations = np.indices((shape[0], shape[1])) ######## 0,1 # shape now reversed
        locations.shape = (2, shape[0]*shape[1]) ######## 0,1 # no effect
        locations = np.swapaxes(locations, 0, 1) ######## 0,1 # no effect

        tmodel = ToothModel()
        tmodel.pct_min_interp = img_sm[:,:,:]
        tmodel.locations = locations[:,:]
        tmodel.ages = self.ages[:]
        tmodel.n_pix = shape[0] * shape[1] ########## 0,1 # no effect
        tmodel.n_ages = self.n_ages
        tmodel.n_samples = n_samples

        return tmodel

    def gen_isotope_image(self, blood_step, mode='sample'):
        idx_mask = np.isnan(self.pct_min_interp)
        pct_min_interp = np.ma.array(self.pct_min_interp, mask=idx_mask, fill_value=0.)
        pct_min_diff = diff_with_first(pct_min_interp[:,:,:].filled(), axis=2)

        n_days = blood_step.size
        pct_min_diff_days = np.empty((self.n_pix, self.n_samples, n_days), dtype='f8')
        pct_min_diff_days[:] = np.nan
        
        for k,(a1,a2) in enumerate(zip(self.ages[:-1], self.ages[1:])):
            if a1 > n_days:
                break

            dt = a2 - a1
            
            if a2 > n_days:
                a2 = n_days
            
            pct_min_diff_days[:,:,a1:a2] = (pct_min_diff[:,:,k] / dt)[:,:,None] # All checks out except 0th, 1st, 2nd val

        #print 'calculating cumulative mineral increase in tooth over time...' # takes 100 seconds
        pct_min_diff_days[np.isnan(pct_min_diff_days)] = 0.
        pct_min_days = np.cumsum(pct_min_diff_days, axis=2) # All checks out except 0th, 1st, 2nd val
        pct_min_days[pct_min_days==0.] = np.nan

        #print 'multiplying daily mineral additions by daily isotope ratios...' # takes 100 seconds
        isotope = np.cumsum(
            blood_step[None, None, :]
            * pct_min_diff_days,
            axis=2
        ) # Works at and after 3rd day. Something weird (all zero values?) happens before that.

        #print 'calculating isotope ratios in tooth for each day of growth...' # takes 60-100 seconds
        isotope /= pct_min_days

        if mode == 'sample':
            return self._pix2img(isotope)
        elif isinstance(mode, list):
            return [self._pix2img(isotope, mode=pct) for pct in mode]
        elif isinstance(mode, int):
            return [self._pix2img(isotope) for k in xrange(mode)]
        else:
            raise ValueError('mode not understood: {0}'.format(mode))

def diff_with_first(x, axis=0):
    y = np.swapaxes(x, 0, axis)

    z = np.empty(y.shape, dtype=y.dtype)
    z[0] = y[0]
    z[1:] = y[1:] - y[:-1]
    
    return np.swapaxes(z, 0, axis)

def interp_over_nans(x_data, y_data):
    '''
    Return a version of y_data, where NaN values have been replaced
    with linearly interpolated values.
    '''
    y = np.empty(y_data.shape, dtype=y_data.dtype)
    
    idx_nan = np.isnan(y_data)
    xp = x_data[~idx_nan]
    yp = y_data[~idx_nan]

    y[~idx_nan] = yp[:]
    y[idx_nan] = np.interp(x[idx_nan], xp, yp, left=0.)

    return y

def gen_mnzt_movie(tooth_model, outfname):
    #tooth_model = tooth_model.downsample(shape)
    #isotope_pct = tooth_model.isotope_pct(blood_step)
    #diff = isotope_pct - data
    #chisq = np.sum(diff**2)

    print 'generating mineralization movie...'

    img_interp = tooth_model.gen_mnzt_image(interp=True, mode='sample')

    ages = np.arange(tooth_model.ages[0], tooth_model.ages[-1]+1)

    img = img_interp(ages[-1])
    shape = img.shape[:2]
    
    img = np.empty((ages.size, shape[0], shape[1]), dtype='f8')

    for k,t in enumerate(ages):
        img[k] = img_interp(t)

    img = np.diff(img, axis=0)
    sigma_t = 4
    sigma_x, sigma_y = 0, 0
    img = gaussian_filter(img, (sigma_t,sigma_x,sigma_y), mode='nearest')
    
    idx = np.isfinite(img)
    vmax = np.percentile(img[idx], 99.8)
    #vmax = 1.

    fig = plt.figure(figsize=(6,3), dpi=100)
    ax = fig.add_subplot(1,1,1)
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off'
    )
    
    im = ax.imshow(
        np.zeros(shape[::-1], dtype='f8'),
        origin='lower',
        interpolation='nearest',
        vmin=0.,
        vmax=vmax
    )
    
    for k,t in enumerate(ages[:-1]):
        fn = '%s_k%04d.png' % (outfname, k)

        print 'Generating frame {0}...'.format(fn)

        #img = img_interp(t)
        im.set_data(img[k].T) #was(img[k].T)

        ax.set_title(r'$t = %d \ \mathrm{days}$' % t, fontsize=14)
        
        fig.savefig(fn, dpi=100)

def gen_isomap_movie(tooth_model, blood_step):

    print 'generating movie...'
    img = tooth_model.gen_isotope_image(blood_step, mode='sample')
    
    sigma_t = 0
    sigma_x, sigma_y = 0, 0
    img = gaussian_filter(img, (sigma_t,sigma_x,sigma_y), mode='nearest')
    
    #idx = np.isfinite(img)
    #vmax = np.percentile(img[idx], 99.5)
    vmax = 1.

    fig = plt.figure(figsize=(6,3), dpi=100)
    ax = fig.add_subplot(1,1,1)

    shape = img.shape[::2]

    im = ax.imshow(np.zeros(shape[::], dtype='f8'), origin='lower',
                                                interpolation='nearest',
                                                vmin=0., vmax=vmax)
    
    for k in xrange(img.shape[2]):
        im.set_data(img[:,:,k].T)
        print 'printing image k%04d.png' % k
        ax.set_title(r'$t = %d \ \mathrm{days}$' % k, fontsize=14)
        fig.savefig('jan01_k%04d.png' % k, dpi=100)

def count_number(iso_shape, iso_data):
    '''

    :param iso_shape:   shape of isotope data
    :param iso_data:    measured isotope values
    :return:            isotope data, number of isotope measurements per column
                        in isotope data formatted as string.
    '''
    iso_data = np.reshape(iso_data, (iso_shape[1],iso_shape[0]))
    iso_data = iso_data.T
    iso_data = np.fliplr(iso_data)
    iso_data_x_ct = iso_shape[1] - np.sum(np.isnan(iso_data), axis=1)

    return (iso_data, iso_data_x_ct)

def resize(iso_shape, model_row, new_size):
    '''

    :param iso_shape:   shape of original data as tuple in length times height
    :param model_row:   number of isotope values in model row
    :param new_size:    new number of isotope values into which will be fit model
                        row isotope ratios
    :return:            new model row with values interpolated such that model isotope
                        predictions are reshaped to fit number of measurements from
                        real data.
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

def complex_resize(iso_shape, model, iso_data_x_ct):
    '''

    :param iso_shape:       tuple, the shape of real isotope data in length by height
    :param model:           model isotope data
    :param iso_data_x_ct:   a list of measured isotope values per column in a real
                            tooth
    :return:                reshaped model data to fit size of real data
    '''
    model = np.reshape(model, (iso_shape[1], iso_shape[0]))
    model = model.T
    #model = np.flipud(model)
    fill = []

    for ct, row in zip(iso_data_x_ct, model):
        fill.append(resize(iso_shape, row, ct))

    model_resized = np.array(fill)
    return model_resized

def imresize1(x, iso_shape, method=Image.BILINEAR):
    '''

    :param x:           isotope image
    :param iso_shape:   shape of real isotope data
    :param method:      method of resize interpolation
    :return:            resized image with iso_shape dimentions
    '''
    #x = x[:,:,-1] # OR SUM, ALSO BROKEN BECAUSE SUM OR MEAN TAKES NANS
    assert len(x.shape) == 2
    
    im = Image.fromarray(x)
    im_resized = im.resize(iso_shape, method)
    
    x_resized = np.array(im_resized.getdata()).reshape(iso_shape[::-1]).T
    
    return x_resized

def import_iso_data(**kwargs):
    '''
    Takes isotope per mil values measured from a tooth as a string, and the
    shape of measured data, and returns the shape tuple in length x height,
    the reshaped data, and a list of the number of height isotope measurements
    per length coordinate.

    :param kwargs:      iso_shape is the shape of the measured isotope data
                        as a tuple in length x height
    :return:            the shape of isotope data as a tuple, reshaped isotope
                        data, and count of isotope values in a column (height)
                        per element in a row (length)
    '''

    # Felicitas (34,5)
    #iso_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 10.38, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 13.11, 12.70, 12.69, 12.23, 12.56, 11.53, 12.82, 12.36, 12.51, 10.69, 11.33, 13.33, 13.12, 13.21, 13.07, 13.76, 12.90, 14.63, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 12.31, 12.44, 12.12, 10.84, 12.85, 12.90, 13.13, 13.74, 13.18, 11.91, 12.53, 13.10, 12.28, 12.92, 10.95, 12.83, 13.20, 13.25, 12.10, 11.95, 12.08, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 10.19, 11.24, 10.55, 11.33, 12.09, 12.56, 13.71, 12.03, 10.78, 12.75, 12.67, 12.50, 12.48, 12.50, 11.96, 12.21, 12.28, 9.88, 11.85, 12.44, 11.07, 11.18, 10.68, 11.42, 12.39, 10.08]) #old data

    # 964 (21,4)
    #iso_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 10.38, 13.29, 13.36, 12.85, 13.15, 12.35, 13.31, 12.89, 12.92, 13.35, 13.12, 13.21, 13.08, 13.30, 13.67, 12.45, 11.82, 11.32, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 13.00, 13.07, 13.11, 12.98, 13.20, 13.37, 13.24, 12.26, 12.61, 13.19, 12.50, 13.01, 12.75, 13.08, 12.97, 13.15, 12.52, 12.33, 12.08, 11.87, 11.07, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 12.86, 12.69, 12.95, 12.66, 12.89, 13.52, 12.47, 12.91, 12.95, 12.87, 12.41, 12.72, 12.82, 12.38, 12.44, 12.89, 11.03, 12.63, 12.99, 13.13, 12.43, 7.35, 12.10, 11.42, 12.39, 10.08])

    # 962 (27,6)
    iso_shape = kwargs.get('iso_shape', (27, 6))
    iso_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 13.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.88, 13.61, 13.97, 13.69, 13.71, 13.49, 13.67, 0.00, 0.00, 0.00, 14.13, 14.55, 0.00, 0.00, 0.00, 0.00, 12.51, 11.53, 10.54, 10.72, 11.10, 10.93, 11.45, 10.99, 10.96, 11.74, 12.09, 12.88, 13.61, 13.97, 13.69, 13.71, 13.49, 13.46, 13.76, 14.12, 14.23, 14.37, 14.55, 0.00, 0.00, 0.00, 13.20, 12.51, 11.56, 10.54, 10.72, 10.42, 9.87, 10.39, 10.10, 10.91, 11.99, 11.82, 12.29, 12.54, 13.16, 13.33, 13.32, 13.44, 13.66, 13.63, 13.89, 13.88, 14.06, 14.35, 0.00, 0.00, 13.80, 13.20, 12.51, 11.56, 10.54, 10.13, 9.58, 9.45, 9.73, 9.89, 10.91, 10.88, 11.49, 12.41, 12.06, 11.96, 12.59, 13.00, 13.16, 13.27, 13.46, 13.36, 13.58, 13.75, 14.34, 14.94, 14.27, 13.80, 13.20, 12.51, 11.56, 11.07, 10.11, 9.61, 9.88, 9.47, 10.70, 10.91, 11.42, 12.33, 11.94, 11.66, 11.12, 11.17, 12.12, 12.96, 13.27, 13.30, 13.31, 13.49, 13.28, 13.90, 14.45])

    iso_data[iso_data==0.] = np.nan
    iso_data, iso_data_x_ct = count_number(iso_shape, iso_data)

    return iso_shape, iso_data, iso_data_x_ct

def load_iso_data(fname):
    '''
    Load a tooth isotope map from a CSV file.
    :param fname: The filename of the CSV file.
    :return:
      iso_data       The tooth isotope image
      iso_shape      Shape of the tooth image
      iso_data_x_ct  Number of nans in each column
    '''

    'data.csv'

    iso_data = np.loadtxt(fname, delimiter=',').T
    iso_data[iso_data==0.] = np.nan
    iso_data = iso_data[:,::-1]

    iso_data_x_ct = iso_data.shape[1] - np.sum(np.isnan(iso_data), axis=1)

    iso_shape = iso_data.shape
    #iso_data, iso_data_x_ct = count_number(iso_shape, iso_data)

    return iso_data, iso_shape, iso_data_x_ct

def wizard(array, shape):
    '''
    Takes an array and broadcasts it into a new shape, ignoring NaNs.
    '''

    # Calculate shapes
    oldshape = array.shape
    axis0 = np.repeat(array, shape[0], axis=0)
    axis1 = np.repeat(axis0, shape[1], axis=1)

    # Resize
    axis1ravel = np.ravel(axis1)
    axis1stack = np.reshape(axis1ravel, (shape[0] * shape[1] * oldshape[0], oldshape[1]))
    m_axis1stack = np.ma.masked_array(axis1stack, np.isnan(axis1stack))
    axis1mean = np.mean(m_axis1stack, axis=1)
    axis2stack = np.reshape(axis1mean, (oldshape[0] * shape[0], shape[1])).T

    axis2reshape = np.reshape(axis2stack, (shape[0]*shape[1], oldshape[0]))

    m_axis2reshape = np.ma.masked_array(axis2reshape, np.isnan(axis2reshape))
    axis2reshape_mean = np.mean(m_axis2reshape, axis=1).reshape(shape[1], shape[0])
    new_array = axis2reshape_mean.T
    # Add back in NaNs, threshold > 50% NaN
    nan_map = np.zeros(oldshape)
    nan_map[np.isnan(array)] = 1.
    sm_nan_map = imresize(nan_map, shape, interp='bilinear', mode='F')
    new_array[sm_nan_map >= 0.5] = np.nan

    return new_array

def grow_nan(iso_column, number_delete):
    '''
    Takes a string of floats with nans at the end, and replaces the last float value
    with an additional nan. Designed to function with gen_isomaps, where unreliable
    topmost model pixels can be replaced with nans, and so not be used for fitting.
    :param string:      a string of floats with nans at the end
    :return:            a string of the same size with the last float replaced by a nan
    '''

    nan_ct = np.sum(np.isnan(iso_column))
    iso_column[-(nan_ct+number_delete):iso_column.shape[0]-nan_ct] = np.nan

    return iso_column

def gen_isomaps(iso_shape, iso_data_x_ct, tooth_model, blood_step, day=-1):
    '''
    Takes mineralization model and blood isotope model to create modeled tooth isotope data,
    then downsamples these to the same resolution as real tooth isotope data, so that real
    and modeled results can be compared.
    :param iso_shape:           tuple, shape of real isotope data
    :param iso_data:            real tooth isotope data as string or array of floats
    :param iso_data_x_ct:       list with number nans per column in real isotope data
    :param tooth_model:         Class object including information about tooth mineralization
    :param blood_step:          string of blood isotope ratios per day as floats
    :param day:                 day of mineralization at which to carry out comparison with data
    :return:                    modeled tooth isotope data scaled to real data size, with the
                                appropriate number of nans, and and isotope data
    '''

    model_isomap = tooth_model.gen_isotope_image(blood_step[:day], mode=10) # did go from [:day+1] for some reason?
    for k in xrange(len(model_isomap)):
        model_isomap[k] = model_isomap[k][:,1:,day] + 18.6 #*** No. in middle denotes deletion from bottom PHOSPHATE_OFFSET*** was 18.8
        for c in xrange(model_isomap[k].shape[0]):
            model_isomap[k][c,:] = grow_nan(model_isomap[k][c,:], 2) # ***No. at end denotes deletion from top***

    re_shape = (iso_shape[0], iso_shape[1], len(model_isomap))
    remodeled = np.empty(re_shape, dtype='f8')

    for i in xrange(re_shape[2]):
        tmp = wizard(model_isomap[i], iso_shape)
        remodeled[:,:,i] = np.array(complex_resize(iso_shape, tmp.T.flatten(), iso_data_x_ct))

    return remodeled

def compare(model_isomap, model_isomap_2, model_isomap_3, data_isomap, w_iso_hist, prior_rate, score_max=100., data_sigma=0.20, sigma_floor=0.05):
    '''

    :param model_isomap:        modeled tooth isotope data
    :param data_isomap:         real tooth isotope data
    :param score_max:           maximum effect of single pixel comparison on likelihood
    :param data_sigma:
    :param sigma_floor:
    :return:
    '''

    m_model_isomap = np.ma.masked_array(model_isomap, np.isnan(model_isomap))
    m_model_isomap2 = np.ma.masked_array(model_isomap_2, np.isnan(model_isomap_2))
    m_model_isomap3 = np.ma.masked_array(model_isomap_3, np.isnan(model_isomap_3))
    m_data = np.ma.masked_array(data_isomap, np.isnan(data_isomap))
    model_isomap_1D = np.mean(m_model_isomap, axis=1)
    model_isomap_1D2 = np.mean(m_model_isomap2, axis=1)
    model_isomap_1D3 = np.mean(m_model_isomap3, axis=1)
    data_isomap_1D = np.mean(m_data, axis=1)

    sample_number = 18.

    model_isomap_1Dreshape = np.mean(np.repeat(model_isomap_1D, sample_number, axis=0).reshape(sample_number,27,model_isomap_1D.shape[1]), axis=1)
    model_isomap_1Dreshape2 = np.mean(np.repeat(model_isomap_1D2, sample_number, axis=0).reshape(sample_number,27,model_isomap_1D2.shape[1]), axis=1)
    model_isomap_1Dreshape3 = np.mean(np.repeat(model_isomap_1D3, sample_number, axis=0).reshape(sample_number,27,model_isomap_1D3.shape[1]), axis=1)
    data_isomap_reshape = np.mean(np.repeat(data_isomap_1D, sample_number).reshape(sample_number,27), axis=1)

    time_sigma_box = np.concatenate((model_isomap_1Dreshape, model_isomap_1Dreshape2, model_isomap_1Dreshape3), axis=1)
    time_sigma = np.std(time_sigma_box, axis=1)

    mu = np.mean(model_isomap_1Dreshape, axis=1)

    sigma = np.sqrt(time_sigma**2 + data_sigma**2 + sigma_floor**2.)

    score = (mu - data_isomap_reshape) / sigma

    score[~np.isfinite(score)] = 0.
    score[score > score_max] = score_max
    score = np.sum(score**2)

    trial_iteration[0] += 1.
    print trial_iteration[0]
    #prior_score = prior_histogram(mu, data_isomap)
    prior_score_rate = prior_rate_change(w_iso_hist, prior_rate) # Rate prior, Division due to pixel#
    #prior_score_hist = prior_histogram(mu, data_isomap)
    score_prior_counter.append(np.array([trial_iteration[0], score, prior_rate, prior_score_rate, score*(110./len(mu))]))

    return (score*(110./len(mu)))+prior_score_rate

def prior_histogram(model_isomap, data_isomap):

    model_real = np.isfinite(model_isomap)
    data_real = np.isfinite(data_isomap)
    min_max = ((np.min([np.min(model_isomap[model_real]), np.min(data_isomap[data_real])])), np.max([np.max(model_isomap[model_real]), np.max(data_isomap[data_real])]))
    model_hist = np.histogram(model_isomap[model_real], bins=10, range=min_max)
    data_hist = np.histogram(data_isomap[data_real], bins=10, range=min_max)

    hist_sigma = 0.3
    prior_score = (model_hist[0] - data_hist[0]) / hist_sigma
    prior_score = (np.sum(prior_score**2)) + np.sum(prior_score[0])

    return prior_score

def prior_rate_change(w_iso_hist, rate):

    diff_water = np.diff(w_iso_hist)
    #diff_water[int(M2_switch_days[0])-3:int(M2_switch_days[0])+1] = 0.
    #diff_water[int(M2_switch_days[1])-3:int(M2_switch_days[1])+1] = 0.
    prior_score = np.sum((diff_water/rate)**2.)

    return prior_score

def water_hist_likelihood(w_iso_hist, PO4_t, PO4_pause, PO4_flux, **kwargs):

    # Calculate water history on each day
    block_length = int(kwargs.get('block_length', 1))
    M2_inverse_water_hist = calc_water_step2(w_iso_hist, block_length)

    # Declare tooth growth parameters
    tooth_model = kwargs.get('tooth_model', None)
    assert(tooth_model != None)
    isomap_shape = kwargs.get('isomap_shape', None)
    data_isomap = kwargs.get('data_isomap', None)
    isomap_data_x_ct = kwargs.get('isomap_data_x_ct', None)
    prior_rate = kwargs.get('prior_rate', None)
    assert(isomap_shape != None)
    assert(data_isomap != None)
    assert(isomap_data_x_ct != None)

    m2_m1_params = np.array([67.974, 0.003352, -25.414, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m1_params2 = np.array([62.00, .0037, 0.00, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m1_params3 = np.array([74.00, .0031, -48.00, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k

    m1_m2_params = np.array([21.820, .007889, 29.118, 35., 67.974, 0.003352, -25.414, 41.]) # 'synch86', outlier, 100k
    m1_m2_params2 = np.array([21.820, .007889, 29.118, 35., 62.00, .0037, 0.00, 41.]) # 'synch86', outlier, 100k
    m1_m2_params3 = np.array([21.820, .007889, 29.118, 35., 74.00, .0031, -48.00, 41.]) # 'synch86', outlier, 100k

    # Declare physiological parameters
    d_O2 = kwargs.get('d_O2', 23.5)
    d_feed = kwargs.get('d_feed', 25.3)
    metabolic_kw = kwargs.get('metabolic_kw', {})

    # Generate blood and PO4 from proposed water
    M2_inverse_days = np.arange(84., 84.+np.size(M2_inverse_water_hist))
    M2_inverse_blood_hist = blood_delta(23.5, w_iso_hist, 25.3, **metabolic_kw)
    M2_inverse_PO4_eq = PO4_dissoln_reprecip(PO4_t, PO4_pause, PO4_flux, M2_inverse_blood_hist, **kwargs)

    # Create M1 days, water, blood and phosphate histories from M2 inversion results
    M1_inverse_days = tooth_timing_convert(M2_inverse_days, *m2_m1_params)
    M1_inverse_days = M1_inverse_days - M1_inverse_days[0]
    M1_inverse_PO4_hist_tmp = np.ones(M1_inverse_days.size)
    for k,d in enumerate(M1_inverse_days):
        d = int(d)
        M1_inverse_PO4_hist_tmp[d:] = M2_inverse_PO4_eq[k]
    M1_inverse_PO4_hist = M1_inverse_PO4_hist_tmp

    # Create M1 days, water, blood and phosphate histories from M2 inversion results
    M1_inverse_days2 = tooth_timing_convert(M2_inverse_days, *m2_m1_params2)
    M1_inverse_days2 = M1_inverse_days2 - M1_inverse_days2[0]
    M1_inverse_PO4_hist_tmp2 = np.ones(M1_inverse_days2.size)
    for k,d in enumerate(M1_inverse_days2):
        d = int(d)
        M1_inverse_PO4_hist_tmp2[d:] = M2_inverse_PO4_eq[k]
    M1_inverse_PO4_hist_2 = M1_inverse_PO4_hist_tmp2

    # Create M1 days, water, blood and phosphate histories from M2 inversion results
    M1_inverse_days3 = tooth_timing_convert(M2_inverse_days, *m2_m1_params3)
    M1_inverse_days3 = M1_inverse_days3 - M1_inverse_days3[0]
    M1_inverse_PO4_hist_tmp3 = np.ones(M1_inverse_days3.size)
    for k,d in enumerate(M1_inverse_days3):
        d = int(d)
        M1_inverse_PO4_hist_tmp3[d:] = M2_inverse_PO4_eq[k]
    M1_inverse_PO4_hist_3 = M1_inverse_PO4_hist_tmp3

    # Create M1 equivalent isomap models for M2 inversion results
    inverse_model_PO4 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_inverse_PO4_hist)
    inverse_model_PO4_2 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_inverse_PO4_hist_2)
    inverse_model_PO4_3 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_inverse_PO4_hist_3)

    # Calculate score comparing inverse to real
    score = compare(inverse_model_PO4, inverse_model_PO4_2, inverse_model_PO4_3, data_isomap, w_iso_hist, prior_rate) #

    return score, inverse_model_PO4

def water_hist_prob_4param(w_params, **kwargs):

    PO4_t = w_params[4]
    PO4_pause = w_params[5]
    PO4_flux = w_params[6]
    w_params = w_params[:4]
    w_iso_hist = water_4_param(*w_params)
    p, model_isomap = water_hist_likelihood(w_iso_hist, PO4_t, PO4_pause, PO4_flux, **kwargs)

    #prior_score = prior(w_params)
    #p += prior

    list_params = np.array([w_params[0], w_params[1], w_params[2], w_params[3], PO4_t, PO4_pause, PO4_flux])
    list_tuple = (p, list_params)
    my_list.append(list_tuple)

    return p, model_isomap

def water_hist_prob(w_params, **kwargs):

    iso_hist = np.ones(np.size(w_params))
    for i,j in enumerate(w_params):
        iso_hist[i] = j

    # Turning w parameters into daily d18O history, excluding switch params
    w_iso_hist = spline_input_signal(iso_hist[:40], 14, 1)

    p, model_isomap = water_hist_likelihood(w_iso_hist, 3.0, 34.5, .3, **kwargs)

    list_tuple = (p, np.array(w_params))
    my_list.append(list_tuple)

    return p, model_isomap

def score_v_score(w_iso_hist, fit_kwargs, step_size=0.1):
    f_min = lambda x: water_hist_likelihood(x, **fit_kwargs)
    score1 = f_min(w_iso_hist)
    iso2 = w_iso_hist + np.random.normal(loc=0., scale=step_size, size=w_iso_hist.size)
    score2 = f_min(iso2)
    return w_iso_hist, iso2, score1, score2

def water_4_param(mu, switch_mu, switch_start, switch_length):
    length_overall = np.ones(450.)
    w_iso_hist = length_overall * mu
    switch_start = int(switch_start)
    switch_length = int(switch_length)
    w_iso_hist[switch_start:switch_start+switch_length] = switch_mu

    return w_iso_hist

def figureplot_PO4_line(mu, switch_mu, switch_start, switch_length, d_O2, d_feed, PO4_t, PO4_pause, PO4_flux):
    length_overall = np.ones(450.)
    w_iso_hist = length_overall * mu
    switch_start = int(switch_start)
    switch_length = int(switch_length)
    w_iso_hist[switch_start:switch_start+switch_length] = switch_mu

    blood_hist = blood_delta(d_O2, w_iso_hist, d_feed)
    fig_tooth_PO4_eq = PO4_dissoln_reprecip(PO4_t, PO4_pause, PO4_flux, blood_hist)

    return fig_tooth_PO4_eq

def tooth_timing_convert(conversion_times, a1, s1, o1, max1, a2, s2, o2, max2):
    '''
    Takes an array of events in days occurring in one tooth, calculates where
    these will appear spatially during tooth extension, then maps these events
    onto the spatial dimensions of a second tooth, and calculates when similar
    events would have occurred in days to produce this mapping in the second
    tooth.

    Inputs:
    conversion_times:   a 1-dimensional numpy array with days to be converted.
    a1, s1, o1, max1:   the amplitude, slope, offset and max height of the error
                        function describing the first tooth's extension, in mm,
                        over time in days.
    a2, s2, o2, max2:   the amplitude, slope, offset and max height of the error
                        function describing the second tooth's extension, in mm,
                        over time in days.
    Returns:            converted 1-dimensional numpy array of converted days.

    '''
    t1_ext = a1*spec.erf(s1*(conversion_times-o1))+(max1-a1)
    t1_pct = t1_ext / max1
    t2_ext = t1_pct * max2
    converted_times = (spec.erfinv((a2+t2_ext-max2)/a2) + (o2*s2)) / s2

    return converted_times

def getkey(item):
    return item[0]

my_list = []

def spline_input_signal(iso_values, value_days, smoothness):
    '''
    Takes a series of iso_values, each lasting for a number of days called value_days,
    and interpolates to create a water history of the appropriate length iso_values*value_days.
    Has blood and water data from sheep 962 arranged from birth and outputs a
    day-by-day spline-smoothed version.
    '''

    spline_data_days = np.arange(np.size(iso_values))*value_days
    spline_output = InterpolatedUnivariateSpline(spline_data_days, iso_values, k=smoothness)
    days = np.arange(value_days*np.size(iso_values))
    water_spl = spline_output(days)

    return water_spl

def spline_962_input(smoothness):

    days_data = np.array([1.0, 31.0, 46.0, 58.0, 74.0, 102.0, 131.0, 162.0, 170.0, 198.0, 199.0, 200.0, 201.0, 202.0, 204.0, 204.0, 206.0, 208.0, 212.0, 212.0, 216.0, 219.0, 220.0, 221.0, 222.0, 232.0, 240.0, 261.0, 262.0, 272.0, 281.0, 282.0, 283.0, 284.0, 286.0, 290.0, 292.0, 298.0, 298.0, 310.0, 322.0, 358.0, 383.0, 411.0, 423.0, 453.0, 469.0, 483.0, 496.0])
    blood_days = np.array([58.0, 74.0, 102.0, 131.0, 162.0, 199.0, 201.0, 202.0, 204.0, 208.0, 212.0, 219.0, 222.0, 232.0, 261.0, 262.0, 281.0, 283.0, 284.0, 290.0, 298.0, 310.0, 322.0, 358.0, 383.0, 423.0, 453.0, 483.0])
    water_days = np.array([1.0, 31.0, 46.0, 74.0, 131.0, 170.0, 198.0, 199.0, 200.0, 201.0, 216.0, 219.0, 220.0, 221.0, 222.0, 261.0, 262.0, 272.0, 322.0, 358.0, 383.0, 411.0, 423.0, 469.0, 483.0, 496.0])
    blood_data = np.array([-5.71, -5.01, -4.07, -3.96, -4.53, -3.95, -4.96, -8.56, -10.34, -12.21, -13.09, -13.49, -13.16, -12.93, -13.46, -13.29, -5.68, -4.87, -4.76, -4.97, -4.60, -4.94, -5.45, -9.34, -5.56, -6.55, -4.25, -4.31])
    water_data = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])
    days = np.arange(84., 84.+np.max(days_data), 1.)

    water_spl = InterpolatedUnivariateSpline(water_days, water_data, k=smoothness)
    blood_spl = InterpolatedUnivariateSpline(blood_days, blood_data, k=smoothness)
    days_spl = days
    water_spl = np.array(water_spl(days))
    blood_spl = np.array(blood_spl(days))

    return water_spl, blood_spl, days_spl

def twoweek_input_signal(iso_values, value_days, smoothness):
    '''
    Takes a series of iso_values, each lasting for a number of days called value_days,
    and interpolates to create a water history of the appropriate length iso_values*value_days.
    Has blood and water data from sheep 962 arranged from birth and outputs a
    day-by-day spline-smoothed version.
    '''

    spline_data_days = np.arange(np.size(iso_values))*value_days
    spline_output = InterpolatedUnivariateSpline(spline_data_days, iso_values, k=smoothness)
    days = np.arange(value_days*np.size(iso_values))
    water_spl = spline_output(days)
    water_spl = water_spl[::14]

    return water_spl

def d2R(delta, standard=0.0020052):
    '''
    Convert isotope delta to Ratio.

    :param delta: delta of isotope
    :param standard: Ratio in standard (default: d18O/d16O in SMOW)
    :return: Ratio of isotope
    '''
    return (delta/1000. + 1.) * standard

def R2d(Ratio, standard=0.0020052):
    '''
    Convert isotope Ratio to delta.

    :param Ratio: Ratio of isotope
    :param standard: Ratio in standard (default: d18O/d16O in SMOW)
    :return: delta of isotope
    '''
    return (Ratio/standard - 1.) * 1000.

def guess_first(d_tooth, d_O2, d_feed, sample_number):

    d_tooth_reshape = np.mean(np.repeat(d_tooth, sample_number, axis=0).reshape(sample_number,d_tooth.shape[0]), axis=1)
    d_tooth_reshape -= 18.6

    # Get defaults
    f_H2O = 0.69
    f_O2 = 0.181
    alpha_O2 = 0.990
    f_feed = 0.129

    f_H2O_en = 0.69
    alpha_H2O_ef = .990
    f_H2O_ef = 0.129
    alpha_CO2_H2O = 1.040
    f_CO2 = 0.181

    # Calculate tooth water equilibrium for each sample

    R_water = (
        (alpha_CO2_H2O * f_CO2 * d2R(d_tooth_reshape))
        + (alpha_H2O_ef * f_H2O_ef * d2R(d_tooth_reshape))
        + (f_H2O_en * d2R(d_tooth_reshape))
        - (d2R(d_O2) * alpha_O2 * f_O2)
        - (d2R(d_feed) * f_feed)
    )

    R_water /= f_H2O

    R_water = R2d(R_water)

    guess_multiplier = 720. / (len(R_water)*2.2)
    data_guess_days = spline_input_signal(R_water, int(guess_multiplier), 1)
    intermediate = np.append(data_guess_days, np.linspace(data_guess_days[-1], np.mean(data_guess_days), 30.))
    first_guess = np.append(intermediate, np.ones(720.-len(intermediate))*np.mean(data_guess_days))

    return first_guess

def normal_sampling(d_tooth, d_O2, d_feed, sample_number):

    d_tooth_reshape = np.mean(np.repeat(d_tooth, sample_number, axis=0).reshape(sample_number,d_tooth.shape[0]), axis=1)
    d_tooth_reshape -= 18.6

    # Get defaults
    f_H2O = 0.69
    f_O2 = 0.181
    alpha_O2 = 0.990
    f_feed = 0.129

    f_H2O_en = 0.69
    alpha_H2O_ef = .990
    f_H2O_ef = 0.129
    alpha_CO2_H2O = 1.040
    f_CO2 = 0.181

    # Calculate tooth water equilibrium for each sample

    R_water = (
        (alpha_CO2_H2O * f_CO2 * d2R(d_tooth_reshape))
        + (alpha_H2O_ef * f_H2O_ef * d2R(d_tooth_reshape))
        + (f_H2O_en * d2R(d_tooth_reshape))
        - (d2R(d_O2) * alpha_O2 * f_O2)
        - (d2R(d_feed) * f_feed)
    )

    R_water /= f_H2O

    normal_samples = R2d(R_water)

    return normal_samples

def fit_tooth_data(data_fname, model_fname='equalsize_jul2015a.h5', **kwargs):
    '''
    '''

    print 'importing isotope data...'
    data_isomap, isomap_shape, isomap_data_x_ct = load_iso_data(data_fname)
    data_isomap_mask = np.ma.masked_array(data_isomap, np.isnan(data_isomap))
    data_mean_1D = np.mean(data_isomap_mask, axis=1)

    print 'loading tooth model ...'
    tooth_model_lg = ToothModel(model_fname)
    tooth_model = tooth_model_lg.downsample_model((isomap_shape[0]+5, isomap_shape[1]+5), 10) # Addition typically 5

    # Set keyword arguments to be used in fitting procedure
    fit_kwargs = kwargs.copy()

    fit_kwargs['tooth_model'] = tooth_model
    fit_kwargs['data_isomap'] = data_isomap
    fit_kwargs['isomap_shape'] = isomap_shape
    fit_kwargs['isomap_data_x_ct'] = isomap_data_x_ct

    # Blood and water isotope measurements from sheep 962
    blood_day_measures = np.array([(59., -5.71), (201., -4.96), (205., -10.34), (209., -12.21), (213., -13.14), (217., -13.49), (221., -13.16), (241., -13.46), (263., -13.29), (281., -4.87), (291., -4.97), (297., -4.60), (311., -4.94)])
    blood_days = np.array([i[0] for i in blood_day_measures])
    blood_measures = np.array([i[1] for i in blood_day_measures])
    water_iso_day_measures = np.array([(201., -6.6), (201., -19.4), (221., -19.3), (263., -19.4)])
    water_iso_days = np.array([i[0] for i in water_iso_day_measures])
    water_iso_measures = np.array([i[1] for i in water_iso_day_measures])

    # Newer blood and water isotope data sheep 962
    blood_days_962 = np.array([58.0, 74.0, 102.0, 131.0, 162.0, 199.0, 201.0, 202.0, 204.0, 208.0, 212.0, 219.0, 222.0, 232.0, 261.0, 262.0, 281.0, 283.0, 284.0, 290.0, 298.0, 310.0, 322.0, 358.0, 383.0, 423.0, 453.0, 483.0])
    water_days_962 = np.array([1.0, 31.0, 46.0, 74.0, 131.0, 170.0, 198.0, 199.0, 200.0, 201.0, 216.0, 219.0, 220.0, 221.0, 222.0, 261.0, 262.0, 272.0, 322.0, 358.0, 383.0, 411.0, 423.0, 469.0, 483.0, 496.0])
    blood_data_962 = np.array([-5.71, -5.01, -4.07, -3.96, -4.53, -3.95, -4.96, -8.56, -10.34, -12.21, -13.09, -13.49, -13.16, -12.93, -13.46, -13.29, -5.68, -4.87, -4.76, -4.97, -4.60, -4.94, -5.45, -9.34, -5.56, -6.55, -4.25, -4.31])
    water_data_962 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])

    # Model a M1 combined with different M2 possibilities
    m2_m1_params = np.array([67.974, 0.003352, -25.414, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m1_params2 = np.array([62.00, .0037, 0.00, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m1_params3 = np.array([74.00, .0031, -48.00, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k

    fit_kwargs['block_length'] = 1
    #record_scores = np.empty((trials,2), dtype='f4')

    data_guess = guess_first(data_mean_1D, 23.5, 25.3, 18)
    minmax = max(data_guess) - min(data_guess)
    minmax_diff = 1.2*np.log(minmax) + 4.

    guess_g = list(data_guess[:560:14])
    max_g = [x + minmax_diff for x in guess_g]
    min_g = [x - minmax_diff for x in guess_g]
    p_number = 40

    f_objective = lambda x, grad: water_hist_prob(x, **fit_kwargs)[0]

    # Parameters are main d18O, switch d18O, switch onset, switch length

    prior_rate = 1./5.
    fit_kwargs['prior_rate'] = prior_rate
    trials1 = 10000
    trials2 = 10000
    trials3 = 10000
    keep_pct = 30. # Percent of trials to record

    keep_pct = int((trials1+trials2+trials3)*(keep_pct/100.))
    keep_pct_jump = int(keep_pct/80.)

    # Parameters for time series only
    p_number = 40
    fit_kwargs['time_interval'] = 14.

    local_method = 'LN_COBYLA'
    local_opt = nlopt.opt(nlopt.LN_COBYLA, p_number)
    local_opt.set_xtol_abs(.01)
    local_opt.set_lower_bounds(min_g)
    local_opt.set_upper_bounds(max_g)
    local_opt.set_min_objective(f_objective)

    global_method = 'G_MLSL_LDS'
    global_opt = nlopt.opt(nlopt.G_MLSL_LDS, p_number)
    global_opt.set_maxeval(trials1)
    global_opt.set_lower_bounds(min_g)
    global_opt.set_upper_bounds(max_g)
    global_opt.set_min_objective(f_objective)
    global_opt.set_local_optimizer(local_opt)
    global_opt.set_population(p_number)
    print 'Running global optimizer ...'
    t1 = time()
    x_opt1 = global_opt.optimize(guess_g) # Is typically 'first_guess'

    prior_rate = 1./2.
    fit_kwargs['prior_rate'] = prior_rate
    global_opt.set_maxeval(trials2)
    x_opt2 = global_opt.optimize(x_opt1) # Should begin at last result

    prior_rate = 1./1.
    fit_kwargs['prior_rate'] = prior_rate
    global_opt.set_maxeval(trials3)
    x_opt = global_opt.optimize(x_opt1) # Should begin at last result
    trials = (trials1+trials2+trials3)

    minf = global_opt.last_optimum_value()
    print "optimum at", x_opt
    print "minimum value = ", minf
    print "result code = ", global_opt.last_optimize_result()
    t2 = time()
    run_time = t2-t1
    print 'run time = ', run_time
    eval_p_sec = trials/run_time

    # Model a M1 combined with different M2 possibilities
    m1_m2_params = np.array([21.820, .007889, 29.118, 35., 67.974, 0.003352, -25.414, 41.]) # 'synch86', outlier, 100k
    m1_m2_params2 = np.array([21.820, .007889, 29.118, 35., 62.00, .0037, 0.00, 41.]) # 'synch86', outlier, 100k
    m1_m2_params3 = np.array([21.820, .007889, 29.118, 35., 74.00, .0031, -48.00, 41.]) # 'synch86', outlier, 100k

    # Make trial forward data *******FORWARD BASED ON EXPECTATIONS PRIOR TO INVERSION*******
    forward_962_water_hist,forward_962_blood_hist,days_spl_962 = spline_962_input(1)
    forward_metabolic_kw = kwargs.get('metabolic_kw', {})

    # Introduce artificial forward
    sin_180 = 10.*np.sin((2*np.pi/180.)*(np.arange(600.)))-11.
    forward_962_water_hist = sin_180
    forward_962_blood_hist = blood_delta(23.5, forward_962_water_hist, 25.3, **forward_metabolic_kw)

    forward_962_phosphate_eq = PO4_dissoln_reprecip(3.0, 34.5, .3, forward_962_blood_hist, **kwargs)
    # Create M1-equivalent days, water, blood and PO4 eq
    m1_days_spl_962 = tooth_timing_convert(days_spl_962, *m2_m1_params)
    m1_days_spl_962 = m1_days_spl_962 - m1_days_spl_962[0]
    water_spl_tmp = np.ones(m1_days_spl_962.size)
    blood_spl_tmp = np.ones(m1_days_spl_962.size)
    PO4_spl_tmp = np.ones(m1_days_spl_962.size)
    for k,d in enumerate(m1_days_spl_962):
        d = int(d)
        water_spl_tmp[d:] = forward_962_water_hist[k]
        blood_spl_tmp[d:] = forward_962_blood_hist[k]
        PO4_spl_tmp[d:] = forward_962_phosphate_eq[k]
    forward_962_water_hist_m1 = water_spl_tmp
    forward_962_blood_hist_m1 = blood_spl_tmp
    forward_962_PO4_hist_m1 = PO4_spl_tmp
    #precip(3.0, 34.5, .3, forward_962_blood_hist_m1, **kwargs)

    forward_model_blood = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, forward_962_blood_hist_m1) # This takes the blood history from 962 scaled to the M1 without also downscaling the blood turnover
    forward_model_PO4 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, forward_962_PO4_hist_m1)

    # Assemble inverse trial result data in M2 format
    M2_inverse_water_hist = spline_input_signal(x_opt[:40], 14., 1)
    M2_switch_params = x_opt[40:]
    #M2_inverse_water_hist[M2_switch_params[2]:M2_switch_params[2]+M2_switch_params[3]] = M2_switch_params[1]
    M2_inverse_days = np.arange(84., 84.+np.size(M2_inverse_water_hist))
    M2_inverse_blood_hist = blood_delta(23.5, M2_inverse_water_hist, 25.3, **forward_metabolic_kw)
    M2_inverse_PO4_eq = PO4_dissoln_reprecip(3.0, 34.5, .3, M2_inverse_blood_hist, **kwargs)
    # Truncate M2 inverse trial results before conversion to M1 timing
    print 'm2 inverse days = ', M2_inverse_days

    # Create M1 days, water, blood and phosphate histories from M2 inversion results
    M1_inverse_days = tooth_timing_convert(M2_inverse_days, *m2_m1_params)
    M1_inverse_days = M1_inverse_days - M1_inverse_days[0]
    M1_inverse_PO4_hist_tmp = np.ones(M1_inverse_days.size)
    for k,d in enumerate(M1_inverse_days):
        d = int(d)
        M1_inverse_PO4_hist_tmp[d:] = M2_inverse_PO4_eq[k]
    M1_inverse_PO4_hist = M1_inverse_PO4_hist_tmp

    M1_inverse_days2 = tooth_timing_convert(M2_inverse_days, *m2_m1_params2)
    M1_inverse_days2 = M1_inverse_days2 - M1_inverse_days2[0]
    M1_inverse_PO4_hist_tmp2 = np.ones(M1_inverse_days2.size)
    for k,d in enumerate(M1_inverse_days2):
        d = int(d)
        M1_inverse_PO4_hist_tmp2[d:] = M2_inverse_PO4_eq[k]
    M1_inverse_PO4_hist2 = M1_inverse_PO4_hist_tmp2

    M1_inverse_days3 = tooth_timing_convert(M2_inverse_days, *m2_m1_params3)
    M1_inverse_days3 = M1_inverse_days3 - M1_inverse_days3[0]
    M1_inverse_PO4_hist_tmp3 = np.ones(M1_inverse_days3.size)
    for k,d in enumerate(M1_inverse_days3):
        d = int(d)
        M1_inverse_PO4_hist_tmp3[d:] = M2_inverse_PO4_eq[k]
    M1_inverse_PO4_hist3 = M1_inverse_PO4_hist_tmp3

    # Create M1 equivalent isomap models for M2 inversion results
    inverse_model_PO4 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_inverse_PO4_hist)
    inverse_model_PO4_2 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_inverse_PO4_hist2)
    inverse_model_PO4_3 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_inverse_PO4_hist3)

    m_model_isomap = np.ma.masked_array(inverse_model_PO4, np.isnan(inverse_model_PO4))
    m_model_isomap2 = np.ma.masked_array(inverse_model_PO4_2, np.isnan(inverse_model_PO4_2))
    m_model_isomap3 = np.ma.masked_array(inverse_model_PO4_3, np.isnan(inverse_model_PO4_3))
    m_data = np.ma.masked_array(data_isomap, np.isnan(data_isomap))
    model_isomap_1D = np.mean(m_model_isomap, axis=1)
    model_isomap_1D2 = np.mean(m_model_isomap2, axis=1)
    model_isomap_1D3 = np.mean(m_model_isomap3, axis=1)
    data_isomap_1D = np.mean(m_data, axis=1)

    sample_number = 18.

    model_isomap_1Dreshape = np.mean(np.repeat(model_isomap_1D, sample_number, axis=0).reshape(sample_number,27,model_isomap_1D.shape[1]), axis=1)
    model_isomap_1Dreshape2 = np.mean(np.repeat(model_isomap_1D2, sample_number, axis=0).reshape(sample_number,27,model_isomap_1D2.shape[1]), axis=1)
    model_isomap_1Dreshape3 = np.mean(np.repeat(model_isomap_1D3, sample_number, axis=0).reshape(sample_number,27,model_isomap_1D3.shape[1]), axis=1)
    data_isomap_reshape = np.mean(np.repeat(data_isomap_1D, sample_number).reshape(sample_number,27), axis=1)

    time_sigma_box = np.concatenate((model_isomap_1Dreshape, model_isomap_1Dreshape2, model_isomap_1Dreshape3), axis=1)
    time_sigma = np.std(time_sigma_box, axis=1)

    mu = np.mean(model_isomap_1Dreshape, axis=1)

    sigma = np.sqrt(time_sigma**2 + 0.20**2 + 0.05**2.)

    mu.shape = (sample_number, 1)
    data_isomap_reshape.shape = (sample_number, 1)
    sigma.shape = (sample_number, 1)

    # Real climate data
    #month_d180 = np.array([-1.95, -1.92, -2.94, -3.44, -2.22, -1.10, -0.67, -1.71, -0.81, -1.47, -2.31, -3.19]) # Dar Es Salaam
    month_addis = np.array([-0.21, 0.30, -0.04, 0.25, -0.75, -0.19, -3.16, -4.53, -0.95, 0.29, -1.26, -1.73]) # Addis Ababa
    month_entebbe = np.array([-1.39, -0.35, -2.42, -3.25, -3.08, -1.44, -0.98, -1.88, -1.33, -3.10, -3.80, -1.63]) # Entebbe, Uganda
    #month_d180 = np.array([-6.31, -7.09, -4.87, -3.33, -1.83, -1.22, -1.08, -0.47, -0.17, -0.48, -2.92, -5.90]) # Harare, Zimbabwe
    #month_d180 = np.array([-2.98, -2.20, -4.74, -5.94, -2.64, -3.80, -0.25, -1.80, -1.25, -4.15, -5.80, -5.42]) # Kinshasa, DRC
    #month_d180 = np.array([-1.58, -1.54, -1.81, -3.08, -3.40, -3.69, -3.38, -3.78, -2.46, -2.19, -2.12, -1.79]) # Cape Town
    #month_d180 = np.array([-4.31, -3.50, -4.14, -4.68, -4.87, -5.11, -4.77, -4.80, -4.71, -4.50, -4.53, -4.77]) # Marion Island
    #month_d180 = np.array([0.00, -2.40, -1.75, -3.70, -3.90, -6.20, -7.75, -8.10, -6.25, -3.30, -4.75, -8.95, -2.10, -0.40, -4.55, -3.25, -5.75, -3.70, -8.60, -7.10, -8.50, -5.30, -4.55, -3.10, -2.75	-4.60, -2.00, -3.10, -5.25, -6.10]) # Hong Kong
    #month_d180 = np.array([-2.75, -5.35, -2.70, -1.60, -6.30, -7.25, -9.00, -8.10, -9.50, -5.30, -5.75, -4.00]) # Liuzhou
    month_mulu = np.array([-5.30, -4.73, -7.44, -4.38, -4.39, -7.07, -9.76, -3.99, -3.95, -5.81, -8.98, -9.89, -8.62, -8.88, -8.25, -8.21, -9.74, -6.83, -6.69, -6.38, -10.33, -7.95, -5.72, -10.52, -10.74, -7.48, -9.30, -8.50, -12.66, -10.52, -10.82, -6.01, -8.34, -5.51, -7.03, -5.75, -8.14, -6.85, -4.82, -7.31, -8.79, -4.77, -6.14, -2.96, -2.31, -5.13, -9.31, -8.88, -9.22, -9.08, -7.51, -7.72, -10.29, -10.38, -9.69, -8.64, -10.66, -7.85, -6.94]) # Mulu, Borneo
    #week_d180 = np.array([-19.40, -19.4, -19.4, -19.4, -15.9, -15.9, -15.9, -23.1, -23.1, -23.1, -23.1, -23.1, -23.1, -23.1, -16.5, -16.5, -8.8, -8.8, -10.6, -10.6, -2.5, -9.3, -6.7, -8.2, -1.6, -6, -7, -4.4, -8.8, -6.5, -6.1, -6.1, -6.1, -0.6, 1.7, -4.5, -4.5, -4.5, -12.4, -12.4, -9.7, -12.2, -12.2, -12.2, -15.1, -15.1, -11, -11, -11, -30.5, -30.5, -30.5])  # North Platte Nebraska
    month_platte = np.array([-18.50, -17.93, -12.16, -12.08, -6.88, -7.00, -7.49, -5.60, -8.87, -13.91, -14.20, -23.70]) # North Platte, Nebraska

    # np.concatenate((month_d180, month_d180)) month_d180[:24]
    addis = spline_input_signal(np.concatenate((month_addis, month_addis)), 30, 1)
    entebbe = spline_input_signal(np.concatenate((month_entebbe, month_entebbe)), 30, 1)
    mulu = spline_input_signal(month_mulu[:24], 30, 1)
    platte = spline_input_signal(np.concatenate((month_platte, month_platte)), 30, 1)

    water_hist = mulu

    #Save my result trials
    my_list.sort(key=getkey)
    save_list = [my_list[i] for i in range(0, keep_pct, keep_pct_jump)]
    list_water_results = [s[1] for s in save_list]

    hist_list = [z[0] for j,z in enumerate(my_list[i] for i in range(len(my_list)))]

    # Save score data over time
    score_prior_counter_array = np.array(score_prior_counter)

    t_save = time()

    # Synthetic signal production

    sm_360 = 2.*np.sin((2*np.pi/360.)*(np.arange(600.)))-11.
    sm_180 = 2.*np.sin((2*np.pi/180.)*(np.arange(600.)))-11.
    sm_090 = 2.*np.sin((2*np.pi/90.)*(np.arange(600.)))-11.
    sm_045 = 2.*np.sin((2*np.pi/45.)*(np.arange(600.)))-11.

    sm_360_180 = (1.*np.sin((2*np.pi/180.)*(np.arange(600.)))) + sm_360
    sm_360_90 = (1.*np.sin((2*np.pi/90.)*(np.arange(600.)))) + sm_360
    sm_360_45 = (1.*np.sin((2*np.pi/45.)*(np.arange(600.)))) + sm_360
    sm_180_90 = (1.*np.sin((2*np.pi/90.)*(np.arange(600.)))) + sm_180
    sm_180_45 = (1.*np.sin((2*np.pi/45.)*(np.arange(600.)))) + sm_180

    sin_360 = 10.*np.sin((2*np.pi/360.)*(np.arange(600.)))-11.
    sin_180 = 10.*np.sin((2*np.pi/180.)*(np.arange(600.)))-11.
    sin_090 = 10.*np.sin((2*np.pi/90.)*(np.arange(600.)))-11.
    sin_045 = 10.*np.sin((2*np.pi/45.)*(np.arange(600.)))-11.

    sin_360_180 = (5.*np.sin((2*np.pi/180.)*(np.arange(600.)))) + sin_360
    sin_360_90 = (5.*np.sin((2*np.pi/90.)*(np.arange(600.)))) + sin_360
    sin_360_45 = (5.*np.sin((2*np.pi/45.)*(np.arange(600.)))) + sin_360
    sin_180_90 = (5.*np.sin((2*np.pi/90.)*(np.arange(600.)))) + sin_180
    sin_180_45 = (5.*np.sin((2*np.pi/45.)*(np.arange(600.)))) + sin_180

    textstr = 'min= %.2f, time= %.1f \n trials= %.1f, trials/sec= %.2f \n%s, %s' % (minf, run_time, trials, eval_p_sec, local_method, global_method)
    print textstr

    number = 'sin_180_45'
    #np.savetxt('{0}_{1}.csv'.format(number, t_save), np.array(save_list), delimiter=',', fmt='%.2f')

    water_hist = sin_180_45

    # Forward and Inverse result Diffs for histogram plotting
    for_inv_diff = M2_inverse_water_hist - water_hist[:len(M2_inverse_water_hist)]

    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    days = M2_inverse_days
    ax1.plot(days, water_hist[:days.size], 'k--', linewidth=1.0) # *************** *************** ***************
    ax1.plot(days, M2_inverse_water_hist, 'b-', linewidth=2.0)
    ax1.plot(days, M2_inverse_blood_hist, 'r-', linewidth=2.0)
    ax1.plot(days, M2_inverse_PO4_eq, 'g-.', linewidth=1.0)
    for s in list_water_results[:-1]:
        s = spline_input_signal(s[:40], 14., 1)
        M2_switch_params = s[40:]
        #s[M2_switch_params[2]:M2_switch_params[2]+M2_switch_params[3]] = M2_switch_params[1]
        ax1.plot(days, s, 'b-', alpha=0.03)
    ax1.text(350, -20, textstr, fontsize=8)
    ax1.set_ylim(-35, 15)
    ax1.set_xlim(85, 550)

    ax2 = fig.add_subplot(4,1,2)
    ax2text = 'Synthetic data'
    ax2.text(15, 1, ax2text, fontsize=8)
    cimg2 = ax2.imshow(data_isomap_reshape.T, aspect='equal', interpolation='nearest', origin='lower', cmap='bwr')
    cax2 = fig.colorbar(cimg2)

    ax3 = fig.add_subplot(4,1,3)
    ax3text = 'Inverse model result - PO4'
    ax3.text(15, 1, ax3text, fontsize=8)
    cimg3 = ax3.imshow(mu.T, aspect='equal', interpolation='nearest', origin='lower', cmap='bwr')
    cax3 = fig.colorbar(cimg3)

    ax4 = fig.add_subplot(4,1,4)
    ax4text = 'Inverse model sigma'
    ax4.text(15, 1, ax4text, fontsize=8)
    cimg4 = ax4.imshow(sigma.T, aspect='equal', interpolation='nearest', origin='lower', cmap='RdGy')
    cax4 = fig.colorbar(cimg4)

    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}a.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(days, water_hist[:days.size], 'k--', linewidth=1.0) # **************** ***************** ***********
    ax1.plot(days, M2_inverse_water_hist, 'b-', linewidth=2.0)
    ax1.plot(days, M2_inverse_blood_hist, 'r-', linewidth=2.0)
    ax1.plot(days, M2_inverse_PO4_eq, 'g-.', linewidth=1.0)
    for s in list_water_results[:-1]:
        s = spline_input_signal(s[:40], 14., 1)
        M2_switch_params = s[40:]
        #s[M2_switch_params[2]:M2_switch_params[2]+M2_switch_params[3]] = M2_switch_params[1]
        ax1.plot(days, s, 'b-', alpha=0.03)
    ax1.text(350, -18, textstr, fontsize=8)
    ax1.set_ylim(-35, 15)
    ax1.set_xlim(85, 550)

    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}b.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(days, water_hist[:days.size], 'k--', linewidth=1.0) # **************** sin curve here **********
    ax1.plot(days, M2_inverse_water_hist, 'b-', linewidth=2.0)
    ax1.plot(days, M2_inverse_blood_hist, 'r-', linewidth=2.0)
    ax1.plot(days, M2_inverse_PO4_eq, 'g-.', linewidth=1.0)
    ax1.plot(days[::14], guess_g, 'ko', markersize=3)
    for s in list_water_results[:-1]:
        s = spline_input_signal(s[:40], 14., 1)
        ax1.plot(days, s, 'b-', alpha=0.03)
    ax1.text(350, -4, textstr, fontsize=8)
    ax1.set_ylim(np.min(water_hist)-6., np.max(water_hist)+6.)
    ax1.set_xlim(84, 550)

    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}b2.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(days, water_hist[:days.size], 'k--', linewidth=1.0) # **************** sin curve here **********
    ax1.plot(days, M2_inverse_water_hist, 'b-', linewidth=2.0)
    ax1.plot(days, M2_inverse_blood_hist, 'r-', linewidth=2.0)
    ax1.plot(days, M2_inverse_PO4_eq, 'g-.', linewidth=1.0)
    for s in list_water_results[:-1]:
        s = spline_input_signal(s[:40], 14., 1)
        ax1.plot(days, s, 'b-', alpha=0.03)
    ax1.text(350, -18, textstr, fontsize=8)
    ax1.set_ylim(np.min(water_hist)-2., np.max(water_hist)+2.)
    ax1.set_xlim(84, 550)

    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}b3.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    plt.hist(hist_list, bins=np.logspace(0.0, 5.0, 30), alpha=.6)
    plt.gca().set_xscale("log")
    plt.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}c.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.hist(for_inv_diff, bins=np.linspace(-10.0, 10.0, 20), alpha=.6)
    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}d.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.hist(for_inv_diff, bins=np.linspace(-8.0, 8.0, 40), alpha=.6)
    ax1.hist(0.5*np.random.randn(len(for_inv_diff)), bins=np.linspace(-8.0, 8.0, 40), alpha=.6)
    ax1.set_ylim(0, 220)
    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}e.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(score_prior_counter_array[:,0], score_prior_counter_array[:,4], 'k-', label='multiplied score') # multiplied score
    ax1.plot(score_prior_counter_array[:,0], score_prior_counter_array[:,3], 'b-', label='prior score') # prior score
    #ax1.plot(score_prior_counter_array[:,0], score_prior_counter_array[:,2], 'k-', label='prior rate') # prior rate
    #ax1.plot(score_prior_counter_array[:,0], score_prior_counter_array[:,1], 'k-', label='raw score') # score
    ax1.legend(fontsize=8)
    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}f.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(score_prior_counter_array[:,0], score_prior_counter_array[:,3]/(score_prior_counter_array[:,3]+score_prior_counter_array[:,4])*100, 'k-', label='prior rate') # prior percent
    #ax1.plot(score_prior_counter_array[:,0], score_prior_counter_array[:,1], 'k-', label='raw score') # score
    ax1.legend(fontsize=8)
    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}g.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    normal_samples = normal_sampling(data_mean_1D, 23.5, 25.3, 18)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(days, water_hist[:days.size], 'k--', linewidth=1.0) # **************** sin curve here **********
    ax1.plot(days, M2_inverse_water_hist, 'b-', linewidth=1.0)
    for s in list_water_results[:-1]:
        s = spline_input_signal(s[:40], 14., 1)
        ax1.plot(days, s, 'b-', alpha=0.03)
    ax1.text(350, -6, textstr, fontsize=8)
    ax1.set_ylim(np.min(water_hist)-6., np.max(water_hist)+6.)
    ax1.set_xlim(84, 550)
    ax2 = ax1.twiny()
    ax2.plot(np.linspace(0, 42, len(normal_samples)), normal_samples, 'ko')
    ax2.set_xlim(0, 42)
    fig.savefig('1Dpx2o3_r{0}_18p6_{1}_{2}h.svg'.format(prior_rate, number, t_save), dpi=300, bbox_inches='tight')

    np.savetxt('1Dpx2o3_r{0}_18p6_{1}_{2}.csv'.format(prior_rate, number, t_save), score_prior_counter_array, fmt='%.4f', delimiter=',')

def main():

    fit_tooth_data('/Users/darouet/Documents/code/mineralization/clean code/PO4_sin_180_45.csv')

    return 0

if __name__ == '__main__':
    main()




