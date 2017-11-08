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
        model_isomap[k] = model_isomap[k][:,2:,day] + 18.6 #*** No. in middle denotes deletion from bottom PHOSPHATE_OFFSET*** was 18.8
        for c in xrange(model_isomap[k].shape[0]):
            model_isomap[k][c,:] = grow_nan(model_isomap[k][c,:], 1) # ***No. at end denotes deletion from top***

    re_shape = (iso_shape[0], iso_shape[1], len(model_isomap))
    remodeled = np.empty(re_shape, dtype='f8')

    for i in xrange(re_shape[2]):
        tmp = wizard(model_isomap[i], iso_shape)
        remodeled[:,:,i] = np.array(complex_resize(iso_shape, tmp.T.flatten(), iso_data_x_ct))

    return remodeled

def compare(model_isomap, model_isomap_2, model_isomap_3, data_isomap, score_max=100., data_sigma=0.20, sigma_floor=0.05):
    '''

    :param model_isomap:        modeled tooth isotope data
    :param data_isomap:         real tooth isotope data
    :param score_max:           maximum effect of single pixel comparison on likelihood
    :param data_sigma:
    :param sigma_floor:
    :return:
    '''
    time_sigma_box = np.concatenate((model_isomap, model_isomap_2, model_isomap_3), axis=2)
    time_sigma = np.std(time_sigma_box, axis=2)

    mu = np.median(model_isomap, axis=2)

    sigma = np.sqrt(time_sigma**2. + data_sigma**2. + sigma_floor**2)

    score = (mu - data_isomap) / sigma
    score[~np.isfinite(score)] = 0.
    score[score > score_max] = score_max
    score = np.sum(score**2)

    trial_iteration[0] += 1.

    #prior_score = prior_histogram(mu, data_isomap)
    #prior_score_hist = prior_histogram(mu, data_isomap)
    prior_score = 1.
    score_prior_counter.append(np.array([trial_iteration[0], score, prior_score, prior_score/score]))

    print trial_iteration[0], score

    return score

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
    prior_score = np.sum((diff_water/rate)**2.)

    return prior_score

def water_hist_likelihood(w_blood_hist, PO4_t, PO4_pause, PO4_flux, **kwargs):

    # Declare tooth growth parameters
    tooth_model = kwargs.get('tooth_model', None)
    assert(tooth_model != None)
    isomap_shape = kwargs.get('isomap_shape', None)
    data_isomap = kwargs.get('data_isomap', None)
    isomap_data_x_ct = kwargs.get('isomap_data_x_ct', None)
    assert(isomap_shape != None)

    m2_m1_params = np.array([67.974, 0.003352, -25.414, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m1_params2 = np.array([62.00, .0037, 0.00, 41.00, 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m1_params3 = np.array([74.00, .0031, -48.00, 41.0, 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k

    m1_m2_params = np.array([21.820, .007889, 29.118, 35., 67.974, 0.003352, -25.414, 41.]) # 'synch86', outlier, 100k
    m1_m2_params2 = np.array([21.820, .007889, 29.118, 35., 62.00, .0037, 0.00, 41.]) # 'synch86', outlier, 100k
    m1_m2_params3 = np.array([21.820, .007889, 29.118, 35., 74.00, .0031, -48.00, 41.]) # 'synch86', outlier, 100k

    # Generate blood and PO4 from proposed water
    M2_days = np.arange(84., 84.+np.size(w_blood_hist))
    M2_PO4_eq = new_PO4_equations(PO4_t, PO4_pause, PO4_flux, w_blood_hist)

    # Create M1 days, water, blood and phosphate histories from M2 inversion results
    M1_days = tooth_timing_convert(M2_days, *m2_m1_params)
    M1_days = M1_days - M1_days[0]
    M1_PO4_hist_tmp = np.ones(M1_days.size)
    for k,d in enumerate(M1_days):
        d = int(d)
        M1_PO4_hist_tmp[d:] = M2_PO4_eq[k]
    M1_PO4_hist = M1_PO4_hist_tmp

    # Create M1 days, water, blood and phosphate histories from M2 inversion results
    M1_days2 = tooth_timing_convert(M2_days, *m2_m1_params2)
    M1_days2 = M1_days2 - M1_days2[0]
    M1_PO4_hist_tmp2 = np.ones(M1_days2.size)
    for k,d in enumerate(M1_days2):
        d = int(d)
        M1_PO4_hist_tmp2[d:] = M2_PO4_eq[k]
    M1_PO4_hist_2 = M1_PO4_hist_tmp2

    # Create M1 days, water, blood and phosphate histories from M2 inversion results
    M1_days3 = tooth_timing_convert(M2_days, *m2_m1_params3)
    M1_days3 = M1_days3 - M1_days3[0]
    M1_PO4_hist_tmp3 = np.ones(M1_days3.size)
    for k,d in enumerate(M1_days3):
        d = int(d)
        M1_PO4_hist_tmp3[d:] = M2_PO4_eq[k]
    M1_PO4_hist_3 = M1_PO4_hist_tmp3

    # Create M1 equivalent isomap models for M2 inversion results
    model_PO4 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_PO4_hist)
    model_PO4_2 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_PO4_hist_2)
    model_PO4_3 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, M1_PO4_hist_3)

    # Calculate score comparing inverse to real
    score = compare(model_PO4, model_PO4_2, model_PO4_3, data_isomap)

    return score, model_PO4

def water_hist_prob(PO4_params, **kwargs):

    w_blood_hist = kwargs.get('blood_hist', {})

    p, model_isomap = water_hist_likelihood(w_blood_hist, PO4_params[0], PO4_params[1], PO4_params[2], **kwargs)

    list_tuple = (p, np.array(PO4_params))
    my_list.append(list_tuple)

    return p, model_isomap

def score_v_score(w_iso_hist, fit_kwargs, step_size=0.1):
    f_min = lambda x: water_hist_likelihood(x, **fit_kwargs)
    score1 = f_min(w_iso_hist)
    iso2 = w_iso_hist + np.random.normal(loc=0., scale=step_size, size=w_iso_hist.size)
    score2 = f_min(iso2)
    return w_iso_hist, iso2, score1, score2

def figureplot_PO4_line(mu, switch_mu, switch_start, switch_length, d_O2, d_feed, PO4_t, PO4_pause, PO4_flux):
    length_overall = np.ones(450.)
    w_iso_hist = length_overall * mu
    switch_start = int(switch_start)
    switch_length = int(switch_length)
    w_iso_hist[switch_start:switch_start+switch_length] = switch_mu

    blood_hist = blood_delta(d_O2, w_iso_hist, d_feed)
    fig_tooth_PO4_eq = new_PO4_equations(PO4_t, PO4_pause, PO4_flux, blood_hist)

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
prior_pct_list = []

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
    values_spl = spline_output(days)

    return values_spl, days

def spline_input(blood_days, water_days, blood_data, water_data, smoothness):

    max_days = np.max([np.max(blood_days),np.max(water_days)])
    days = np.arange(84., 84.+np.max(max_days), 1.)

    water_spl = InterpolatedUnivariateSpline(water_days, water_data, k=smoothness)
    blood_spl = InterpolatedUnivariateSpline(blood_days, blood_data, k=smoothness)

    days_spl = days

    water_spl = np.array(water_spl(days))
    blood_spl = np.array(blood_spl(days))

    return water_spl, blood_spl, days_spl

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

def PO4_evolve(start, blood_hist, PO4lambda, tex):
    '''
    x_frac = 1_d array of fractional contribution of original d18O over time in days
    x_frac_contrib = 1_d array fractional contribution each day of resetting PO4
    initial_value = composition original d18O
    '''

    blood_to_PO4_hist = blood_hist[start:]
    initial_value = blood_to_PO4_hist[0]
    x_frac = np.ones(len(blood_to_PO4_hist))
    for a,b in enumerate(x_frac[tex:]):
        x_frac[tex+a] = np.exp(-PO4lambda*a)
    x_frac_contrib = np.ones(len(blood_to_PO4_hist))
    for a,b in enumerate(x_frac[:-1]):
        x_frac_contrib[a] = x_frac[a] - x_frac[a+1]
    x_frac_contrib[-1] = 0.
    frac_HAp_added = np.sum(x_frac_contrib)
    added_d18O = blood_to_PO4_hist*(x_frac_contrib/frac_HAp_added)
    added_d18O[np.isnan(added_d18O)] = 0.
    added_d18O = np.sum(added_d18O)
    PO4_evolved = frac_HAp_added*added_d18O + (1-frac_HAp_added)*initial_value

    return PO4_evolved

def new_PO4_equations(half_life, tex, frac, blood_hist):

    PO4lambda = np.log(2.) / half_life
    tex = int(np.rint(tex))

    new_PO4_eq = np.ones(len(blood_hist))
    for a,b in enumerate(blood_hist):
        new_PO4_eq[a] = PO4_evolve(a, blood_hist, PO4lambda, tex)
        #print new_PO4_eq[a]
        new_PO4_eq[a] = frac*new_PO4_eq[a]+(1-frac)*blood_hist[a]
        #print new_PO4_eq[a]

    #print new_PO4_eq

    return new_PO4_eq

def create_water_hist(input_name):

    number = "{0}".format(input_name)
    number = number[-12:-9]

    # blood and water isotope data sheep 947
    blood_days_947 = np.array([58.00, 74.00, 102.00, 131.00, 162.00, 170.00, 200.00, 202.00, 204.00, 208.00, 212.00, 216.00, 220.00, 232.00, 240.00, 262.00, 272.00, 282.00, 284.00, 290.00, 292.00, 298.00, 310.00, 358.00, 383.00, 423.00, 453.00, 469.00, 483.00])
    blood_data_947 = np.array([-5.69, -4.91, -3.95, -3.89, -3.00, -4.80, -4.17, -7.58, -9.55, -10.87, -11.99, -12.26, -12.25, -11.75, -12.11, -12.36, -6.10, -4.43, -4.25, -4.41, -4.25, -4.15, -4.96, -12.34, -7.28, -8.01, -4.63, -5.22, -3.81])
    water_days_947 = np.array([1.00, 31.00, 46.00, 74.00, 131.00, 170.00, 198.00, 199.00, 200.00, 201.00, 216.00, 219.00, 220.00, 221.00, 222.00, 261.00, 262.00, 272.00, 322.00, 358.00, 383.00, 411.00, 423.00, 469.00, 483.00, 496.00])
    water_data_947 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])

    # blood and water isotope data sheep 949
    blood_days_949 = np.array([58.00, 74.00, 102.00, 131.00, 162.00, 170.00, 200.00, 204.00, 232.00, 272.00, 284.00, 286.00, 298.00, 322.00, 358.00, 383.00, 423.00, 453.00, 469.00, 483.00])
    blood_data_949 = np.array([-5.86, -4.88, -4.21, -3.86, -3.35, -3.00, -4.94, -5.02, -3.57, -4.37, -4.34, -4.67, -5.22, -4.72, -9.14, -6.02, -6.16, -4.97, -5.04, -4.15])
    water_days_949 = np.array([1.0, 31.0, 46.0, 74.0, 131.0, 170.0, 198.0, 199.0, 200.0, 201.0, 216.0, 219.0, 220.0, 221.0, 222.0, 261.0, 262.0, 272.0, 322.0, 358.0, 383.0, 411.0, 423.0, 469.0, 483.0, 496.0])
    water_data_949 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])

    # blood and water isotope data sheep 950
    blood_days_950 = np.array([58.00, 74.00, 102.00, 131.00, 162.00, 170.00, 200.00, 202.00, 204.00, 206.00, 220.00, 232.00, 240.00, 262.00, 272.00, 282.00, 286.00, 298.00, 310.00, 322.00, 423.00, 453.00, 469.00, 483.00, 496.00])
    blood_data_950 = np.array([-6.07, -4.99, -3.59, -4.02, -3.67, -4.08, -4.85, -8.46, -10.43, -11.33, -12.55, -4.53, -4.59, -4.70, -5.52, -5.01, -4.89, -4.56, -4.78, -4.61, -6.87, -4.66, -5.94, -4.04, -5.25])
    water_days_950 = np.array([1.0, 31.0, 46.0, 74.0, 131.0, 170.0, 198.0, 199.0, 200.0, 201.0, 216.0, 219.0, 220.0, 221.0, 222.0, 261.0, 262.0, 272.0, 322.0, 358.0, 383.0, 411.0, 423.0, 469.0, 483.0, 496.0])
    water_data_950 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])

    # blood and water isotope data sheep 962
    blood_days_962 = np.array([58.0, 74.0, 102.0, 131.0, 162.0, 199.0, 201.0, 202.0, 204.0, 208.0, 212.0, 219.0, 222.0, 232.0, 261.0, 262.0, 281.0, 283.0, 284.0, 290.0, 298.0, 310.0, 322.0, 358.0, 383.0, 423.0, 453.0, 483.0])
    blood_data_962 = np.array([-5.71, -5.01, -4.07, -3.96, -4.53, -3.95, -4.96, -8.56, -10.34, -12.21, -13.09, -13.49, -13.16, -12.93, -13.46, -13.29, -5.68, -4.87, -4.76, -4.97, -4.60, -4.94, -5.45, -9.34, -5.56, -6.55, -4.25, -4.31])
    water_days_962 = np.array([1.0, 31.0, 46.0, 74.0, 131.0, 170.0, 198.0, 199.0, 200.0, 201.0, 216.0, 219.0, 220.0, 221.0, 222.0, 261.0, 262.0, 272.0, 322.0, 358.0, 383.0, 411.0, 423.0, 469.0, 483.0, 496.0])
    water_data_962 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])

    # blood and water isotope data sheep 964
    blood_days_964 = np.array([58.00, 74.00, 102.00, 131.00, 162.00, 170.00, 200.00, 202.00, 204.00, 206.00, 212.00, 220.00, 232.00, 240.00, 262.00, 272.00, 282.00, 284.00, 286.00, 292.00, 298.00, 310.00, 322.00, 358.00, 423.00, 453.00, 469.00, 483.00])
    blood_data_964 = np.array([-5.86, -5.03, -4.23, -3.98, -4.14, -3.80, -4.42, -8.39, -9.92, -11.17, -12.76, -12.55, -4.60, -4.16, -4.45, -12.11, -12.87, -10.44, -8.81, -6.13, -4.87, -4.75, -6.06, -9.16, -5.57, -5.17, -5.61, -4.73])
    water_days_964 = np.array([1.0, 31.0, 46.0, 74.0, 131.0, 170.0, 198.0, 199.0, 200.0, 201.0, 216.0, 219.0, 220.0, 221.0, 222.0, 261.0, 262.0, 272.0, 322.0, 358.0, 383.0, 411.0, 423.0, 469.0, 483.0, 496.0])
    water_data_964 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])

    blood_days = np.ones(1000)
    water_days = np.ones(1000)
    blood_data = np.ones(1000)
    water_data = np.ones(1000)

    blood_days[blood_days==1.] = np.nan
    water_days[water_days==1.] = np.nan
    blood_data[blood_data==1.] = np.nan
    water_data[water_data==1.] = np.nan

    number = int(number)

    if number == 947:
        print "chosen:", number
        for i,j in enumerate(blood_days_947):
            blood_days[i] = j
        for i,j in enumerate(water_days_947):
            water_days[i] = j
        for i,j in enumerate(blood_data_947):
            blood_data[i] = j
        for i,j in enumerate(water_data_947):
            water_data[i] = j

    if number == 949:
        print "chosen:", number
        for i,j in enumerate(blood_days_949):
            blood_days[i] = j
        for i,j in enumerate(water_days_949):
            water_days[i] = j
        for i,j in enumerate(blood_data_949):
            blood_data[i] = j
        for i,j in enumerate(water_data_949):
            water_data[i] = j

    if number == 950:
        print "chosen:", number
        for i,j in enumerate(blood_days_950):
            blood_days[i] = j
        for i,j in enumerate(water_days_950):
            water_days[i] = j
        for i,j in enumerate(blood_data_950):
            blood_data[i] = j
        for i,j in enumerate(water_data_950):
            water_data[i] = j

    if number == 962:
        print "chosen:", number
        for i,j in enumerate(blood_days_962):
            blood_days[i] = j
        for i,j in enumerate(water_days_962):
            water_days[i] = j
        for i,j in enumerate(blood_data_962):
            blood_data[i] = j
        for i,j in enumerate(water_data_962):
            water_data[i] = j

    if number == 964:
        print "chosen:", number
        for i,j in enumerate(blood_days_964):
            blood_days[i] = j
        for i,j in enumerate(water_days_964):
            water_days[i] = j
        for i,j in enumerate(blood_data_964):
            blood_data[i] = j
        for i,j in enumerate(water_data_964):
            water_data[i] = j

    blood_days = blood_days[~np.isnan(blood_days)]
    water_days = water_days[~np.isnan(water_days)]
    blood_data = blood_data[~np.isnan(blood_data)]
    water_data = water_data[~np.isnan(water_data)]

    return number, blood_days, water_days, blood_data, water_data

def fit_tooth_data(data_fname, model_fname='equalsize_jul2015a.h5', **kwargs):
    '''
    '''

    print 'importing isotope data...'
    data_isomap, isomap_shape, isomap_data_x_ct = load_iso_data(data_fname)
    data_isomap_mask = np.ma.masked_array(data_isomap, np.isnan(data_isomap))
    data_mean_1D = np.mean(data_isomap_mask, axis=1)
    number, blood_days, water_days, blood_data, water_data = create_water_hist(data_fname)

    # Create day-by-day interpolation of water and blood history
    water_hist, blood_hist, days = spline_input(blood_days, water_days, blood_data, water_data, 1)

    print 'loading tooth model ...'
    tooth_model_lg = ToothModel(model_fname)
    tooth_model = tooth_model_lg.downsample_model((isomap_shape[0]+5, isomap_shape[1]+5), 10) # Model resampling size, sample number

    # Set keyword arguments to be used in fitting procedure
    fit_kwargs = kwargs.copy()

    fit_kwargs['tooth_model'] = tooth_model
    fit_kwargs['data_isomap'] = data_isomap
    fit_kwargs['isomap_shape'] = isomap_shape
    fit_kwargs['isomap_data_x_ct'] = isomap_data_x_ct
    fit_kwargs['blood_hist'] = blood_hist

    # Model a M1 combined with different M2 possibilities
    m2_m1_params = np.array([67.974, 0.003352, -25.414, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m1_params2 = np.array([62.00, .0037, 0.00, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m1_params3 = np.array([74.00, .0031, -48.00, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k

    f_objective = lambda x, grad: water_hist_prob(x, **fit_kwargs)[0]

    # Parameters are main d18O, switch d18O, switch onset, switch length

    trials = 600
    keep_pct = 30. # Percent of trials to record

    keep_pct = int((trials)*(keep_pct/100.))
    keep_pct_jump = int(keep_pct/80.)

    local_method = 'LN_COBYLA'
    local_opt = nlopt.opt(nlopt.LN_COBYLA, 3)
    local_opt.set_xtol_abs(.01)
    local_opt.set_lower_bounds([1., 1.0, 0.1])
    local_opt.set_upper_bounds([20., 80.0, 0.8])
    local_opt.set_min_objective(f_objective)

    global_method = 'G_MLSL_LDS'
    global_opt = nlopt.opt(nlopt.G_MLSL_LDS, 3)
    global_opt.set_maxeval(trials)
    global_opt.set_lower_bounds([1., 1.0, 0.1])
    global_opt.set_upper_bounds([20., 80.0, 0.8])
    global_opt.set_min_objective(f_objective)
    global_opt.set_local_optimizer(local_opt)
    global_opt.set_population(3)
    print 'Running global optimizer ...'
    t1 = time()
    x_opt = global_opt.optimize([3.0, 34., 0.3])

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

    # Calculate start time for model
    m1_gestation_times = np.array([-49., 0.])
    m2_gestation_times_curve = tooth_timing_convert([-49., 0.], *m1_m2_params)
    m2_gestation_curve = int(m2_gestation_times_curve[1] - m2_gestation_times_curve[0])

    # Make trial forward data *******FORWARD BASED ON EXPECTATIONS PRIOR TO INVERSION*******
    forward_metabolic_kw = kwargs.get('metabolic_kw', {})
    forward_phosphate_eq = new_PO4_equations(3.0, 35., .3, blood_hist)
    # Create M1-equivalent days, water, blood and PO4 eq
    m1_days_spl = tooth_timing_convert(days, *m2_m1_params)
    m1_days_spl = m1_days_spl - m1_days_spl[0]
    blood_spl_tmp = np.ones(m1_days_spl.size)
    PO4_spl_tmp = np.ones(m1_days_spl.size)
    for k,d in enumerate(m1_days_spl):
        d = int(d)
        blood_spl_tmp[d:] = blood_hist[k]
        PO4_spl_tmp[d:] = forward_phosphate_eq[k]
    forward_blood_hist_m1 = blood_spl_tmp
    forward_PO4_hist_m1 = PO4_spl_tmp

    # This takes the blood history scaled to the M1 without also downscaling the blood turnover
    forward_model_blood = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, forward_blood_hist_m1)
    forward_model_PO4 = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, forward_PO4_hist_m1)

    # Assemble inverse trial result data in M2 format
    M2_inverse_PO4_eq = new_PO4_equations(x_opt[0], x_opt[1], x_opt[2], blood_hist)

    # Truncate M2 inverse trial results before conversion to M1 timing
    print 'm2 inverse days = ', days
    # Create M1 days, water, blood and phosphate histories from M2 inversion results
    M1_inverse_days = tooth_timing_convert(days, *m2_m1_params)
    M1_inverse_days = M1_inverse_days - M1_inverse_days[0]
    M1_inverse_PO4_hist_tmp = np.ones(M1_inverse_days.size)
    for k,d in enumerate(M1_inverse_days):
        d = int(d)
        M1_inverse_PO4_hist_tmp[d:] = M2_inverse_PO4_eq[k]
    M1_inverse_PO4_hist = M1_inverse_PO4_hist_tmp

    M1_inverse_days2 = tooth_timing_convert(days, *m2_m1_params2)
    M1_inverse_days2 = M1_inverse_days2 - M1_inverse_days2[0]
    M1_inverse_PO4_hist_tmp2 = np.ones(M1_inverse_days2.size)
    for k,d in enumerate(M1_inverse_days2):
        d = int(d)
        M1_inverse_PO4_hist_tmp2[d:] = M2_inverse_PO4_eq[k]
    M1_inverse_PO4_hist_2 = M1_inverse_PO4_hist_tmp2

    M1_inverse_days3 = tooth_timing_convert(days, *m2_m1_params3)
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

    # calculate sigma

    time_sigma_box = np.concatenate((inverse_model_PO4, inverse_model_PO4_2, inverse_model_PO4_3), axis=2)
    time_sigma = np.std(time_sigma_box, axis=2)
    model_sigma = np.std(inverse_model_PO4, axis=2)
    sigma = np.sqrt(time_sigma**2 + 0.20**2 + 0.05**2)
    t_sigma = np.sqrt(time_sigma**2)

    # Calculate sigma for guess

    #Save my result trials BLANKED FOR NOW 2016-01-01
    my_list.sort(key=getkey)
    save_list = [my_list[i] for i in range(0, keep_pct, keep_pct_jump)]
    list_water_results = [s[1] for s in save_list]
    hist_list = [z[0] for j,z in enumerate(my_list[i] for i in range(len(my_list)))]

    # Save score data over time
    score_prior_counter_array = np.array(score_prior_counter)

    t_save = time()
    textstr = 'min= %.2f, time= %.1f \n trials= %.1f, trials/sec= %.2f \n%s, %s' % (minf, run_time, trials, eval_p_sec, local_method, global_method)

    # Forward and Inverse result Diffs for histogram plotting
    #for_inv_diff = M2_inverse_water_hist - water_hist[:len(M2_inverse_water_hist)]

    first_result_blood = np.ones(m2_gestation_curve)*blood_hist[0]
    first_result_PO4_eq = np.ones(m2_gestation_curve)*M2_inverse_PO4_eq[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    days = days
    ax1text = 'optimization result'
    ax1.plot(days[:m2_gestation_curve], first_result_blood, 'r-.', linewidth=2.0)
    ax1.plot(days[:m2_gestation_curve], first_result_PO4_eq, 'g-.', linewidth=1.0)
    ax1.plot(days[m2_gestation_curve:], blood_hist[:-m2_gestation_curve], 'r-', linewidth=2.0)
    ax1.plot(days[m2_gestation_curve:], M2_inverse_PO4_eq[:-m2_gestation_curve], 'g-', linewidth=1.0)
    ax1.plot(blood_days, blood_data, 'r*', linewidth=1.0)
    ax1.plot(water_days, water_data, 'b*', linewidth=1.0)
    for s in list_water_results[:-1]:
        s = new_PO4_equations(s[0], s[1], s[2], blood_hist)
        s_init = np.ones(m2_gestation_curve)*s[0]
        ax1.plot(days[:m2_gestation_curve], s_init, 'g-.', alpha=0.03)
        ax1.plot(days[m2_gestation_curve:], s[m2_gestation_curve:], 'g-', alpha=0.03)
    ax1.text(350, -20, textstr, fontsize=8)
    ax1.set_ylim(-35, 15)
    ax1.set_xlim(35, 750)
    fig.savefig('2D_r{0}_18p6_{1}_z.svg'.format(number, t_save), dpi=300, bbox_inches='tight')


    fig = plt.figure()
    ax2 = fig.add_subplot(4,1,1)
    ax2text = 'Data'
    ax2.text(3, 5, ax2text, fontsize=8)
    cimg2 = ax2.imshow(data_isomap.T, aspect='auto', interpolation='nearest', origin='lower', cmap='bwr', vmin=6.4, vmax=12.8)
    cax2 = fig.colorbar(cimg2)

    ax3 = fig.add_subplot(4,1,2)
    ax3text = 'Inverse model result PO4 {0}'.format(number)
    ax3.text(3, 5, ax3text, fontsize=8)
    cimg3 = ax3.imshow(np.mean(inverse_model_PO4, axis=2).T, aspect='auto', interpolation='nearest', origin='lower', cmap='bwr')
    cax3 = fig.colorbar(cimg3)

    residuals = np.mean(inverse_model_PO4, axis=2).T - data_isomap.T
    ax4 = fig.add_subplot(4,1,3)
    ax4text = 'residuals'
    ax4.text(3, 5, ax4text, fontsize=8)
    cimg4 = ax4.imshow(residuals, aspect='auto', interpolation='nearest', origin='lower', cmap='RdGy') # Residuals
    cax4 = fig.colorbar(cimg4)

    m_sigma = np.ma.masked_array(sigma, ~np.isnan(sigma))
    m_t_sigma = np.ma.masked_array(t_sigma, ~np.isnan(t_sigma))
    m_m_sigma = np.ma.masked_array(model_sigma, ~np.isnan(model_sigma))
    sigma_percentiles = np.percentile(m_sigma, (5,50,95))
    t_sigma_percentiles = np.percentile(m_t_sigma, (5,50,95))
    m_sigma_percentiles = np.percentile(m_m_sigma, (5,50,95))
    ax5 = fig.add_subplot(4,1,4)
    ax5text = 'sigma, {0}'.format(sigma_percentiles)
    ax5text2 = 'time sigma, {0}'.format(t_sigma_percentiles)
    ax5text3 = 'model sigma, {0}'.format(m_sigma_percentiles)
    ax5.text(3, 4, ax5text, fontsize=8)
    ax5.text(3, 3, ax5text2, fontsize=8)
    ax5.text(3, 2, ax5text3, fontsize=8)
    cimg5 = ax5.imshow(sigma.T, aspect='auto', interpolation='nearest', origin='lower', cmap='RdGy') # Residuals
    cax5 = fig.colorbar(cimg5)

    fig.savefig('2D_r{0}_18p6_{1}_a.svg'.format(number, t_save), dpi=300, bbox_inches='tight')

def main():

    fit_tooth_data('/Users/darouet/Documents/code/mineralization/clean code/947_data.csv')

    return 0

if __name__ == '__main__':
    main()




