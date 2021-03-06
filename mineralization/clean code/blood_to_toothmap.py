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
import matplotlib
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import pylab as Pylb
import scipy.special as spec
import nlopt
from time import time


from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.misc import imresize
from blood_delta import calc_blood_step, calc_water_step2, calc_water_gaussian, calc_blood_gaussian, blood_delta, tooth_phosphate_reservoir
from blood_delta_experimentation import PO4_dissoln_reprecip
from scipy.optimize import curve_fit, minimize, leastsq

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

        print 'calculating cumulative mineral increase in tooth over time...' # takes 100 seconds
        pct_min_diff_days[np.isnan(pct_min_diff_days)] = 0.
        pct_min_days = np.cumsum(pct_min_diff_days, axis=2) # All checks out except 0th, 1st, 2nd val
        pct_min_days[pct_min_days==0.] = np.nan

        print 'multiplying daily mineral additions by daily isotope ratios...' # takes 100 seconds
        isotope = np.cumsum(
            blood_step[None, None, :]
            * pct_min_diff_days,
            axis=2
        ) # Works at and after 3rd day. Something weird (all zero values?) happens before that.

        print 'calculating isotope ratios in tooth for each day of growth...' # takes 60-100 seconds
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
    sigma_t = 8
    sigma_x, sigma_y = 1, 1
    img = gaussian_filter(img, (sigma_t,sigma_x,sigma_y), mode='nearest')
    
    idx = np.isfinite(img)
    vmax = np.percentile(img[idx], 99.8)
    #vmax = 1.

    fig = plt.figure(figsize=(6,3), dpi=100)
    ax = fig.add_subplot(1,1,1)
    plt.tick_params(
        axis='none',
        which='none',
        bottom='off',
        top='off',
        labelbottom='off'
    )
    
    im = ax.imshow(
        np.zeros(shape[::-1], dtype='f8'),
        origin='lower',
        interpolation='nearest',
        vmin=0.,
        vmax=vmax,
        aspect=5
    )
    
    for k,t in enumerate(ages[:-1]):
        fn = '%s_k%04d.png' % (outfname, k)

        print 'Generating frame {0}...'.format(fn)

        #img = img_interp(t)
        im.set_data(img[k].T) #was(img[k].T)

        ax.set_title(r'$\mathrm{Sheep} \ \mathrm{molar} \ \mathrm{mineralization,} \ t = %03d \ \mathrm{days}$' % t, fontsize=12)
        plt.axis('off')
        fig.savefig(fn, dpi=300)

def gen_isomap_movie(tooth_model, blood_step):

    print 'generating movie...' #
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

def gen_isomaps_fullsize(tooth_model, blood_step, day=-1):
    '''
    Takes mineralization model and blood isotope model to create modeled tooth isotope data,
    then downsamples these to the same resolution as real tooth isotope data, so that real
    and modeled results can be compared. Also returns full size model.
    :param iso_shape:           tuple, shape of real isotope data
    :param iso_data:            real tooth isotope data as string or array of floats
    :param tooth_model:         Class object including information about tooth mineralization
    :param blood_step:          string of blood isotope ratios per day as floats
    :param day:                 day of mineralization at which to carry out comparison with data
    :returns:                   modeled tooth isotope data at full scale, and also scaled to real
                                data size (with the appropriate number of nans).
    '''

    model_isomap = tooth_model.gen_isotope_image(blood_step, mode=10)
    for k in xrange(len(model_isomap)):
        model_isomap[k] = model_isomap[k][:,1:,day] + 18.6
        for c in xrange(model_isomap[k].shape[0]):
            model_isomap[k][c,:] = grow_nan(model_isomap[k][c,:], 2)

    return model_isomap

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
                                appropriate number of nans.
    '''

    model_isomap = tooth_model.gen_isotope_image(blood_step, mode=10)
    for k in xrange(len(model_isomap)):
        model_isomap[k] = model_isomap[k][:,1:,day] + 18.6
        for c in xrange(model_isomap[k].shape[0]):
            model_isomap[k][c,:] = grow_nan(model_isomap[k][c,:], 2)

    re_shape = (iso_shape[0], iso_shape[1], len(model_isomap))
    remodeled = np.empty(re_shape, dtype='f8')

    for i in xrange(re_shape[2]):
        tmp = wizard(model_isomap[i], iso_shape)
        remodeled[:,:,i] = np.array(complex_resize(iso_shape, tmp.T.flatten(), iso_data_x_ct))

    return remodeled

def compare(model_isomap, data_isomap, score_max=3., data_sigma=0.15, sigma_floor=0.05):
    '''

    :param model_isomap:        modeled tooth isotope data
    :param data_isomap:         real tooth isotope data
    :param score_max:           maximum effect of single pixel comparison on likelihood
    :param data_sigma:
    :param sigma_floor:
    :return:
    '''

    mu = np.median(model_isomap, axis=2)
    sigma = np.std(model_isomap, axis=2)
    sigma = np.sqrt(sigma**2. + data_sigma**2. + sigma_floor**2.)
    score = (mu - data_isomap) / sigma
    score[~np.isfinite(score)] = 0.
    score[score > score_max] = score_max

    return -np.sum(score**2)


def water_hist_likelihood(w_iso_hist, **kwargs):
    # Calculate water history on each day
    #block_length = int(kwargs.get('block_length'))
    #w_iso_hist = calc_water_step2(w_iso_hist, block_length)

    # Water to blood history
    d_O2 = kwargs.get('d_O2', 23.5)
    d_feed = kwargs.get('d_feed', 25.3)
    metabolic_kw = kwargs.get('metabolic_kw', {})
    blood_hist = blood_delta(d_O2, w_iso_hist, d_feed, **metabolic_kw)
    phosphate_eq = PO4_dissoln_reprecip(3., 34.5, .3, blood_hist, **kwargs) #***** 17.609, 34.515, 54.922 *****

    # Access tooth model
    tooth_model = kwargs.get('tooth_model', None)
    assert(tooth_model != None)

    # Access tooth data
    isomap_shape = kwargs.get('isomap_shape', None)
    data_isomap = kwargs.get('data_isomap', None)
    isomap_data_x_ct = kwargs.get('isomap_data_x_ct', None)
    assert(isomap_shape != None)
    assert(data_isomap != None)
    assert(isomap_data_x_ct != None)

    # Calculate model tooth isomap
    model_isomap = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model, phosphate_eq)

    return compare(model_isomap, data_isomap), model_isomap

def water_hist_likelihood_fl(w_iso_hist, **kwargs):
    # Calculate water history on each day
    #block_length = int(kwargs.get('block_length'))
    #w_iso_hist = calc_water_step2(w_iso_hist, block_length)

    # Water to blood history
    d_O2 = kwargs.get('d_O2', 23.5)
    d_feed = kwargs.get('d_feed', 25.3)
    metabolic_kw = kwargs.get('metabolic_kw', {})
    blood_hist = blood_delta(d_O2, w_iso_hist, d_feed, **metabolic_kw)
    phosphate_eq = PO4_dissoln_reprecip(17.609, 34.515, .54922, blood_hist, **kwargs) #***** 17.609, 34.515, 54.922 *****

    # Access tooth model
    tooth_model = kwargs.get('tooth_model', None)
    assert(tooth_model != None)

    # Calculate model tooth isomap
    model_isomap = gen_isomaps_fullsize(tooth_model, phosphate_eq)

    return model_isomap

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


#def generate_input_signal(fname, time_unit=1., signal_type=water, smoothing=0.):
#    '''
#    This function is not written yet.
#    Inputs
#    fname       filename where blood or water data will be retrieved.
#    time_unit   the number of days requested per data point.
#    signal_type specify whether mineralization model will take blood data directly
#                or water data and convert to blood.
#    smoothing   request if input data should be smoothed on scale of 0-10.
#    PO4_reset   specify if PO4 resetting should occur.
#
#    Outputs
#    blood_hist  outputs daily blood isotope history
#    '''
#    blood_hist = np.arange(10)
#
#    return blood_hist


def spline_962_input(smoothness):
    '''
    Has blood and water data from sheep 962 arranged from birth and outputs a
    day-by-day spline-smoothed version.
    '''

    days_data = np.array([1.0, 31.0, 46.0, 58.0, 74.0, 102.0, 131.0, 162.0, 170.0, 198.0, 199.0, 200.0, 201.0, 202.0, 204.0, 204.0, 206.0, 208.0, 212.0, 212.0, 216.0, 219.0, 220.0, 221.0, 222.0, 232.0, 240.0, 261.0, 262.0, 272.0, 281.0, 282.0, 283.0, 284.0, 286.0, 290.0, 292.0, 298.0, 298.0, 310.0, 322.0, 358.0, 383.0, 411.0, 423.0, 453.0, 469.0, 483.0, 496.0])
    blood_days = np.array([58.0, 74.0, 102.0, 131.0, 162.0, 199.0, 201.0, 202.0, 204.0, 208.0, 212.0, 219.0, 222.0, 232.0, 261.0, 262.0, 281.0, 283.0, 284.0, 290.0, 298.0, 310.0, 322.0, 358.0, 383.0, 423.0, 453.0, 483.0])
    water_days = np.array([1.0, 31.0, 46.0, 74.0, 131.0, 170.0, 198.0, 199.0, 200.0, 201.0, 216.0, 219.0, 220.0, 221.0, 222.0, 261.0, 262.0, 272.0, 322.0, 358.0, 383.0, 411.0, 423.0, 469.0, 483.0, 496.0])
    blood_data = np.array([-5.71, -5.01, -4.07, -3.96, -4.53, -3.95, -4.96, -8.56, -10.34, -12.21, -13.09, -13.49, -13.16, -12.93, -13.46, -13.29, -5.68, -4.87, -4.76, -4.97, -4.60, -4.94, -5.45, -9.34, -5.56, -6.55, -4.25, -4.31])
    water_data = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])
    ice_33 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -9.87, -5.93, -8.89, -9.36, -6.83, -6.65, -6.98])
    ice_50 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -11.89, -5.93, -10.15, -10.51, -6.83, -6.65, -6.98])
    ice_66 = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -13.67, -5.93, -11.21, -11.45, -6.83, -6.65, -6.98])

    days = np.arange(1., np.max(days_data), 1.)

    water_spl = InterpolatedUnivariateSpline(water_days, water_data, k=smoothness)
    blood_spl = InterpolatedUnivariateSpline(blood_days, blood_data, k=smoothness)
    i33_spl = InterpolatedUnivariateSpline(water_days, ice_33, k=smoothness)
    i50_spl = InterpolatedUnivariateSpline(water_days, ice_50, k=smoothness)
    i66_spl = InterpolatedUnivariateSpline(water_days, ice_66, k=smoothness)
    plt.plot(water_days, water_data, 'bo', ms=5)
    plt.plot(blood_days, blood_data, 'ro', ms=5)
    plt.plot(days, water_spl(days), 'b', lw=2, alpha=0.6)
    plt.plot(days, blood_spl(days), 'r', lw=2, alpha=0.6)
    #plt.show()
    days_spl = days
    water_spl = np.array(water_spl(days))
    blood_spl = np.array(blood_spl(days))
    i33_spl = np.array(i33_spl(days))
    i50_spl = np.array(i50_spl(days))
    i66_spl = np.array(i66_spl(days))

    return water_spl, blood_spl, days_spl, i33_spl, i50_spl, i66_spl

def spline_input_signal(iso_values, value_days, smoothness):
    '''
    Takes a series of iso_values, each lasting for a number of days called value_days,
    and interpolates to create a water history of the appropriate length iso_values*value_days.
    Has blood and water data from sheep 962 arranged from birth and outputs a
    day-by-day spline-smoothed version.
    '''

    #days_data = np.array([1.0, 31.0, 46.0, 58.0, 74.0, 102.0, 131.0, 162.0, 170.0, 198.0, 199.0, 200.0, 201.0, 202.0, 204.0, 204.0, 206.0, 208.0, 212.0, 212.0, 216.0, 219.0, 220.0, 221.0, 222.0, 232.0, 240.0, 261.0, 262.0, 272.0, 281.0, 282.0, 283.0, 284.0, 286.0, 290.0, 292.0, 298.0, 298.0, 310.0, 322.0, 358.0, 383.0, 411.0, 423.0, 453.0, 469.0, 483.0, 496.0])
    #blood_days = np.array([58.0, 74.0, 102.0, 131.0, 162.0, 199.0, 201.0, 202.0, 204.0, 208.0, 212.0, 219.0, 222.0, 232.0, 261.0, 262.0, 281.0, 283.0, 284.0, 290.0, 298.0, 310.0, 322.0, 358.0, 383.0, 423.0, 453.0, 483.0])
    #water_days = np.array([1.0, 31.0, 46.0, 74.0, 131.0, 170.0, 198.0, 199.0, 200.0, 201.0, 216.0, 219.0, 220.0, 221.0, 222.0, 261.0, 262.0, 272.0, 322.0, 358.0, 383.0, 411.0, 423.0, 469.0, 483.0, 496.0])
    #blood_data = np.array([-5.71, -5.01, -4.07, -3.96, -4.53, -3.95, -4.96, -8.56, -10.34, -12.21, -13.09, -13.49, -13.16, -12.93, -13.46, -13.29, -5.68, -4.87, -4.76, -4.97, -4.60, -4.94, -5.45, -9.34, -5.56, -6.55, -4.25, -4.31])
    #water_data = np.array([-8.83, -8.83, -6.04, -6.19, -6.85, -7.01, -6.61, -6.61, -19.41, -19.41, -19.31, -19.31, -19.31, -19.31, -19.31, -19.31, -6.32, -6.32, -5.94, -17.63, -5.93, -13.66, -13.67, -6.83, -6.65, -6.98])
    #days = np.arange(1., np.max(days_data), 1.)

    #water_spl = InterpolatedUnivariateSpline(water_days, water_data, k=smoothness)
    #blood_spl = InterpolatedUnivariateSpline(blood_days, blood_data, k=smoothness)
    #plt.plot(water_days, water_data, 'bo', ms=5)
    #plt.plot(blood_days, blood_data, 'ro', ms=5)
    #plt.plot(days, water_spl(days), 'b', lw=2, alpha=0.6)
    #plt.plot(days, blood_spl(days), 'r', lw=2, alpha=0.6)
    #plt.show()
    #days_spl = days
    #water_spl = np.array(water_spl(days))
    #blood_spl = np.array(blood_spl(days))

    spline_data_days = np.arange(np.size(iso_values))*value_days
    spline_output = InterpolatedUnivariateSpline(spline_data_days, iso_values, k=smoothness)
    days = np.arange(value_days*np.size(iso_values))
    water_spl = spline_output(days)

    return water_spl


def fit_tooth_data(data_fname, model_fname='equalsize_jul2015a.h5', **kwargs):


    print 'importing isotope data...'

    data_isomap, isomap_shape, isomap_data_x_ct = load_iso_data(data_fname)

    print 'loading tooth model ...'
    tooth_model = ToothModel(model_fname)

    fit_kwargs = kwargs.copy()

    # M1-M2 conversion parameters
    m2_m1_params = np.array([67.974, 0.003352, -25.414, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    #m2_m1_params = np.array([78.940, 0.003379, -49.708, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outliers, 100k
    m1_m2_params = np.array([21.820, .007889, 29.118, 35., 67.974, 0.003352, -25.414, 41.]) # 'synch86', outlier, 100k
    #m1_m2_params = np.array([21.820, .007889, 29.118, 35., 78.940, 0.003379, -49.708, 41.]) # 'synch86', outliers, 100k

    # Real climate data
    #month_d180 = np.array([-1.95, -1.92, -2.94, -3.44, -2.22, -1.10, -0.67, -1.71, -0.81, -1.47, -2.31, -3.19]) # Dar Es Salaam
    #month_d180 = np.array([-0.21, 0.30, -0.04, 0.25, -0.75, -0.19, -3.16, -4.53, -0.95, 0.29, -1.26, -1.73]) # Addis Ababa
    #month_d180 = np.array([-1.39, -0.35, -2.42, -3.25, -3.08, -1.44, -0.98, -1.88, -1.33, -3.10, -3.80, -1.63]) # Entebbe, Uganda
    #month_d180 = np.array([-6.31, -7.09, -4.87, -3.33, -1.83, -1.22, -1.08, -0.47, -0.17, -0.48, -2.92, -5.90]) # Harare, Zimbabwe
    #month_d180 = np.array([-2.98, -2.20, -4.74, -5.94, -2.64, -3.80, -0.25, -1.80, -1.25, -4.15, -5.80, -5.42]) # Kinshasa, DRC
    #month_d180 = np.array([-1.58, -1.54, -1.81, -3.08, -3.40, -3.69, -3.38, -3.78, -2.46, -2.19, -2.12, -1.79]) # Cape Town
    #month_d180 = np.array([-4.31, -3.50, -4.14, -4.68, -4.87, -5.11, -4.77, -4.80, -4.71, -4.50, -4.53, -4.77]) # Marion Island
    #month_d180 = np.array([0.00, -2.40, -1.75, -3.70, -3.90, -6.20, -7.75, -8.10, -6.25, -3.30, -4.75, -8.95, -2.10, -0.40, -4.55, -3.25, -5.75, -3.70, -8.60, -7.10, -8.50, -5.30, -4.55, -3.10, -2.75	-4.60, -2.00, -3.10, -5.25, -6.10]) # Hong Kong
    #month_d180 = np.array([-2.75, -5.35, -2.70, -1.60, -6.30, -7.25, -9.00, -8.10, -9.50, -5.30, -5.75, -4.00]) # Liuzhou
    #month_d180 = np.array([-5.30, -4.73, -7.44, -4.38, -4.39, -7.07, -9.76, -3.99, -3.95, -5.81, -8.98, -9.89, -8.62, -8.88, -8.25, -8.21, -9.74, -6.83, -6.69, -6.38, -10.33, -7.95, -5.72, -10.52, -10.74, -7.48, -9.30, -8.50, -12.66, -10.52, -10.82, -6.01, -8.34, -5.51, -7.03, -5.75, -8.14, -6.85, -4.82, -7.31, -8.79, -4.77, -6.14, -2.96, -2.31, -5.13, -9.31, -8.88, -9.22, -9.08, -7.51, -7.72, -10.29, -10.38, -9.69, -8.64, -10.66, -7.85, -6.94]) # Mulu, Borneo
    #week_d180 = np.array([-19.40, -19.4, -19.4, -19.4, -15.9, -15.9, -15.9, -23.1, -23.1, -23.1, -23.1, -23.1, -23.1, -23.1, -16.5, -16.5, -8.8, -8.8, -10.6, -10.6, -2.5, -9.3, -6.7, -8.2, -1.6, -6, -7, -4.4, -8.8, -6.5, -6.1, -6.1, -6.1, -0.6, 1.7, -4.5, -4.5, -4.5, -12.4, -12.4, -9.7, -12.2, -12.2, -12.2, -15.1, -15.1, -11, -11, -11, -30.5, -30.5, -30.5])  # North Platte Nebraska
    month_d180 = np.array([-18.50, -17.93, -12.16, -12.08, -6.88, -7.00, -7.49, -5.60, -8.87, -13.91, -14.20, -23.70]) # North Platte, Nebraska

    # np.concatenate((month_d180,month_d180)), month_d180[:24]
    water_hist = spline_input_signal(np.concatenate((month_d180,month_d180)), 30, 1)

    # Small tooth model generation
    tooth_model_sm = tooth_model.downsample_model((isomap_shape[0]+5, isomap_shape[1]+5), 1)

    # Set keyword arguments to be used in fitting procedure

    fit_kwargs['tooth_model'] = tooth_model_sm
    fit_kwargs['data_isomap'] = data_isomap
    fit_kwargs['isomap_shape'] = isomap_shape
    fit_kwargs['isomap_data_x_ct'] = isomap_data_x_ct

    # Synthetic signal production

    bg_360 = 4.*np.sin((2*np.pi/360.)*(np.arange(600.)))-11.
    bg_360_90 = 2.*np.sin((2*np.pi/90.)*(np.arange(600.))) + bg_360

    sm_360 = 2.*np.sin((2*np.pi/360.)*(np.arange(600.)))-11.
    sm_180 = 2.*np.sin((2*np.pi/180.)*(np.arange(600.)))-11.
    sm_090 = 2.*np.sin((2*np.pi/90.)*(np.arange(600.)))-11.
    sm_045 = 2.*np.sin((2*np.pi/45.)*(np.arange(600.)))-11.

    sm_360_180 = (1.0*np.sin((2*np.pi/180.)*(np.arange(600.)))) + sm_360
    sm_360_90 = (1.0*np.sin((2*np.pi/90.)*(np.arange(600.)))) + sm_360
    sm_360_45 = (1.0*np.sin((2*np.pi/45.)*(np.arange(600.)))) + sm_360
    sm_180_90 = (1.0*np.sin((2*np.pi/90.)*(np.arange(600.)))) + sm_180
    sm_180_45 = (1.0*np.sin((2*np.pi/45.)*(np.arange(600.)))) + sm_180

    sin_360 = 10.*np.sin((2*np.pi/360.)*(np.arange(600.)))-11.
    sin_180 = 10.*np.sin((2*np.pi/180.)*(np.arange(600.)))-11.
    sin_090 = 10.*np.sin((2*np.pi/90.)*(np.arange(600.)))-11.
    sin_045 = 10.*np.sin((2*np.pi/45.)*(np.arange(600.)))-11.

    sin_360_180 = (5.*np.sin((2*np.pi/180.)*(np.arange(600.)))) + sin_360
    sin_360_90 = (5.*np.sin((2*np.pi/90.)*(np.arange(600.)))) + sin_360
    sin_360_45 = (5.*np.sin((2*np.pi/45.)*(np.arange(600.)))) + sin_360
    sin_180_90 = (5.*np.sin((2*np.pi/90.)*(np.arange(600.)))) + sin_180
    sin_180_45 = (5.*np.sin((2*np.pi/45.)*(np.arange(600.)))) + sin_180

    number = 'bg_360_90'

    # Make water, blood and PO4 history from synthetic water input
    forward_metabolic_kw = kwargs.get('metabolic_kw', {})
    water_hist = sm_360_90 # <----- ******** WATER HISTORY HERE *********
    days = np.arange(84., len(water_hist)+84.)
    blood_hist = blood_delta(23.5, water_hist, 25.3, **forward_metabolic_kw)
    PO4_hist = PO4_dissoln_reprecip(3., 34.5, .3, blood_hist, **kwargs)

    # Convert to M1 timing and space
    #m2days, m2water_hist, m2blood_hist, m2PO4_hist = days[84:], water_hist[84:], blood_hist[84:], PO4_hist[84:]
    m1_days = tooth_timing_convert(days, *m2_m1_params)
    m1_days = m1_days - m1_days[0]
    days_tmp, water_tmp, blood_tmp, PO4_tmp = np.ones(m1_days.size), np.ones(m1_days.size), np.ones(m1_days.size), np.ones(m1_days.size)
    for k,d in enumerate(m1_days):
        d = int(d)
        water_tmp[d:],blood_tmp[d:],PO4_tmp[d:] = water_hist[k], blood_hist[k], PO4_hist[k]
    m1water_hist, m1blood_hist, m1PO4_hist = water_tmp, blood_tmp, PO4_tmp
    print 'M2 water hist = ', water_hist
    print 'M1 water hist = ', m1water_hist

    #TESTING TO MAKE SURE ALL'S WORKING
    #fig = plt.figure()
    #ax1 = fig.add_subplot(1,1,1)
    #days = np.arange(sin_180.size)
    #ax1.plot(days, sin_180, 'k--', linewidth=1.0)
    #ax1.plot(days, m1water_hist, 'b-', linewidth=2.0)
    #ax1.plot(days, m1blood_hist, 'r-', linewidth=2.0)
    #ax1.plot(days, m1PO4_hist, 'g-.', linewidth=1.0)
    #plt.show()

    #return 0

    # Create M1 isomaps
    #blood_model = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model_sm, m1blood_hist)
    PO4_model = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model_sm, m1PO4_hist)

    '''
    # Generate blood d18O and tooth d18O isomap from drinking water history, with score compared to measured data
    score_sm_w, model_isomap_sm_w = water_hist_likelihood(water_spl, **fit_kwargs)
    score_i33, model_isomap_i33 = water_hist_likelihood(i33_spl, **fit_kwargs)
    score_i50, model_isomap_i50 = water_hist_likelihood(i50_spl, **fit_kwargs)
    score_i66, model_isomap_i66 = water_hist_likelihood(i66_spl, **fit_kwargs)
    mu_sm_w = np.median(model_isomap_sm_w, axis=2)
    sigma_sm_w = np.std(model_isomap_sm_w, axis=2)
    sigma_sm_w = np.sqrt(sigma_sm_w**2. + 0.15**2 + 0.05**2)
    #resid_img = (mu - data_isomap) / sigma

    # Generate tooth isomap from blood measurements
    forward_metabolic_kw = kwargs.get('metabolic_kw', {})
    forward_phosphate_eq_b_PO4 = PO4_dissoln_reprecip(3., 34.5, .3, blood_spl, **kwargs)
    forward_blood_model = gen_isomaps(isomap_shape, isomap_data_x_ct, tooth_model_sm, forward_phosphate_eq_b_PO4) # This takes the blood history from 962 scaled to the M1 without also downscaling the blood turnover

    plt.plot(m1_days_spl, water_spl, 'b', lw=2)
    plt.plot(m1_days_spl, blood_spl, 'r', lw=2)
    plt.plot(m1_days_spl, forward_phosphate_eq_b_PO4, 'g', lw=2)
    plt.show()

    m_mu_sm = np.ma.masked_array(mu_sm_w, np.isnan(mu_sm_w))
    mu_sm_r = np.mean(m_mu_sm, axis=1)
    small_sample = np.ones(int(mu_sm_r.size/2))
    for k,d in enumerate(small_sample):
        small_sample[k] = (mu_sm_r[(k*2)]+mu_sm_r[(k*2)+1])/2
    mu_sm_r = small_sample
    mu_sm_r.shape = (mu_sm_r.size, 1)
    print mu_sm_r.shape

    # Save generated isomap as CSV file
    #save_tooth = mu_sm.T
    #save_tooth[np.isnan(save_tooth)] = 0.
    #np.savetxt('m2_predicted_a_hist_half=3.csv', np.flipud(save_tooth), delimiter=',', fmt='%.2f')

    mu_sm_w[mu_sm_w==0.] = np.nan

    print score_sm_w
    textstr = '%.1f' % score_sm_w
    '''

    t_save = time()

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 4}

    matplotlib.rc('font', **font)

    save_PO4_array = np.flipud(np.mean(PO4_model, axis=2).T)
    #save_blood_array = np.flipud(np.mean(blood_model, axis=2).T)
    save_PO4_array[np.isnan(save_PO4_array)] = 0.00
    #save_blood_array[np.isnan(save_blood_array)] = 0.00
    np.savetxt('PO4_{0}_{1}.csv'.format(number, t_save), save_PO4_array, delimiter=',', fmt='%.2f')
    #np.savetxt('blood_{0}_{1}.csv'.format(number, t_save), save_blood_array, delimiter=',', fmt='%.2f')

    fig = plt.figure(figsize=(2,2), dpi=300)
    ax1 = fig.add_subplot(2,1,1)
    ax1text = 'M2->M1 PO4_{0}'.format(number)
    ax1.text(4, 3, ax1text, fontsize=4)
    cimg1 = ax1.imshow(np.mean(PO4_model, axis=2).T, aspect='equal', interpolation='nearest', origin='lower', cmap='bwr')
    cax1 = fig.colorbar(cimg1)
    #ax2 = fig.add_subplot(2,1,2)
    #ax2text = 'M2->M1 Blood_{0}'.format(number)
    #ax2.text(4, 3, ax2text, fontsize=4)
    #cimg2 = ax2.imshow(np.mean(blood_model, axis=2).T, aspect='equal', interpolation='nearest', origin='lower', cmap='bwr')
    #cax2 = fig.colorbar(cimg2)

    fig.savefig('PO4_and_Blood_{0}_{1}a.svg'.format(number, t_save), dpi=300)
    #plt.show()

    fig = plt.figure(figsize=(2,2), dpi=300)
    ax1 = fig.add_subplot(2,1,1)
    ax1text = 'M2->M1 PO4_and_Blood_{0}'.format(number)
    ax1.text(19, -20, ax1text, fontsize=4)
    ax1.plot(days, water_hist, 'b-', linewidth=1.0)
    ax1.plot(days, blood_hist, 'r-', linewidth=1.0)
    ax1.plot(days, PO4_hist, 'g-.', linewidth=1.0)
    ax2 = fig.add_subplot(2,1,2)
    ax2text = 'M2->M1 PO4_and_Blood_{0} in M1 timing'.format(number)
    ax2.text(19, -20, ax2text, fontsize=4)
    ax2.plot(days, m1water_hist, 'b-', linewidth=1.0)
    ax2.plot(days, m1blood_hist, 'r-', linewidth=1.0)
    ax2.plot(days, m1PO4_hist, 'g-.', linewidth=1.0)
    fig.savefig('PO4_and_Blood_{0}_{1}b.svg'.format(number, t_save), dpi=300)
    #plt.show()

    '''
    ax1 = fig.add_subplot(6,1,1)
    ax1text = 'forward model from water data 100% ice'
    ax1.text(19, 3, ax1text, fontsize=8)
    cimg1 = ax1.imshow(mu_sm_w.T, aspect='equal', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap='bwr')
    cax1 = fig.colorbar(cimg1)

    ax2 = fig.add_subplot(6,1,2)
    ax2text = 'data'
    ax2.text(19, 3, ax2text, fontsize=8)
    cimg2 = ax2.imshow(data_isomap.T, aspect='equal', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap='bwr')
    cax2 = fig.colorbar(cimg2)

    ax3 = fig.add_subplot(6,1,3)
    ax3text = 'forward model from blood data'
    ax3.text(19, 3, ax3text, fontsize=8)
    cimg3 = ax3.imshow(np.mean(forward_blood_model, axis=2).T, aspect='equal', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap='bwr')
    cax3 = fig.colorbar(cimg3)

    ax4 = fig.add_subplot(6,1,4)
    ax4text = 'forward model 33% ice'
    ax4.text(19, 3, ax4text, fontsize=8)
    cimg4 = ax4.imshow(np.mean(model_isomap_i33, axis=2).T, aspect='equal', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap='bwr')
    cax4 = fig.colorbar(cimg4)

    ax5 = fig.add_subplot(6,1,5)
    ax5text = 'forward model 50% ice'
    ax5.text(19, 3, ax5text, fontsize=8)
    cimg5 = ax5.imshow(np.mean(model_isomap_i50, axis=2).T, aspect='equal', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap='bwr')
    cax5 = fig.colorbar(cimg5)

    ax6 = fig.add_subplot(6,1,6)
    ax6text = 'forward model 66% ice'
    ax6.text(19, 3, ax6text, fontsize=8)
    cimg6 = ax6.imshow(np.mean(model_isomap_i66, axis=2).T, aspect='equal', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap='bwr')
    cax6 = fig.colorbar(cimg6)

    t_save = time()
    fig.savefig('spline_water_snow_2015_11_14_18p6_w_ice_blood_w_PO4.svg'.format(t_save), dpi=300)
    plt.show()

    r_mu_sm = np.ravel(mu_sm)
    r_mu_sm = r_mu_sm[~np.isnan(r_mu_sm)]

    r_mu_fl = np.ravel(mu_fl)
    r_mu_fl = r_mu_fl[~np.isnan(r_mu_fl)]

    small_large_diff_pct = (np.max(r_mu_sm)-np.min(r_mu_sm)) / (np.max(r_mu_fl)-np.min(r_mu_fl)) * 100.
    print small_large_diff_pct

    plt.hist(mu_sm_r, bins=(np.linspace(6., 16., 24)), histtype='stepfilled', normed=True, color='#0040FF', alpha=.8, label='Low')
    #plt.hist(r_mu_sm, bins=(np.linspace(8., 15., 25)), histtype='stepfilled', normed=True, color='#0040FF', alpha=1, label='Medium')
    plt.hist(r_mu_fl, bins=(np.linspace(6., 16., 24)), histtype='stepfilled', normed=True, color='#FF0040', alpha=.8, label='High')
    plt.xlabel(r'$\delta ^{18} \mathrm{O}$')
    plt.ylabel('Percent pixels')
    plt.legend(loc='upper left')
    plt.savefig('hist_13pm_60d@80d_april_2015_%.2f.pdf' % small_large_diff_pct, dpi=300)
    plt.show()
    '''


def main():
    fit_tooth_data('/Users/darouet/Desktop/tooth_example.csv')

    '''
    #print 'importing isotope data...'
    #iso_shape, iso_data, iso_data_x_ct = import_iso_data()

    print 'loading tooth model ...'
    tooth_model = ToothModel('equalsize_jul2015a.h5')

    #tooth_model_sm = tooth_model.downsample_model((iso_shape[0]+5, iso_shape[1]+5), 1)

    print 'Generating movies...'
    gen_mnzt_movie(tooth_model, 'frames/fullres')
    #gen_mnzt_movie(tooth_model_sm, 'frames/50x30')

    print 'importing blood isotope history...'
    water_step, blood_step = calc_blood_step()

    model_isomap, data_isomap, remodeled = gen_isomaps(iso_shape, iso_data, iso_data_x_ct, tooth_model_sm, blood_step)
    model_isomap = np.array(model_isomap)

    score = compare(remodeled, data_isomap)
    print 'score = ', score

    print 'plotting figures...'
    fig = plt.figure(dpi=100)
    ax1 = plt.subplot(3,1,1)
    cimg1 = ax1.imshow(model_isomap[1,:,:].T, aspect='auto', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap=plt.get_cmap('bwr'))
    cax1 = fig.colorbar(cimg1)
    ax2 = plt.subplot(3,1,2)
    cimg2 = ax2.imshow(data_isomap.T, aspect='auto', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap=plt.get_cmap('bwr'))
    cax2 = fig.colorbar(cimg2)
    ax3 = plt.subplot(3,1,3)
    cimg3 = ax3.imshow(remodeled[:,:,1].T, aspect='auto', interpolation='nearest', origin='lower', vmin=9., vmax=15., cmap=plt.get_cmap('bwr'))
    cax2 = fig.colorbar(cimg2)
    plt.show()
    '''
    #gen_min_movie(tooth_model)
    #gen_isomap_movie(tooth_model_sm, blood_step)


    return 0

if __name__ == '__main__':
    main()




