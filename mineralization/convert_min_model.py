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
import h5py
from PIL import Image

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.misc import imresize
from lmfit import minimize, Parameters

iso_shape = (34, 5)

def water_hist(days=365., Am=10., P=182., offset=40., mean=-7.):
    
    days = np.arange(days)
    waterhist = Am * np.sin((days-offset) * (2*np.pi) / P) + mean
    
    return waterhist, days

def blood_hist(waterhist, feed=-8., air=-18., half_life=8.,
               feed_frac=0.3, air_frac=0.1):
    
    water_frac = 1. - feed_frac - air_frac
    air_feed = (feed_frac * feed) + (air_frac * air)
    
    remaining = .5**(1/half_life)
    bloodhist = np.empty(len(waterhist), dtype='f8')
    bloodhist[0] = water_frac*waterhist[0] + air_feed
    
    for k in xrange(1, len(waterhist)):
        bloodhist[k] = remaining*bloodhist[k-1] + (1-remaining)*(water_frac*waterhist[k]+air_feed)

    return bloodhist

def blood_pixel_mnzt(pct_min_samples, age, bloodhist):
    '''
    
    '''
    
    n_days = bloodhist.size
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
    d18O_addition = mnzt_rate * bloodhist[None, :]
    d18O_addition[np.isnan(d18O_addition)] = 0.
    tot_isotope = np.sum(d18O_addition, axis=1)

    return tot_isotope

def mnzt_all_pix(pct_min_samples, age_mask, ages, bloodhist):
    '''

    '''

    n_pix = pct_min_samples.shape[0]
    mnzt_pct = np.empty((3, n_pix), dtype='f8')

    for n in xrange(n_pix):
        samples = blood_pixel_mnzt(pct_min_samples[n],
                                   ages[age_mask[n]],
                                   bloodhist)
        mnzt_pct[:, n] = np.percentile(samples, [5., 50., 95.])
    
    return mnzt_pct


class ToothModel:
    def __init__(self, fname=None):
        if fname == None:
            return
        
        self._load_toothmodel(fname)
        self._interp_missing_ages()

    def _load_toothmodel(self, fname):
        f = h5py.File('final_equalsize_dec2014.h5', 'r')
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
        print 'pix_val.shape in pix2img =', pix_val.shape
        
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
        axis1stack = np.reshape(axis1ravel, (shape[0]*shape[1]*oldshape[0],oldshape[1]))
        m_axis1stack = np.ma.masked_array(axis1stack,np.isnan(axis1stack))
        axis1mean = np.mean(m_axis1stack, axis = 1)
        axis2stack = np.reshape(axis1mean, (shape[0]*shape[1], oldshape[0]))
        m_axis2stack = np.ma.masked_array(axis2stack,np.isnan(axis2stack))
        axis2mean = np.mean(m_axis2stack, axis=1)
        new_array = np.reshape(axis2mean, shape)

        # Add back in NaNs, threshold > 50% NaN
        nan_map = np.zeros(oldshape)
        nan_map[np.isnan(array)] = 1.
        sm_nan_map = imresize(nan_map, shape, interp='bilinear', mode='F')
        new_array[sm_nan_map >= 0.5] = np.nan
                
        return new_array
    
    def downsample_model(self, shape, n_samples):
        img_sm = np.empty((shape[1], shape[0], n_samples, self.n_ages), dtype='f8') ######## 0,1
        
        for n in xrange(n_samples):
            img = self.gen_mnzt_image(mode='sample')

            for t in xrange(self.n_ages):
                img_sm[:,:,n,t] = np.fliplr((self._resize(img[:,:,t], (shape[0], shape[1]))).T) ######## 0,1 #
                #print 'first', img_sm[:,:,n,t].shape
                #img_sm[:,:,n,t] = img_sm[:,:,n,t].T
                #print 'second', img_sm[:,:,n,t].shape

        img_sm.shape = (shape[0]*shape[1], n_samples, self.n_ages) ########## 0,1 # no effect

        locations = np.indices((shape[1], shape[0])) ######## 0,1 # shape now reversed
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

    def gen_isotope_image(self, bloodhist, mode='sample'):
        idx_mask = np.isnan(self.pct_min_interp)
        pct_min_interp = np.ma.array(self.pct_min_interp, mask=idx_mask, fill_value=0.)
        pct_min_diff = diff_with_first(pct_min_interp[:,:,:].filled(), axis=2)

        n_days = bloodhist.size
        pct_min_diff_days = np.empty((self.n_pix, self.n_samples, n_days), dtype='f8')
        pct_min_diff_days[:] = np.nan
        
        for k,(a1,a2) in enumerate(zip(self.ages[:-1], self.ages[1:])):
            if a1 > n_days:
                break

            dt = a2 - a1
            
            if a2 > n_days:
                a2 = n_days
            
            pct_min_diff_days[:,:,a1:a2] = (pct_min_diff[:,:,k] / dt)[:,:,np.newaxis]

        pct_min_diff_days[np.isnan(pct_min_diff_days)] = 0.
        pct_min_days = np.cumsum(pct_min_diff_days, axis=2)
        pct_min_days[pct_min_days==0.] = np.nan

        isotope = np.cumsum(
            bloodhist[np.newaxis, np.newaxis, :]
            * pct_min_diff_days,
            axis=2
        )
        
        isotope /= pct_min_days

    '''
    def gen_isotope_image(self, bloodhist, mode='sample'):
        idx_mask = np.isnan(self.pct_min_interp)
        pct_min_interp = np.ma.array(self.pct_min_interp, mask=idx_mask, fill_value=0.)
        pct_min_diff = diff_with_first(pct_min_interp[:,:,:].filled(), axis=2)

        n_days = bloodhist.size
        pct_min_diff_days = np.empty((self.n_pix, self.n_samples, n_days), dtype='f8')
        pct_min_diff_days[:] = np.nan
        
        for k,(a1,a2) in enumerate(zip(self.ages[:-1], self.ages[1:])):
            if a1 > n_days:
                break

            dt = a2 - a1
            
            if a2 > n_days:
                a2 = n_days
            
            pct_min_diff_days[:,:,a1:a2] = (pct_min_diff[:,:,k] / dt)[:,:,np.newaxis]

        pct_min_diff_days[np.isnan(pct_min_diff_days)] = 0.
        pct_min_days = np.cumsum(pct_min_diff_days, axis=2)
        pct_min_days[pct_min_days==0.] = np.nan

        isotope = np.cumsum(
            bloodhist[np.newaxis, np.newaxis, :]
            * pct_min_diff_days,
            axis=2
        )
        
        isotope /= pct_min_days
        
        return self._pix2img(isotope, mode=mode, interp=False)
    '''

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

def gen_min_movie(toothmodel):
    #toothmodel = toothmodel.downsample(shape)
    #isotope_pct = toothmodel.isotope_pct(bloodhist)
    #diff = isotope_pct - data
    #chisq = np.sum(diff**2)

    print 'generating mineralization movie...'

    img_interp = toothmodel.gen_mnzt_image(interp=True, mode='sample')

    ages = np.arange(toothmodel.ages[0], toothmodel.ages[-1]+1)

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
    plt.tick_params(\
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    
    im = ax.imshow(np.zeros(shape[::-1], dtype='f8'), origin='lower',
                                                interpolation='nearest',
                                                vmin=0., vmax=vmax)
    
    for k,t in enumerate(ages[:-1]):
        #img = img_interp(t)
        im.set_data(img[k].T) #was(img[k].T)

        ax.set_title(r'$t = %d \ \mathrm{days}$' % t, fontsize=14)
        
        fig.savefig('jan_rate_equalsize3_k%04d.png' % k, dpi=100)

def gen_isomap_movie(toothmodel, bloodhist):

    print 'generating movie...'
    img = toothmodel.gen_isotope_image(bloodhist, mode='sample')
    
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

def count_number(iso_data):
    '''
    '''
    iso_data = np.reshape(iso_data, (iso_shape[1],iso_shape[0]))
    iso_data = iso_data.T
    iso_data = np.fliplr(iso_data)
    iso_data_x_ct = iso_shape[1] - np.sum(np.isnan(iso_data), axis=1)

    return (iso_data, iso_data_x_ct)

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

def imresize1(x, iso_shape, method=Image.BILINEAR):
    '''
    '''
    x = x[:,:,-1] # OR SUM, ALSO BROKEN BECAUSE SUM OR MEAN TAKES NANS
    assert len(x.shape) == 2
    
    im = Image.fromarray(x)
    im_resized = im.resize(iso_shape, method)
    
    x_resized = np.array(im_resized.getdata()).reshape(iso_shape[::-1]).T
    
    return x_resized

def isotope_data(toothmodel, bloodhist):

    #Am = params['amp'].value
    #offset = params['offset'].value
    #P = params['period'].value
    #mean = params['mean'].value
    #air = params['air'].value
    #feed = params['feed'].value

    model_isomap = toothmodel.gen_isotope_image(bloodhist, mode='sample')
    x_resized = imresize1(model_isomap, iso_shape, method=Image.BILINEAR)
    
    iso_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 10.38, 13.29, 13.36, 12.85, 13.15, 12.35, 13.31, 12.89, 12.92, 13.35, 13.12, 13.21, 13.08, 13.30, 13.67, 12.45, 11.82, 11.32, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 13.00, 13.07, 13.11, 12.98, 13.20, 13.37, 13.24, 12.26, 12.61, 13.19, 12.50, 13.01, 12.75, 13.08, 12.97, 13.15, 12.52, 12.33, 12.08, 11.87, 11.07, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 12.86, 12.69, 12.95, 12.66, 12.89, 13.52, 12.47, 12.91, 12.95, 12.87, 12.41, 12.72, 12.82, 12.38, 12.44, 12.89, 11.03, 12.63, 12.99, 13.13, 12.43, 7.35, 12.10, 11.42, 12.39, 10.08])
    iso_data[iso_data==0.] = np.nan
    iso_data, iso_data_x_ct = count_number(iso_data)
    temp_x_r = np.fliplr(x_resized.T)
    model_resized = complex_resize(temp_x_r.flatten(), iso_data_x_ct)
    remodeled = np.array(model_resized)
    model_isomap = remodeled
    data_isomap = iso_data

    return model_isomap, data_isomap

def main():

    print 'Loading tooth model ...'
    toothmodel = ToothModel('final_equalsize_jan2015.h5')
    age_max = np.max(toothmodel.ages)
    print 'Downsampling tooth model ...'
    #toothmodel_sm = toothmodel.downsample_model((50,12), 1)
    #waterhist, days = water_hist()
    #bloodhist = blood_hist(waterhist)
    #bloodhist = np.zeros(days.size)
    #bloodhist[bloodhist==0] = 4.
    #bloodhist[90:100] = 8.

    #params = Parameters()
    #params.add('amp', value=10.)
    #params.add('offset', value= 40.)
    #params.add('period', value=182.)
    #params.add('mean', value=-7.)
    #params.add('air', value=-18.)
    #params.add('feed', value=-8.)

    '''
    model_isomap, data_isomap = isotope_data(toothmodel_sm, bloodhist)
    fig = plt.figure(dpi=100)
    ax1 = plt.subplot(2,1,1)
    cimg1 = ax1.imshow(model_isomap.T, aspect='auto', interpolation='nearest', origin='lower', vmin=4, vmax=5)
    cax1 = fig.colorbar(cimg1)
    ax2 = plt.subplot(2,1,2)
    cimg2 = ax2.imshow(data_isomap.T, aspect='auto', interpolation='nearest', origin='lower')
    cax2 = fig.colorbar(cimg2)
    plt.show()
    '''
    
    gen_min_movie(toothmodel)
    #gen_isomap_movie(toothmodel_sm, bloodhist)
    
    return 0

if __name__ == '__main__':
    main()




