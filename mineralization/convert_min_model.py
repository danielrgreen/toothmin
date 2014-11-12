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


class ToothModel:
    def __init__(self, fname=None):
        if fname == None:
            return
        
        self._load_toothmodel(fname)
        self._interp_missing_ages()

    def _load_toothmodel(self, fname):
        f = h5py.File('final.h5', 'r')
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
        print 'pix_val.shape =', pix_val.shape
        
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

        print 'idx0.shape =', idx0.shape
        print 'idx1.shape =', idx1.shape
        print 'pix_val.shape =', pix_val.shape
        
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
        img_sm = np.empty((shape[0], shape[1], n_samples, self.n_ages), dtype='f8')
        
        for n in xrange(n_samples):
            img = self.gen_mnzt_image(mode='sample')

            for t in xrange(self.n_ages):
                img_sm[:,:,n,t] = self._resize(img[:,:,t], shape)

        img_sm.shape = (shape[0]*shape[1], n_samples, self.n_ages)

        locations = np.indices(shape)
        locations.shape = (2, shape[0]*shape[1])
        locations = np.swapaxes(locations, 0, 1)
        
        tmodel = ToothModel()
        tmodel.pct_min_interp = img_sm[:,:,:]
        tmodel.locations = locations[:,:]
        tmodel.ages = self.ages[:]
        tmodel.n_pix = shape[0] * shape[1]
        tmodel.n_ages = self.n_ages
        tmodel.n_samples = n_samples

        return tmodel
    
    def gen_isotope_image(self, blood_hist, mode='sample'):
        idx_mask = np.isnan(self.pct_min_interp)
        pct_min_interp = np.ma.array(self.pct_min_interp, mask=idx_mask, fill_value=0.)
        pct_min_diff = diff_with_first(pct_min_interp[:,:,:].filled(), axis=2)

        n_days = blood_hist.size
        pct_min_diff_days = np.empty((self.n_pix, self.n_samples, n_days), dtype='f8')
        pct_min_diff_days[:] = np.nan
        
        for k,(a1,a2) in enumerate(zip(self.ages[:-1], self.ages[1:])):
            if a1 > n_days:
                break

            dt = a2 - a1
            
            if a2 > n_days:
                a2 = n_days
            
            pct_min_diff_days[:,:,a1:a2] = (pct_min_diff[:,:,k] / dt)[:,:,np.newaxis]

        print 'frost', pct_min_diff_days

        pct_min_diff_days[np.isnan(pct_min_diff_days)] = 0.
        pct_min_days = np.cumsum(pct_min_diff_days, axis=2)
        pct_min_days[pct_min_days==0.] = np.nan

        print 'stevens', pct_min_days

        print pct_min_diff_days.shape
        print pct_min_days.shape
        
        isotope = np.cumsum(
            blood_hist[np.newaxis, np.newaxis, :]
            * pct_min_diff_days,
            axis=2
        )
        
        isotope /= pct_min_days

        print isotope.shape
        print self.locations.shape
        
        return self._pix2img(isotope, mode=mode, interp=False)


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

    img_interp = toothmodel.gen_mnzt_image(interp=True, mode='sample')

    ages = np.arange(toothmodel.ages[0], toothmodel.ages[-1]+1)

    img = img_interp(ages[-1])
    shape = img.shape[:2]
    
    img = np.empty((ages.size, shape[0], shape[1]), dtype='f8')

    for k,t in enumerate(ages):
        img[k] = img_interp(t)

    img = np.diff(img, axis=0)
    sigma_t = 10
    sigma_x, sigma_y = 2, 2
    img = gaussian_filter(img, (sigma_t,sigma_x,sigma_y), mode='nearest')
    
    idx = np.isfinite(img)
    vmax = np.percentile(img[idx], 99.8)
    #vmax = 1.

    fig = plt.figure(figsize=(6,3), dpi=100)
    ax = fig.add_subplot(1,1,1)

    im = ax.imshow(np.zeros(shape[::-1], dtype='f8'), origin='lower',
                                                interpolation='nearest',
                                                vmin=0., vmax=vmax)
    
    for k,t in enumerate(ages[:-1]):
        #img = img_interp(t)
        im.set_data(img[k].T)

        ax.set_title(r'$t = %d \ \mathrm{days}$' % t, fontsize=14)
        
        fig.savefig('october_rate_g10_x02_y02_local_k%04d.png' % k, dpi=100)

def gen_isomap_movie(toothmodel, blood_hist):
    img = toothmodel.gen_isotope_image(blood_hist, mode='sample')
    
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
        fig.savefig('november_isomaps01_03setdataT_k%04d.png' % k, dpi=100)


def main():
    #x = np.arange(20)**2
    #x.shape = (4,5)
    #y = diff_with_first(x, axis=1)
    #print x
    #print y

    print 'Loading tooth model ...'
    toothmodel = ToothModel('final.h5')
    age_max = np.max(toothmodel.ages)
    print 'Downsampling tooth model ...'
    toothmodel_sm = toothmodel.downsample_model((40,10), 1)
    blood_hist = np.zeros(age_max, dtype='f8')
    blood_hist[100:120] = 1.

    print 'Generating movie ...'
    #gen_min_movie(toothmodel_sm)
    gen_isomap_movie(toothmodel_sm, blood_hist)
    
    return 0

if __name__ == '__main__':
    main()




