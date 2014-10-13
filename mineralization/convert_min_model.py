# Daniel Green, 2014
# drgreen@fas.harvard.edu
# Human Evolutionary Biology
# Harvard University
#
# Mineralization Model Re-Size:
# this code takes a larger 
#
# 
#
#
# 

import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image

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
    def __init__(self, fname):
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
            print 'samples shape = ', samples.shape
            print 'pct min shape = ', pct_min_interp[n, :, idx].shape
            #pct_min_interp[n, :, idx] = samples[:, :]

            #x = self.ages[~idx]
            #xp = ages[idx]
            
            #for k in xrange(self.n_samples):
                #pct_min_interp[n, k, ~idx] = np.interp(x, xp, samples[k], left=0.)

        #self.pct_min_interp = pct_min_interp
    
    def _gen_rand_image(self):
        '''
        Returns a random mineralization history for each pixel.

        Output shape: (# of pixels, # of ages)
        '''
        idx0 = np.arange(self.n_pix)
        idx1 = np.random.randint(self.n_samples, size=self.n_pix)
        pct_min_rand = self.pct_min_samples[idx0, idx1, :]

        return pct_min_rand
    
    def downsample_model(self, shape):
        pass

    def calc_isomap(self, blood_hist):
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
        mnzt_rate = mnzt_rate / di_sum[:, None]
        mnzt_rate[mnzt_rate==0.] = np.nan
        d18O_addition = mnzt_rate * blood_hist[None, :]
        d18O_addition[np.isnan(d18O_addition)] = 0.
        tot_isotope = np.sum(d18O_addition, axis=1)



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

        

def main():
    toothmodel = ToothModel('final.h5')
    toothmodel = toothmodel.downsample(shape)
    isotope_pct = toothmodel.isotope_pct(bloodhist)
    diff = isotope_pct - data
    chisq = np.sum(diff**2)
    
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
    age_n = ages.size
    mean_pct = np.mean(pct_min_samples, axis=1)
    min_pctiles = np.percentile(pct_min_samples, [5, 50, 95], axis=1)
    min_pctiles = np.array(min_pctiles)

    img = np.empty((Nx, Ny, age_n), dtype='f8')
    img[:,:] = np.nan



    return 0

if __name__ == '__main__':
    main()




