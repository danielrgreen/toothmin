# Make_A.py
# This program is meant to create transformation matrix A
# from the general system A * m = d, using data on tooth
# mineralization as the basis for A construction. Ultimately,
# A interacts with a hydrological vector m to produce d, a
# pattern of isotope ratios across a mature tooth. Both d
# and A can be used to reconstruct ancient hydrological
# patterns.

import numpy as np
import matplotlib.pyplot as plt
import h5py
from bloodhist import calc_blood_hist
from PIL import Image

# User inputs
min_file = 'final.h5'

def load_file(fname):

    f = h5py.File(fname, 'r')
    dset1 = f['/age_mask']
    age_mask = dset1[:].astype(np.bool)
    dset2 = f['/locations']
    locations = dset2[:]
    dset3 = f['/pct_min_samples']
    pct_min_samples = dset3[:]
    dset4 = f['/ages']
    ages = dset4[:]
    f.close()

    print 'age_mask shape =', age_mask.shape
    print 'locations =', locations.shape
    print 'pct_min_samples shape =', pct_min_samples.shape
    print 'ages =', ages.size

    age_expanded = np.einsum('ij,j->ij', age_mask, ages)
    Nx, Ny = np.max(locations, axis=0) + 1
    n_pix = locations.shape[0]

    print 'age expanded size =', age_expanded.size
    print 'Nx, Ny =', Nx, Ny
    print 'n pix =', n_pix

    return age_mask, locations, pct_min_samples, ages

def main():

    fname = min_file
    age_mask, locations, pct_min_samples, ages = load_file(fname)


    return 0

if __name__ == '__main__':
    main()
