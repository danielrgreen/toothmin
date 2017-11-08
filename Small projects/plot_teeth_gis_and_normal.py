# Daniel Green 2017

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.mlab import griddata
from matplotlib import mlab, cm
import h5py
import nlopt
from PIL import Image

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.misc import imresize
import scipy.special as spec
from time import time

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

    return iso_data, iso_shape, iso_data_x_ct

def plot_tooth_data(data_fname, color):
    '''
    '''

    print 'importing isotope data...'
    data_isomap, isomap_shape, isomap_data_x_ct = load_iso_data(data_fname)
    #z = np.ravel(data_isomap)
    data_isomap = np.rot90(data_isomap)

    #coolwarm and summer

    temp_color = 'plt.cm.{0}'.format(color)
    print temp_color
    x = np.arange(isomap_shape[0])
    y = np.arange(isomap_shape[1])
    cs = plt.contourf(x, y, np.flipud(data_isomap), 40,  vmin=12.5, vmax=16.,
                  cmap=color,
                  origin='lower',
                  aspect='equal')

    #cs2 = plt.contour(cs, levels=cs.levels[::1],
    #              colors='k',
    #              origin='lower')

    cbar = plt.colorbar(cs)
    #cbar.add_lines(cs2)
    t = time()
    data_name = data_fname[-11:-4]
    print data_name

    plt.savefig('gis_{0}_{1}.svg'.format(t, data_name), aspect='equal', dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    cimg1 = ax1.imshow(np.rot90(data_isomap.T), aspect='auto', interpolation='nearest', origin='lower', cmap=color, vmin=12.5, vmax=16.)
    cax1 = fig.colorbar(cimg1)

    ax2 = fig.add_subplot(2,1,2)
    cimg2 = ax2.imshow(np.rot90(data_isomap.T), aspect='equal', interpolation='nearest', origin='lower', cmap=color, vmin=12.5, vmax=16.)
    cax2 = fig.colorbar(cimg2)
    plt.savefig('regular_{0}_{1}.svg'.format(t, data_name), aspect='equal', dpi=300, bbox_inches='tight')

def main():

    color = 'bwr'
    plot_tooth_data('/Users/darouet/Documents/code/mineralization/clean code/949_dats.csv', color)

    return 0

if __name__ == '__main__':
    main()


