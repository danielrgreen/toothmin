#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  small_make_min_model.py
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
import scipy.interpolate as interp
import scipy.special
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import spline
from numpy import linspace
import scipy.ndimage.interpolation as imginterp
import scipy.ndimage.filters as filters
from os.path import abspath, expanduser
import os, os.path
from datetime import datetime
import argparse, sys, warnings
from matplotlib.ticker import FuncFormatter
import time
import h5py
import emcee
from fit_positive_function import TMonotonicPointModel

# user inputs

voxelsize = 46. # voxel resolution of your scan in microns?
species = 'Ovis_aries' # the species used for this model
calibration_type = 'undefined' # pixel to density calibration method
y_resampling = 0.1 # y pixel size of model compared to voxel size
x_resampling = 2.0 # x pixel size of model compared to voxel size

# Load image
    
def load_image(fname):
    img = plt.imread(abspath(fname))
    return img

# Define x and y, locate edges

#shape is an attribute for numpy arrays, returning the array dimensions.
#If an array has n rows and m columns, array.shape is (n,m).
#n and m would therefore represent y and x coordinates.

#Input: img  numpy array with shape=(Nx, Ny)
#Output: x    array of x-values of pixels used to fit baseline
#spl  list with two splines: one for each edge.
#This object can be called as follows:
#spl[0](x) = y-value of edge #1 at x
#spl[1](x) = y-value of edge #2 at x


def get_baseline(img, exactness=15.):
    '''
    This function takes an image array of shape (Nx,Ny),
    creates a float array to record the upper, lower enamel edges,
    locates nonzero pixels to find the enamel in the image and
    makes arrays for the upper and lowwer enamel edges.
    '''
    Ny, Nx = img.shape
    edge = np.empty((2,Nx), dtype='i4')
    edge[:,:] = -1
    mask = (img > 0.)
    for i in xrange(Nx):
        nonzero = np.where(mask[:,i])[0]
        if len(nonzero):
            edge[0,i] = np.min(nonzero)
            edge[1,i] = np.max(nonzero)

    # Clip edges to region with nonzero image
    
    isReal = np.where(edge[0] >= 0)[0]
    xMin, xMax = isReal[0], isReal[-1]
    edge = edge[:,xMin:xMax+1]
    x = np.linspace(xMin, xMax, xMax-xMin+1)    
    
    # Fit splines to edges

    spl = []
    w = np.ones(len(x))
    w[0] = 10.
    w[-1] = 10.
    for i in xrange(2):
        spl.append( interp.UnivariateSpline(x, edge[i], w=w, s=len(w)/exactness) )
    
    return x, spl

# Create markers and calculate distance along spline

def place_markers(x, spl, spacing=2.):
    fineness = 10
    xFine = np.linspace(x[0], x[-1], fineness*len(x))
    yFine = spl(xFine)
    derivFine = np.empty(len(xFine), dtype='f8')
    for i,xx in enumerate(xFine):
        derivFine[i] = spl.derivatives(xx)[1]
    derivFine = filters.gaussian_filter1d(derivFine, fineness*spacing)
    dx = np.diff(xFine)
    dy = np.diff(yFine)
    dist = np.sqrt(dx*dx + dy*dy)
    dist = np.cumsum(dist)
    nMarkers = int(dist[-1] / spacing)
    if nMarkers > 1e5:
		raise ValueError('nMarkers unreasonably high. Something has likely gone wrong.')
    markerDist = np.linspace(spacing, spacing*nMarkers, nMarkers)
    markerPos = np.empty((nMarkers, 2), dtype='f8')
    markerDeriv = np.empty(nMarkers, dtype='f8')
    cellNo = 0
    for i, (xx, d) in enumerate(zip(xFine[1:], dist)):
        if d >= (cellNo+1) * spacing:
            markerPos[cellNo, 0] = xx
            markerPos[cellNo, 1] = spl(xx)
            markerDeriv[cellNo] = derivFine[i+1] #spl.derivatives(xx)[1]
            cellNo += 1
    
    return markerPos, markerDeriv

def get_image_values_2(img, markerPos, DeltaMarker, step=y_resampling):
    ds = np.sqrt(DeltaMarker[:,0]*DeltaMarker[:,0] + DeltaMarker[:,1]*DeltaMarker[:,1])
    nSteps = img.shape[0] / step
    stepSize = step / ds
    n = np.linspace(0., nSteps, nSteps+1)
    sampleOffset = np.einsum('i,ik,j->ijk', stepSize, DeltaMarker, n)
    sampleStart = np.empty(markerPos.shape, dtype='f8')
    sampleStart[:,:] = markerPos[:,:]
    nMarkers, tmp = markerPos.shape
    sampleStart.shape = (nMarkers, 1, 2)
    sampleStart = np.repeat(sampleStart, nSteps+1, axis=1)
    samplePos = sampleStart + sampleOffset

    samplePos.shape = (nMarkers*(nSteps+1),2)
    resampImg = imginterp.map_coordinates(img.T, samplePos.T, order=1)
    resampImg.shape = (nMarkers, nSteps+1)
    
    return resampImg.T[:,:]


def min_pct_prior(log_Delta_y):
    y_final = np.sum(np.exp(log_Delta_y))

    return 1. - scipy.special.erf(1.25 * (y_final - 0.8) / 0.075)


def lnprob(log_Delta_y, y_obs, y_sigma, mu_prior, sigma_prior):
    y_mod = np.cumsum(np.exp(log_Delta_y))
    
    Delta_y = (y_mod - y_obs) / y_sigma
    
    log_likelihood = -0.5 * np.sum(Delta_y * Delta_y)
    log_prior = np.sum(log_Delta_y)
    
    Delta_y = (log_Delta_y - mu_prior) / sigma_prior
    log_prior -= 0.5 * np.sum(Delta_y * Delta_y)
    
    log_prior += np.log(1. - scipy.special.erf(1.25 * (y_mod[-1] - 0.8) / 0.075))
    
    return log_likelihood + log_prior


# Main section of code in which defined functions are used
'''
Loads image
'''

def main():
    parser = argparse.ArgumentParser(prog='enamelsample',
                  description='Resample image of enamel to standard grid',
                  add_help=True)
    parser.add_argument('images', type=str, nargs='+', help='Images of enamel.')
    parser.add_argument('-sp', '--spacing', type=float, default=x_resampling, help='Marker spacing (in pixels).')
    parser.add_argument('-ex', '--exact', type=float, default=10., help='Exactness of baseline spline.')
    parser.add_argument('-l', '--order-by-length', action='store_true',
                              help='Order teeth according to length.')
    parser.add_argument('-s', '--show', action='store_true', help='Show plot.')
    parser.add_argument('-o', '--output-dir', type=str, default='.',
                              help='Directory in which to store output.')
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])

    warnings.simplefilter('ignore')
    
    # Load and standardize each tooth image
    alignedimg = []
    Nx, Ny = [], []
    age = []
    
    for i,fname in enumerate(args.images):
        print 'Processing %s ...' % fname
        
        img = load_image(fname)
        
        # Extract age from filename
        age.append(float(fname[1:4]))
        
        # Generate a spline for each tooth edge
        
        x, spl = get_baseline(img, exactness=args.exact)
        
        # Place makers along the bottom edge
        
        markerPos, markerDeriv = place_markers(x, spl[1], spacing=args.spacing)
        
        # Calculate y values of edges
        
        y = []
        for i in xrange(2):
            y.append(spl[i](x))
        
        # Calculate perpendicular lines to edges
        # Height of line is # in markerPos + #
    
        DeltaMarker = -np.ones((len(markerDeriv), 2), dtype='f8')
        DeltaMarker[:,0] = markerDeriv[:]
        markerPos2 = markerPos + 80. * DeltaMarker
        
        
        # Goal: take our x positions, step along the y positions, and get the
        # values from the image.
        # Plot everything
      
        # Resample image to standard grid
        alignedimg.append(get_image_values_2(img, markerPos, DeltaMarker))
        
        # Keep track of shapes of images
        tmp_y, tmp_x = alignedimg[-1].shape
        Nx.append(tmp_x)
        Ny.append(tmp_y)
            
    # Sort images by length
    if args.order_by_length:
        #nonzero = [np.sum(img > 0.) for img in alignedimg]
        Nx = np.array(Nx)
        Ny = np.array(Ny)
        age = np.array(age, dtype='f8')
        
        idx = np.argsort(Nx, kind='mergesort')

        print 'Rearranging using:'
        print idx
        print ''
        
        alignedimgSorted = [alignedimg[i] for i in idx]
        alignedimg = alignedimgSorted
        Nx = Nx[idx]
        Ny = Ny[idx]
        age = age[idx]

    # Combine images into one 3-dimensional array
    nImages = len(alignedimg)
    imgStack = np.zeros((nImages, max(Nx), max(Ny)), dtype='f8')
    for i,img in enumerate(alignedimg):
        imgStack[i, :Nx[i], :Ny[i]] = img.T[:,:]

    # Convert from pixel value to total density
    imgStack *= 2.**16
    imgStack *= 0.000182009
    imgStack -= 0.077402903
    #imgStack[imgStack < .79] = np.nan

    # Convert from total density to mineral fraction by weight
    imgStack = (3.15 - 3.15 * 0.79 / imgStack) / (3.15 - 0.79)
    idx = (imgStack < 0.05) | (imgStack > 1.2)
    imgStack[idx] = np.nan
    
    # Relate image length to day: output for time x-axis is 'age_plt'
    age_coeff = np.polyfit(Nx, age, 5)
    
    Nx_age = np.zeros(Nx.size, dtype='f8')
    for i in xrange(len(age_coeff)):
        Nx_age += age_coeff[i] * Nx**(len(age_coeff)-i-1)

    print 'Real ages:'
    print age
    print ''
    print 'Ages based on length:'
    print Nx_age
    print ''
    
    # Make Nx_age monotonically increasing
    #increasing = np.linspace(0., 0.1, len(Nx_age))
    #Nx_age = Nx_age + increasing

    Nx_age = np.around(Nx_age)
    Nx_age[Nx_age < 1.] = 1
    
    for k in xrange(1, len(Nx_age)):
        if Nx_age[k] <= Nx_age[k-1]:
            Nx_age[k] = Nx_age[k-1] + 1

    Nx_age = Nx_age.astype('u2')

    print 'Corrected ages based on length:'
    print Nx_age
    print ''
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for n in xrange(100):
        x = np.random.randint(5, 30)
        y = np.random.randint(11, 20)
        m = imgStack[:, x, y]

        idx = (m > 1.)
        m[idx] = np.nan
        
        ax.plot(Nx_age, m, alpha=0.5)

    plt.show()

    s = imgStack.shape
    imgStack.shape = (s[0], s[1]*s[2])
    
    img_monotonic = np.empty(imgStack.shape, dtype='f8')
    img_monotonic[:] = -np.inf
    
    img_monotonic[0, :] = imgStack[0, :]
    idx = ~np.isfinite(img_monotonic[0, :])
    img_monotonic[0, idx] = -np.inf

    for n in xrange(1, s[0]):
        idx = (imgStack[n,:] > img_monotonic[n-1,:]) & np.isfinite(imgStack[n,:])
        img_monotonic[n, idx] = imgStack[n, idx]
        img_monotonic[n, ~idx] = img_monotonic[n-1, ~idx]

    #img_monotonic[~np.isfinite(img_monotonic)] = np.nan
    imgStack.shape = s
    img_monotonic.shape = s

    idx = ~np.isfinite(img_monotonic)
    img_monotonic[idx] = 0.
    
    img_mon_diff = np.diff(img_monotonic, axis=0)
    img_mon_diff = np.concatenate([np.reshape(img_monotonic[0], (1, s[1], s[2])),
                                   img_mon_diff], axis=0)

    img_monotonic[idx] = np.nan
    img_mon_diff[idx] = np.nan

    print 'Ex. img/diff:'
    print img_monotonic[:,10,10]
    print img_mon_diff[:,10,10]

    print ''

    print img_monotonic[:,20,20]
    print img_mon_diff[:,20,20]

    for n, (t, img) in enumerate(zip(Nx_age, img_mon_diff)):
        fig = plt.figure(figsize=(10,3), dpi=300)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(img.T, origin='lower', aspect='auto',
                         vmin=0., vmax=1., interpolation='none')
        fig.suptitle(r'$t = %d \ \mathrm{days}$' % t, fontsize=18)
        fig.savefig('%s/img_mon_diff_%.3d.png' % (args.output_dir, t), dpi=300)

    '''
    dirname = args.output_dir

    if dirname == None:
        return 0

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    for a, img in zip(Nx_age, imgStack):
        fig = plt.figure(figsize=(8,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(img.T, origin='lower', aspect='auto',
                       vmin=0., vmax=1., interpolation='none')
        fig.suptitle(r'$\mathrm{Age} = %d \ \mathrm{days}$' % a,
                     fontsize=24)
        fig.savefig('%s/img_raw_%.2d.png' % (dirname, a), dpi=300)
        plt.close(fig)

    for a, img in zip(Nx_age, img_monotonic):
        fig = plt.figure(figsize=(8,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(img.T, origin='lower', aspect='auto',
                       vmin=0., vmax=1., interpolation='none')
        fig.suptitle(r'$\mathrm{Age} = %d \ \mathrm{days}$' % a,
                     fontsize=24)
        fig.savefig('%s/img_monotonic_%.2d.png' % (dirname, a), dpi=300)
        plt.close(fig)

    for a, img in zip(Nx_age, img_mon_diff):
        fig = plt.figure(figsize=(8,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(img.T, origin='lower', aspect='auto',
                       vmin=0., vmax=1., interpolation='none')
        fig.suptitle(r'$\mathrm{Age} = %d \ \mathrm{days}$' % a,
                     fontsize=24)
        fig.savefig('%s/img_monotonic_diff_%.2d.png' % (dirname, a), dpi=300)
        plt.close(fig)

    
    # Save images to HDF5 file
    f = h5py.File(dirname + '/simple_fit2.h5', 'w')

    dset = f.create_dataset('img_raw', shape=imgStack.shape, dtype='f4',
                                       compression='gzip', compression_opts=9)
    dset[:] = imgStack[:]

    dset = f.create_dataset('img_mon', shape=img_monotonic.shape, dtype='f4',
                                       compression='gzip', compression_opts=9)
    dset[:] = img_monotonic[:]

    dset = f.create_dataset('img_mon_diff', shape=img_mon_diff.shape, dtype='f4',
                                       compression='gzip', compression_opts=9)
    dset[:] = img_mon_diff[:]

    dset = f.create_dataset('age', shape=Nx_age.shape, dtype='u2')
    dset[:] = Nx_age[:]

    f.close()

    # calculate model information for save file

    xpixelsize = voxelsize * args.spacing
    ypixelsize = voxelsize * y_resampling
    tooth_length = xpixelsize * s[1]
    tooth_height = ypixelsize * s[2]

    # Save text file with model information
    txt = 'date: %s\n' % (datetime.now().strftime('%c'))
    txt += 'species: %s\n' % (species)
    txt += 'x_pix_size: %.3f\n' % (xpixelsize)
    txt += 'y_pix_size: %.3f\n' % (ypixelsize)
    txt += 'tooth_length: %.3f\n' % (tooth_length)
    txt += 'tooth_height: %.3f\n' % (tooth_height)
    txt += 'image_shape: %d x %d\n' % (s[1], s[2])
    txt += 'n_time_samples: %d\n' % (s[0])
    
    f = open(dirname + '/info.txt', 'w')
    f.write(txt)
    f.close()
    '''
    
    return 0

if __name__ == '__main__':
    main()




