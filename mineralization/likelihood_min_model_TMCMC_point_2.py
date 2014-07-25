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
#import matplotlib
#matplotlib.use('Agg')
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
from fit_positive_function import TMonotonicPointModel, TMCMC

# user inputs

xcoordinate1 = 29
xcoordinate2 = 30
ycoordinate1 = 29
ycoordinate2 = 30

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

def get_image_values_2(img, markerPos, DeltaMarker, fname, step=y_resampling):
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
    resampImg = np.rot90(resampImg, 1)
    scan = str(fname[-5])

    # CONVERSION
    # Convert from pixel value to HAp density
    # In this case, HAp density is calculated with mu values, keV(1)=111.6
    #if scan == 'g':
        #resampImg *= 2.**16
        #resampImg *= 0.0000689219599491
        #resampImg -= 1.54118269436172
    #else:
        #resampImg *= 2.**16
        #resampImg *= 0.0002642870705047
        #resampImg -= 1.41796343841118

    # Convert from pixel value to HAp density
    # In this case, HAp density is calculated with mu values, keV(1)=119
    if scan == 'g':
        resampImg *= 2.**16
        resampImg *= 0.0000689219599491
        resampImg -= 1.54118269436172
    else:
        resampImg *= 2.**16
        resampImg *= 0.00028045707501
        resampImg -= 1.48671229207043

    # Convert from pixel value to HAp density
    # In this case, HAp density is calculated using standards
    #if scan == 'g':
        #resampImg *= 2.**16
        #resampImg *= 0.000046368
        #resampImg -= 0.077857152
    #else:
        #resampImg *= 2.**16
        #resampImg *= 0.000182009
        #resampImg -= 0.077402903
        
    print scan, np.max(resampImg)
    
    return resampImg[:,:]

def lnprob(log_Delta_y, y_obs, y_sigma, mu_prior, sigma_prior, n_clip=3.):
    y_mod = np.cumsum(np.exp(log_Delta_y))
    
    Delta_y = (y_mod - y_obs) / y_sigma
    idx = Delta_y > n_clip
    Delta_y[idx] = n_clip
    
    log_likelihood = -0.5 * np.sum(Delta_y * Delta_y)
    #log_prior = np.sum(log_Delta_y)
    
    Delta_y = (log_Delta_y - mu_prior) / sigma_prior
    log_prior = -0.5 * np.sum(Delta_y * Delta_y)
    
    return log_likelihood + log_prior

# Main section of code in which defined functions are used
'''
Loads image
'''

def main():
    parser = argparse.ArgumentParser(prog='enamelsample',
                  description='Resample image of enamel to standard grid',
                  add_help=True)
    parser.add_argument('images', type=str, nargs='+', help='Images of enamel 1.')
    #parser.add_argument('images2', type=str, nargs='+', help='Images of enamel 2.')
    parser.add_argument('-sp', '--spacing', type=float, default=x_resampling, help='Marker spacing (in pixels).')
    parser.add_argument('-ex', '--exact', type=float, default=10., help='Exactness of baseline spline.')
    parser.add_argument('-l', '--order-by-length', action='store_true',
                              help='Order teeth according to length.')
    parser.add_argument('-s', '--show', action='store_true', help='Show plot.')
    parser.add_argument('-o', '--output-dir', type=str, default='likelihood min model ',
                              help='Directory in which to store output.')
    parser.add_argument('-f', '--output', type=str, default='likelihood_min_model.h5',
                        help='Name of mineralization model file to be created.')
    parser.add_argument('-p', '--partitions', type=int, nargs=2, default=(1,1),
                              help='Number of partitions, and partition to operate on.')
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
        alignedimg.append(get_image_values_2(img, markerPos, DeltaMarker, fname))
        
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
        alignedimgSorted = [alignedimg[i] for i in idx]
        alignedimg = alignedimgSorted
        Nx = Nx[idx]
        Ny = Ny[idx]
        age = age[idx]

    # Combine images into one 3-dimensional array
    nImages = len(alignedimg)
    imgStack = np.zeros((nImages, max(Ny), max(Nx)), dtype='f8') # Nx, Ny were flipped
    for i,img in enumerate(alignedimg):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        cimg = ax.imshow(img, aspect='auto', interpolation='nearest')
        cax = fig.colorbar(cimg)
        plt.show()
        imgStack[i, :Nx[i], :Ny[i]] = img[:,:] # was img.T[:,:] at end

    # Convert from HAp density to mineral fraction by weight
    imgStack /= 3.15
    # If converting from total density, use this equation instead
    # imgStack = (imgStack - 0.79) / (3.15 - 0.79)
    idx = (imgStack < 0.05) | (imgStack > 1.2)
    imgStack[idx] = np.nan
    
    # Relate image length to day: output for time x-axis is 'age_plt'
    age_coeff = np.polyfit(Nx, age, 5)
    
    Nx_age = np.zeros(Nx.size, dtype='f8')
    for i in xrange(len(age_coeff)):
        Nx_age += age_coeff[i] * Nx**(len(age_coeff)-i-1)
    
    Nx_age = np.around(Nx_age)
    Nx_age[Nx_age < 1.] = 1
    
    for k in xrange(1, len(Nx_age)):
        if Nx_age[k] <= Nx_age[k-1]:
            Nx_age[k] = Nx_age[k-1] + 1

    Nx_age = Nx_age.astype('u2')
    
    # start MCMC ##

    loc_store = []
    mask_store = []
    samples_store = []
    
    n_walkers = 4 * Nx_age.size
    n_steps = 2000
    n_store = 100

    t1 = time.time()

    # Splitting up pixels equally to run on a cluster

    #pix_idx = np.sum(np.isfinite(imgStack), axis=0) >= 3
    #x_idx = np.arange(imgStack.shape[1])
    #y_idx = np.arange(imgStack.shape[2])
    #xx, yy = np.meshgrid(x_idx, y_idx, indexing='ij')

    #pix_idx_x = xx[pix_idx] # was x_idx[pix_idx]
    #pix_idx_y = yy[pix_idx] # was y_idx[pix_idx]

    #n_parts, part_idx = args.partitions
    #partition_size = int(np.ceil(float(pix_idx_x.size) / float(n_parts)))

    #start = partition_size * (part_idx - 1)
    #end = start + partition_size + 1
    #pix_idx_x = pix_idx_x[start:end]
    #pix_idx_y = pix_idx_y[start:end]

    #for x, y in zip(pix_idx_x, pix_idx_y):              
        # Fit monotonically increasing mineralization model
        # to time series in this pixel
        #pct_min = imgStack[:, x, y]

        #idx = np.isfinite(pct_min)
        #n_points = np.sum(idx)

    for x in xrange(xcoordinate1,xcoordinate2):
        for y in xrange(ycoordinate1,ycoordinate2):
            pct_min = imgStack[:, x, y]
            idx = np.isfinite(pct_min)           
            n_points = np.sum(idx)

            # Skip pixel if too few time slices have data
            if n_points < 3:
                continue

            pct_min = pct_min[idx]

            # MCMC sampling
            sigma = 0.03 * np.ones(pct_min.size, dtype='f8')
            mu_prior = -6. * np.ones(pct_min.size, dtype='f8')
            sigma_prior = 2. * np.ones(pct_min.size, dtype='f8')
            
            model = TMonotonicPointModel(pct_min, sigma, mu_prior, sigma_prior)
            _, guess = model.guess(1)
            cov_guess = np.diag(sigma)

            sampler = TMCMC(model, guess, cov_guess/20.)

            N = 150000
            for i in range(N):
		sampler.step()
            sampler.flush()

            accept_frac = sampler.get_acceptance_rate()
            print 'Acceptance rate: %.4f %%' % (100. * accept_frac)
            
            chain = sampler.get_chain(burnin=N/3).T
            np.random.shuffle(chain)
            pct_min_samples = np.cumsum(np.exp(chain[:n_store]), axis=1)

            # Store results for pixel
            loc_store.append([x, y])
            mask_store.append(idx)
            samples_store.append(pct_min_samples)

            # Plot results
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            for s in pct_min_samples:
                ax.plot(Nx_age[idx], s, 'b-', alpha=0.05)
            
            ax.errorbar(Nx_age[idx], pct_min, yerr=sigma,
                        fmt='o')
            ax.set_ylim(0., 0.9)
            ax.set_xlim(0., 500.)
            ax.set_title('Mineralization over time, cervix')
            ax.set_ylabel('Estimated mineralization percent')
            ax.set_xlabel('Time in days')
            plt.show()

            return 0
        
    t2 = time.time()
    print '%.2f seconds per pixel.' % ((t2 - t1) / len(loc_store))

    # create new directory file for output mineralization model
    dirname = args.output_dir + datetime.now().strftime('%c')

    if dirname == None:
        return 0

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Calculate mineralization values for HDF5 file
    fname = args.output
    if fname.endswith('.h5'):
        fname = fname[:-3]
    fname += '%.3d.h5' % part_idx
    
    print 'Writing to %s ...' % fname
    
    loc_store = np.array(loc_store)
    mask_store = np.array(mask_store)

    n_pix = loc_store.shape[0]
    n_ages = Nx_age.size
    shape = (n_pix, n_store, n_ages)
    
    pct_min = np.empty(shape, dtype='f4')
    pct_min[:] = np.nan

    print 'loc_store shape', loc_store.shape
    print 'pct_min shape', pct_min.shape

    for i, samples in enumerate(samples_store):
        n_points = samples.shape[1]
        pct_min[i, :, :n_points] = samples[:, :]

    # Save results to an HDF5 file

    f = h5py.File(dirname + '/' + fname, 'w')
    
    dset = f.create_dataset('/locations', loc_store.shape, 'u2',
                                          compression='gzip',
                                          compression_opts=9)
    dset[:] = loc_store[:]

    dset = f.create_dataset('/ages', Nx_age.shape, 'u2')
    dset[:] = Nx_age
    
    dset = f.create_dataset('/age_mask', mask_store.shape, 'u1',
                                         compression='gzip',
                                         compression_opts=9)
    dset[:] = mask_store[:]
    
    dset = f.create_dataset('/pct_min_samples', pct_min.shape, 'f4',
                                                compression='gzip',
                                                compression_opts=9)
    dset[:] = pct_min[:]

    f.close()

    # calculate model information for save file

    s = imgStack.shape
    xpixelsize = voxelsize * args.spacing
    ypixelsize = voxelsize * y_resampling
    tooth_length = xpixelsize * s[1]
    tooth_height = ypixelsize * s[2]

    # Save text file with model information
    txt = 'Please also read model attributes in the h5py file!\n'
    txt += 'date: %s\n' % (datetime.now().strftime('%c'))
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

    return 0

if __name__ == '__main__':
    main()




