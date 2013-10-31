#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pointsample.py
#  
#  Copyright 2013 Daniel Green, Greg Green
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
import argparse, sys, warnings
from matplotlib.ticker import FuncFormatter
import time
import h5py
import emcee

from fit_positive_function import TMonotonicPointModel

# Load image

#matplotlib.pyplot.imread reads an image from a file into an array.
#fname can be a string path or file-like object; files must be in binary mode.
#returned value is a numpy.array; grayscale images return an MxN array.
#PNG files are most easily read by python.

    
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

#numpy.empty(shape, dtype=float, order='C')
#shape is int or tuple of int,
#dtype (optional) is output,
#order (optional) is column F or row C.
#np.nan is a macro meaning "not a number;"
#np.nan guarantees to unset the signbit, making a "positive" nan.
#mask returns 'true' for img > 0 at all coordinates in array;
#zeroes in mask are therefore 'false'.
#the len function returns the number of items in a sequence or map.
#amin(a, axis=None, out=None) returns the mininum along an array or axis.
#a is array_like input data. Axis and out are optional.
#amin returns ndarray: a new array with the result.

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

    #numpy.isnan tests element-wise for "not a number" NaN, returns bool array.
    #numpy.isnan(x[, out]) where x is an array.
    #The output is a new array where NaN = true, otherwise False, for each coord.
    #xMax+1 because edge should be above tooth mask?
    #linspace creates evenly spaced numbers, carefully handling endpoints.
    #numpy.linspace(start, stop[, num, endpoint, retstep])
    #start is a number (optional); default is zero.
    #stop is a number unless endpoint=False.
    #num is an integer, optional, and determines sample number. Default is 50.
    #retstep is the spacing between values, default 1.
    #If True, retstep returns (samples, step).
    #linspace returns ndarray samples:
    #num equally spaced samples in the closed interval [start, stop]
    #linspace also returns step: a float with the size of spacing.
    #See numpy documentation 1.7.0 p.438 for examples.
    
    isReal = np.where(edge[0] >= 0)[0]
    xMin, xMax = isReal[0], isReal[-1]
    edge = edge[:,xMin:xMax+1]
    x = np.linspace(xMin, xMax, xMax-xMin+1)    
    
    # Fit splines to edges

    #numpy.ones(shape, dtype=None, order='C')
    #This returns a new array of a given shape and type filled with ones.
    #Rules for np.ones follows numpy.zeros(shape, dtype=float, order='C')
    #shape is a single or a sequence of integers. dtype, order are optional.
    #np.ones returns ndarray of ones with a given shape, dtype and order.
    #len(s) returns the length (number of items) of an object;
    #the argument may be a sequence (string, tuple, list) or mapping (dictionary).

    #Scipy.interpolate functions essentially smooth discrete data by resampling.
    #splev evaluates a spline at any point;
    #splint finds the integral between 2 points.
    #1D splines can be interpolated from objects using (UnivariateSpline)
    #This can used to derive interpolated y values from x values, or
    #in our case to smooth data with a smoothing parameter s.
    #The default value is s = m - sqrt(2m) where m is the number of data points.
    #For no smoothing, set s = 0.
    #UnivariateSpline will not pass through all points, whereas
    #InterpolatedUnivariateSpling is a subclass that will.
    #See Scipy documentation 1.11.0 p.28 and p.259 for documentation.

    #scipy.interpolate.UnivariateSpline(x, y, w=None, bbox=[None, None], k=3, s=None)
    #Fits a spline y=s(x) of degree k to the provided x, y data;
    #s specifies the number of knots by giving a smoothing function.
    #x and y are array_like, 1D arrays, independent and dependent;
    #x must be increasing, y must be of same length as x.
    #w is optional, array_like, positive, gives weights for spline fitting;
    #If no w is specified, all weights are equal.
    #bbox is optional, array_like, specifies boundary of approximation interval;
    #No bbox info returns default bbox=[x[0], x[-1]].
    #k is optional, an integer, returns the degree of the spline, 5 or less.
    #s is optional, a float or None. If zero, spline interpolates through all points.
    #Number knots in spline will increase to satisfy this condition:
    #s =/> sum((w[i]*(y[i]-s(x[i])))**2,axis=0). Default s=len(w).
    #s=len(w)/5 is too rigid, s=len(w)/20 too bumpy. 9 to 11 is just right?

    spl = []
    w = np.ones(len(x))
    w[0] = 10.
    w[-1] = 10.
    for i in xrange(2):
        spl.append( interp.UnivariateSpline(x, edge[i], w=w, s=len(w)/exactness) )
    
    return x, spl

# Create markers and calculate distance along spline

#xFine are evenly spaced markers on spline, yFine are evaluated at xFine.
#numpy.diff calculates the discrete difference along the axis.
#The dist output is a sum of x and y distances along the spline.
#nMarkers is the integer number of markers: total distance / spacing.
#markerDist are evenly spaced numbers from spacing to the spline end.
#numpy.empty(shape, dtype=float, order='C') returns a new array:
#the array's type and shape are defined, but entries are not initialized.
#Here, markerPos is the array [(number of markers), 2], without defined entries.
#markerDeriv is the array [number of markers], without defined entries.

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

def get_image_values_2(img, markerPos, DeltaMarker, step=0.1):
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


def lnprob(log_Delta_y, model):
    return model(log_Delta_y) + np.log(min_pct_prior(log_Delta_y))


# Main section of code in which defined functions are used
'''
Loads image
'''

def main():
    parser = argparse.ArgumentParser(prog='enamelsample',
                  description='Resample image of enamel to standard grid',
                  add_help=True)
    parser.add_argument('images', type=str, nargs='+', help='Images of enamel.')
    parser.add_argument('-sp', '--spacing', type=float, default=5, help='Marker spacing (in pixels).')
    parser.add_argument('-ex', '--exact', type=float, default=10., help='Exactness of baseline spline.')
    parser.add_argument('-l', '--order-by-length', action='store_true',
                              help='Order teeth according to length.')
    parser.add_argument('-t', '--threads', type=int, default=4,
                              help='# of threads to run MCMC on.')
    parser.add_argument('-o', '--output', type=str, required=True,
                              help='Filename for output.')
    parser.add_argument('-s', '--show', action='store_true', help='Show plot.')
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
    imgStack[imgStack < .79] = np.nan

    # Convert from total to mineral fraction by weight
    imgStack = (3.15 - 3.15 * 0.79 / imgStack) / (3.15 - 0.79)
    idx = (imgStack < -0.2)
    imgStack[idx] = np.nan
    
    # Relate image length to day: output for time x-axis is 'age_plt'
    age_coeff = np.polyfit(Nx, age, 3)
    
    Nx_age = np.zeros(Nx.size, dtype='f8')
    for i in xrange(len(age_coeff)):
        Nx_age += age_coeff[i] * Nx**(len(age_coeff)-i-1)

    # Make Nx_age monotonically increasing
    increasing = np.linspace(0., .1, len(Nx_age))
    Nx_age = np.add(Nx_age, increasing)
    
    # Generate monotonically increasing model for each coordinate in tooth
    # Standard deviation in each measurement

    loc_store = []
    mask_store = []
    samples_store = []
    
    n_walkers = 4 * Nx_age.size
    n_steps = 300
    n_store = 100

    t1 = time.time()
    
    for x in xrange(imgStack.shape[0]):
        for y in xrange(imgStack.shape[1]):
            
            # Fit monotonically increasing mineralization model
            # to time series in this pixel
            pct_min = imgStack[:, x, y]

            idx = np.isfinite(pct_min)
            n_points = np.sum(idx)

            # Skip pixel if too few time slices have data
            if n_points < 3:
                continue

            print 'Pixel %d, %d ...' % (x, y)
            
            pct_min = pct_min[idx]

            print pct_min
            print idx
            print n_points

            # MCMC sampling
            sigma = 0.025 * np.ones(pct_min.size, dtype='f8')
            mu_prior = -6. * np.ones(pct_min.size, dtype='f8')
            sigma_prior = 2. * np.ones(pct_min.size, dtype='f8')
            
            model = TMonotonicPointModel(pct_min, sigma,
                                         mu_prior, sigma_prior)
            guess = model.guess(n_walkers)
            
            sampler = emcee.EnsembleSampler(n_walkers, n_points,
                                            lnprob, threads=args.threads,
                                            args=[model])
            
            pos, prob, state = sampler.run_mcmc(guess, n_steps)
            sampler.reset()
            pos, prob, state = sampler.run_mcmc(pos, n_steps)
            
            np.random.shuffle(sampler.flatchain)
            pct_min_samples = np.cumsum(np.exp(sampler.flatchain[:n_store]), axis=1)
            
            # Store results for pixel
            loc_store.append([x, y])
            mask_store.append(idx)
            samples_store.append(pct_min_samples)

            del sampler
            del model

            '''
            # Plot results
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            for s in pct_min_samples:
                ax.plot(Nx_age[idx], s, 'b-', alpha=0.05)
            
            ax.errorbar(Nx_age[idx], pct_min, yerr=sigma,
                        fmt='o')

            plt.show()
            '''

    t2 = time.time()
    print '%.2f seconds per pixel.' % ((t2 - t1) / len(loc_store))
    
    # Save the results to an HDF5 file
    print 'Writing to %s ...' % (args.output)
    
    loc_store = np.array(loc_store)
    mask_store = np.array(mask_store)

    n_pix = loc_store.shape[0]
    n_ages = Nx_age.size
    shape = (n_pix, n_store, n_ages)
    
    pct_min = np.empty(shape, dtype='f4')
    pct_min[:] = np.nan

    for i, samples in enumerate(samples_store):
        n_points = samples.shape[1]
        pct_min[i, :, :n_points] = samples[:, :]

    f = h5py.File(args.output, 'w')
    
    dset = f.create_dataset('/locations', loc_store.shape, 'u2',
                                          compression='gzip',
                                          compression_opts=9)
    dset[:] = loc_store[:]

    dset = f.create_dataset('/age_mask', mask_store.shape, 'u1',
                                         compression='gzip',
                                         compression_opts=9)
    dset[:] = mask_store[:]
    
    dset = f.create_dataset('/pct_min_samples', pct_min.shape, 'f4',
                                                compression='gzip',
                                                compression_opts=9)
    dset[:] = pct_min[:]

    f.close()
    
    return 0

if __name__ == '__main__':
    main()


