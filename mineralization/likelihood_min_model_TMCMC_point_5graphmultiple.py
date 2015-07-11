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
import matplotlib
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
from scipy.optimize import curve_fit, minimize, leastsq
from scipy.integrate import quad
from scipy import pi, sin
import scipy.special as spec
from lmfit import Model

# user inputs

x_coordinates = np.array([145, 145, 265, 265, 385, 385, 505, 505, 625, 625])
y_coordinates = np.array([12, 50, 12, 50, 12, 50, 12, 50, 12, 50])

voxelsize = 46. # voxel resolution of your scan in microns?
species = 'Ovis_aries' # the species used for this model
calibration_type = 'undefined' # pixel to density calibration method
y_resampling = 0.1 # y pixel size of model compared to voxel size
x_resampling = 1.0 # x pixel size of model compared to voxel size

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
    Ny, Nx = img.shape #########
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

def downsample_by_2(img):
    img = 0.5 * (img[:-1:2, :] + img[1::2, :])
    img = 0.5 * (img[:, :-1:2] + img[:, 1::2])
    
    return img

def get_image_values_2(img, markerPos, DeltaMarker, fname, step=y_resampling, threshold=0.1):
    #####
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
    #resampImg = np.rot90(resampImg, 1) ##########
    scan = str(fname[-5])  ##########

    # CONVERSION
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

    resampImg /= 2.62
    idx = (resampImg < .1) | (resampImg > 1)
    resampImg[idx] = np.nan
    '''
    mask = (resampImg > threshold)
    fill = np.min(resampImg)
    
    for col in xrange(resampImg.shape[0]):
        if np.any(mask[col,:]):
            nClip = np.min(np.where(mask[col,:])[0])
            
            if nClip < resampImg.shape[1]:
                tmp = resampImg[col,:].copy()
                resampImg[col,:] = fill
                resampImg[col,:resampImg.shape[1]-nClip] = tmp[nClip:]
    
    #ax = fig.add_subplot(3,1,2)
    #cimg = ax.imshow(resampImg.copy().T, origin='lower', aspect='auto', interpolation='none')
    #cax = fig.colorbar(cimg)
    
    # Downsample by a factor of two
    idx = resampImg < threshold
    resampImg[idx] = np.nan
    
    resampImg = downsample_by_2(resampImg)
    idx = (downsample_by_2(idx.astype('f8')) > 0.)
    
    resampImg[idx] = fill
    
    #ax = fig.add_subplot(3,1,3)
    #cimg = ax.imshow(resampImg.T, origin='lower', aspect='auto', interpolation='none')
    #cax = fig.colorbar(cimg)
    
    #plt.show()
    '''
    return resampImg[:,:].T

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

def figure_out_trajectories(x_coord, y_coord, imgStack, n_store,
               loc_store, mask_store, samples_store, Nx_age):

    x = x_coord
    y = y_coord
    pct_min = imgStack[:, x, y]
    idx = np.isfinite(pct_min)           
    n_points = np.sum(idx)
    pct_min = pct_min[idx]

    # MCMC sampling
    sigma = 0.06 * np.ones(pct_min.size, dtype='f8')
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

    return pct_min_samples, loc_store, mask_store, samples_store, pct_min, idx
'''
def plot_points(pct_min_samples, loc_store, mask_store, samples_store, x, y)
            # Plot results
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            for s in pct_min_samples:
                ax.plot(Nx_age[idx], s, 'b-', alpha=0.05)
            
            ax.errorbar(Nx_age[idx], pct_min, yerr=sigma, fmt='o')
            ax.set_ylim(0., 0.9)
            ax.set_xlim(0., 300.)
            ax.set_title('Mineralization over time, x=%d, y=%d' % (x_coord, y_coord))
            ax.set_ylabel('Estimated mineralization percent')
            ax.set_xlabel('Time in days')
            fig.savefig('Dec14_min_over_time_x=%d_y=%d.png' % (x_coord, y_coord), dpi=300, figsize=4, edgecolor='none')
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
    imgStack = np.zeros((nImages, max(Nx), max(Ny)), dtype='f8') # Nx, Ny were flipped
    ##########    
    for i,img in enumerate(alignedimg):
        imgStack[i, :Nx[i], :Ny[i]] = img.T[:,:] # was img.T[:,:] at end
        idx = (imgStack < .15) | (imgStack > 1.1)
        imgStack[idx] = np.nan
    
    # Convert from HAp density to mineral fraction by weight
    #imgStack /= 3.15
    #idx = (imgStack < 0.05) | (imgStack > 1.2)
    #imgStack[idx] = np.nan
    
    # Relate image length to day: output for time x-axis is 'age_plt'
    #age_coeff = np.polyfit(Nx, age, 5)
    
    #Nx_age = np.zeros(Nx.size, dtype='f8')
    #for i in xrange(len(age_coeff)):
        #Nx_age += age_coeff[i] * Nx**(len(age_coeff)-i-1)
    Nx2 = Nx*46./1000.
    Nx2_max = np.max(Nx2) * 1.005
    Nx_age = (spec.erfinv((30.34 + Nx2 - Nx2_max)/30.34) -(11*.0061))/.0061
    
    Nx_age = np.around(Nx_age)
    Nx_age[Nx_age < 1.] = 1
    
    for k in xrange(1, len(Nx_age)):
        if Nx_age[k] <= Nx_age[k-1]:
            Nx_age[k] = Nx_age[k-1] + 1

    #Nx_age = Nx_age.astype('u2')
    
    # start MCMC ##

    loc_store = []
    mask_store = []
    samples_store = []
    
    n_walkers = 4 * Nx_age.size
    n_steps = 2000
    n_store = 100

    pct_min_samples0, loc_store0, mask_store0, samples_store0, pct_min0, idx0 = figure_out_trajectories(x_coordinates[0], y_coordinates[0], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples1, loc_store1, mask_store1, samples_store1, pct_min1, idx1 = figure_out_trajectories(x_coordinates[1], y_coordinates[1], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples2, loc_store2, mask_store2, samples_store2, pct_min2, idx2 = figure_out_trajectories(x_coordinates[2], y_coordinates[2], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples3, loc_store3, mask_store3, samples_store3, pct_min3, idx3 = figure_out_trajectories(x_coordinates[3], y_coordinates[3], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples4, loc_store4, mask_store4, samples_store4, pct_min4, idx4 = figure_out_trajectories(x_coordinates[4], y_coordinates[4], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples5, loc_store5, mask_store5, samples_store5, pct_min5, idx5 = figure_out_trajectories(x_coordinates[5], y_coordinates[5], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples6, loc_store6, mask_store6, samples_store6, pct_min6, idx6 = figure_out_trajectories(x_coordinates[6], y_coordinates[6], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples7, loc_store7, mask_store7, samples_store7, pct_min7, idx7 = figure_out_trajectories(x_coordinates[7], y_coordinates[7], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples8, loc_store8, mask_store8, samples_store8, pct_min8, idx8 = figure_out_trajectories(x_coordinates[8], y_coordinates[8], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)
    pct_min_samples9, loc_store9, mask_store9, samples_store9, pct_min9, idx9 = figure_out_trajectories(x_coordinates[9], y_coordinates[9], imgStack, n_store, loc_store, mask_store, samples_store, Nx_age)

    sigma = 5

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples0: # cusp edj
        ax.plot(Nx_age[idx0], s*100, '-', color='#AEB404', alpha=0.05)
    #ax.errorbar(Nx_age[idx0], pct_min0, yerr=sigma, fmt='o', color='#AEB404', label='x=%d, y=%d' % (x_coordinates[0], y_coordinates[0]))
    ax.errorbar(Nx_age[idx0], pct_min0*100, yerr=sigma, fmt='o', color='#AEB404') # % (x_coordinates[0], y_coordinates[0]))
    ax.plot(Nx_age[idx0], np.mean(pct_min_samples0, axis=0)*100, '-', color='#AEB404')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Cuspal, low')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_cusp_low.svg', dpi=300, figsize=4, edgecolor='none')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples1: # cusp surface
        ax.plot(Nx_age[idx1], s*100, '-', color='#F7FE2E', alpha=0.05)
    #ax.errorbar(Nx_age[idx1], pct_min1, yerr=sigma, fmt='o', color='#F7FE2E', label='x=%d, y=%d' % (x_coordinates[1], y_coordinates[1]))
    ax.errorbar(Nx_age[idx1], pct_min1*100, yerr=sigma, fmt='o', color='#F7FE2E') # % (x_coordinates[1], y_coordinates[1]))
    ax.plot(Nx_age[idx1], np.mean(pct_min_samples1, axis=0)*100, '-', color='#F7FE2E')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Cuspal, high')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_cusp_high.svg', dpi=300, figsize=4, edgecolor='none')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples2: # high crown edj
        ax.plot(Nx_age[idx2], s*100, '-', color='#31B404', alpha=0.05)
    #ax.errorbar(Nx_age[idx2], pct_min2, yerr=sigma, fmt='o', color='#31B404', label='x=%d, y=%d' % (x_coordinates[2], y_coordinates[2]))
    ax.errorbar(Nx_age[idx2], pct_min2*100, yerr=sigma, fmt='o', color='#31B404') # % (x_coordinates[2], y_coordinates[2]))
    ax.plot(Nx_age[idx2], np.mean(pct_min_samples2, axis=0)*100, '-', color='#31B404')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Midcrown, low')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_midcrown_low1.svg', dpi=300, figsize=4, edgecolor='none')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples3: # high crown surface
        ax.plot(Nx_age[idx3], s*100, '-', color='#64FE2E', alpha=0.05)
    #ax.errorbar(Nx_age[idx3], pct_min3, yerr=sigma, fmt='o', color='#64FE2E', label='x=%d, y=%d' % (x_coordinates[3], y_coordinates[3]))
    ax.errorbar(Nx_age[idx3], pct_min3*100, yerr=sigma, fmt='o', color='#64FE2E') # % (x_coordinates[3], y_coordinates[3]))
    ax.plot(Nx_age[idx3], np.mean(pct_min_samples3, axis=0)*100, '-', color='#64FE2E')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Midcrown, high')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_midcrown_high1.svg', dpi=300, figsize=4, edgecolor='none')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples4: # mid crown edj
        ax.plot(Nx_age[idx4], s*100, '-', color='#088A68', alpha=0.05)
    #ax.errorbar(Nx_age[idx4], pct_min4, yerr=sigma, fmt='o', color='#088A68', label='x=%d, y=%d' % (x_coordinates[4], y_coordinates[4]))
    ax.errorbar(Nx_age[idx4], pct_min4*100, yerr=sigma, fmt='o', color='#088A68') # % (x_coordinates[4], y_coordinates[4]))
    ax.plot(Nx_age[idx4], np.mean(pct_min_samples4, axis=0)*100, '-', color='#088A68')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Midcrown, low')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_midcrown_low2.svg', dpi=300, figsize=4, edgecolor='none')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples5: # mid crown surface
        ax.plot(Nx_age[idx5], s*100, '-', color='#00FFBF', alpha=0.05)
    #ax.errorbar(Nx_age[idx5], pct_min5, yerr=sigma, fmt='o', color='#00FFBF', label='x=%d, y=%d' % (x_coordinates[5], y_coordinates[5]))
    ax.errorbar(Nx_age[idx5], pct_min5*100, yerr=sigma, fmt='o', color='#00FFBF') # % (x_coordinates[5], y_coordinates[5]))
    ax.plot(Nx_age[idx5], np.mean(pct_min_samples5, axis=0)*100, '-', color='#00FFBF')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Midcrown, high')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_midcrown_high2.svg', dpi=300, figsize=4, edgecolor='none')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples6: # lower crown edj
        ax.plot(Nx_age[idx6], s*100, '-', color='#045FB4', alpha=0.05)
    #ax.errorbar(Nx_age[idx6], pct_min6, yerr=sigma, fmt='o', color='#045FB4', label='x=%d, y=%d' % (x_coordinates[6], y_coordinates[6]))
    ax.errorbar(Nx_age[idx6], pct_min6*100, yerr=sigma, fmt='o', color='#045FB4') # % (x_coordinates[6], y_coordinates[6]))
    ax.plot(Nx_age[idx6], np.mean(pct_min_samples6, axis=0)*100, '-', color='#045FB4')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Micrown, low')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_midcrown_low3.svg', dpi=300, figsize=4, edgecolor='none')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples7: # lower crown surface
        ax.plot(Nx_age[idx7], s*100, '-', color='#2E9AFE', alpha=0.05)
    #ax.errorbar(Nx_age[idx7], pct_min7, yerr=sigma, fmt='o', color='#2E9AFE', label='x=%d, y=%d' % (x_coordinates[7], y_coordinates[7]))
    ax.errorbar(Nx_age[idx7], pct_min7*100, yerr=sigma, fmt='o', color='#2E9AFE') # % (x_coordinates[7], y_coordinates[7]))
    ax.plot(Nx_age[idx7], np.mean(pct_min_samples7, axis=0)*100, '-', color='#2E9AFE')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Midcrown, high')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_midcrown_high3.svg', dpi=300, figsize=4, edgecolor='none')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples8: # cervical edj
        ax.plot(Nx_age[idx8], s*100, '-', color='#5F04B4', alpha=0.05)
    #ax.errorbar(Nx_age[idx8], pct_min8, yerr=sigma, fmt='o', color='#5F04B4', label='x=%d, y=%d' % (x_coordinates[8], y_coordinates[8]))
    ax.errorbar(Nx_age[idx8], pct_min8*100, yerr=sigma, fmt='o', color='#5F04B4') # % (x_coordinates[8], y_coordinates[8]))
    ax.plot(Nx_age[idx8], np.mean(pct_min_samples8, axis=0)*100, '-', color='#5F04B4')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Cervical, low')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_cervical_low.svg', dpi=300, figsize=4, edgecolor='none')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for s in pct_min_samples9: # cervical surface
        ax.plot(Nx_age[idx9], s*100, '-', color='#9A2EFE', alpha=0.05)
    #ax.errorbar(Nx_age[idx9], pct_min9, yerr=sigma, fmt='o', color='#9A2EFE', label='x=%d, y=%d' % (x_coordinates[9], y_coordinates[9]))
    ax.errorbar(Nx_age[idx9], pct_min9*100, yerr=sigma, fmt='o', color='#9A2EFE') # % (x_coordinates[9], y_coordinates[9]))
    ax.plot(Nx_age[idx9], np.mean(pct_min_samples9, axis=0)*100, '-', color='#9A2EFE')
    ax.set_ylim(20, 100)
    ax.set_xlim(0., 280.)
    #ax.set_title('Cervical, high')
    #ax.set_ylabel('Estimated mineralization percent')
    #ax.set_xlabel('Time in days')
    fig.savefig('Dec_min_cervical_high.svg', dpi=300, figsize=4, edgecolor='none')

    #fig.savefig('Dec15_min_over_time.png', dpi=300, figsize=4, edgecolor='none')

    return 0

if __name__ == '__main__':
    main()




