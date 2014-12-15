#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  calc_tooth_edj_distances.py
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
from scipy.optimize import curve_fit, minimize, leastsq
from scipy.integrate import quad
from scipy import pi, sin
import scipy.special as spec
from lmfit import Model

voxelsize = 46. # voxel resolution of your scan in microns?
species = 'Ovis_aries' # the species used for this model
calibration_type = 'undefined' # pixel to density calibration method
y_resampling = 1 # y pixel size of model compared to voxel size
x_resampling = 1 # x pixel size of model compared to voxel size
    
def load_image(fname):
    img = plt.imread(abspath(fname))
    return img

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

def get_image_values_2(img, markerPos, DeltaMarker, fname, step=y_resampling, threshold=0.1):
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
    age_label = int(fname[1:4])

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

    resampImg /= 3.15
    idx = (resampImg < 0.05) | (resampImg > 1)
    resampImg[idx] = np.nan
    
    return resampImg[:,:].T ########## .T?


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
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                              help='Directory in which to store output.')
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])

    warnings.simplefilter('ignore')
    
    alignedimg = []
    Nx, Ny = [], []
    age = []
    
    for i,fname in enumerate(args.images):
        print 'Processing %s ...' % fname
        
        img = load_image(fname)
        age.append(float(fname[1:4]))
        x, spl = get_baseline(img, exactness=args.exact)
        markerPos, markerDeriv = place_markers(x, spl[1], spacing=args.spacing)
        
        y = []
        for i in xrange(2):
            y.append(spl[i](x))
    
        DeltaMarker = -np.ones((len(markerDeriv), 2), dtype='f8')
        DeltaMarker[:,0] = markerDeriv[:]
        markerPos2 = markerPos + 80. * DeltaMarker

        alignedimg.append(get_image_values_2(img, markerPos, DeltaMarker, fname))

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
    
    # Relate image length to day: output for time x-axis is 'age_plt'
    #age_coeff = np.polyfit(Nx, age, 5) # Coeff 5 appropriate? What about gaussian rather than polyfit?
    Nx2 = Nx*46./1000.
    Nx2_max = np.max(Nx2) * 1.005
    print Nx2_max
    #Nx2[Nx2 > 35.99] = 35.99
    #Nx_age = np.zeros(Nx.size, dtype='f8')
    #for i in xrange(len(age_coeff)):
        #Nx_age += age_coeff[i] * Nx**(len(age_coeff)-i-1)
    Nx_age = (spec.erfinv((30.34 + Nx2 - Nx2_max)/30.34) -(11*.0061))/.0061
    for i,j in zip(Nx2, Nx_age):
        print i,j

    # Make Nx_age monotonically increasing
    #increasing = np.linspace(0., 0.1, len(Nx_age))
    #Nx_age = Nx_age + increasing

    Nx_age = np.around(Nx_age)
    Nx_age[Nx_age < 1.] = 1
    
    for k in xrange(1, len(Nx_age)):
        if Nx_age[k] <= Nx_age[k-1]:
            Nx_age[k] = Nx_age[k-1] + 1

    Nx_age = Nx_age.astype('u2')

    # calculate model information for save file

    xpixelsize = voxelsize * args.spacing
    ypixelsize = voxelsize * y_resampling
    tooth_length = xpixelsize * Nx / 1000

    # calculate um/day
    
    increase_per_day = np.repeat(np.diff(tooth_length), np.diff(Nx_age))
    increase_per_day = increase_per_day / np.repeat(np.diff(Nx_age), np.diff(Nx_age))
    increase_per_day = np.delete(increase_per_day, np.s_[0:4]) # Here I am deleting first numbers
    increase_per_day = increase_per_day * 1000

    for i in xrange(0, increase_per_day.size):
        if increase_per_day[i] > 300:
            increase_per_day[i] = 300

    days = np.linspace(1, increase_per_day.size, increase_per_day.size)
    #w = np.ones(200)
    #exact = .0018
    #s = UnivariateSpline(days, increase_per_day, s=200/exact)
    #xs = linspace(days[0], days[-1], 1000)
    #ys = s(xs)

    # Kierdorf 2013 data

    kday = np.array([13, 27, 41, 55, 75, 154, 168, 185, 200, 212, 230, 245])
    kext = np.array([177, 168, 180, 132, 110, 40, 36, 35, 33, 27, 28, 29])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(age, tooth_length, marker='o', linestyle='none', color='b', label='age')
    ax.plot(Nx_age, tooth_length, marker='o', linestyle='none', color='r', label='fitted age')
    ax2 = ax.twinx()
    #ax2.plot(xs, ys, color='g', label='radiograph extension, smooth')
    ax2.plot(days, increase_per_day, color='y', label='radiograph extension, raw')
    ax2.plot(kday, kext, color='k', mfc='none', marker='o', linestyle='none', label='histology extension')
    ax.set_xlabel('age or Nx_age in days')
    ax.set_ylabel('length in mm')
    ax2.set_ylabel('extension in um/day')
    plt.title('Age & Nx_age vs. tooth_length: tooth length by age at death')
    ax.set_ylim([0,40])
    ax.set_xlim([-50,320])
    ax2.set_ylim([0,350])
    ax2.set_xlim([-50,320])

    plt.show()
           
    return 0

if __name__ == '__main__':
    main()
