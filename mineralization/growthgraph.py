# Graphing growth data

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
from fit_positive_function import TMonotonicPointModel, TMCMC
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

# Load image
    
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
    idx = (resampImg < 0.05) | (resampImg > 0.7)
    resampImg[idx] = np.nan
    vmax = 0.7
    vmin = 0.15
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cimg = ax.imshow(resampImg.T, origin='lower', aspect='equal', interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title(r'$t = %d \ \mathrm{days}$' % age_label, fontsize=14)
    cax = fig.colorbar(cimg)
    plt.show()
    #fig.savefig('december_toothmin_im_%d_days.png' % age_label, dpi=300)
    
    return resampImg[:,:].T, age_label

def est_tooth_extension(x, amplitude, slope, offset):
    
    height_max = 36. # in millimeters
    extension_erf = (amplitude * spec.erf(slope * (x - offset))) + (height_max - amplitude)

    return extension_erf

def curve_residuals(p0, pcurve_fit, x,y):
    penalization = ((p0[2] - pcurve_fit[2]))**2 * .005/x.size
    resids = ((y - est_tooth_extension(x, p0[0],p0[1],p0[2]))**2 + penalization)**.5
    return resids

def main():

    '''
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
    '''

    # My sheep model data from images, video
    model_days = np.array([30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280., 290., 300.])
    model_completion = np.array([2.3, 2.3, 2.76, 2.76, 4.6, 9.2, 11.96, 11.96, 13.8, 13.8, 15.64, 16.56, 16.56, 20.7, 20.7, 22.08, 23.46, 23.46, 23.46, 23.46, 23.46, 24.38, 24.84, 24.84, 27.6, 27.6, np.nan, 30.36])
    model_initiation = np.array([14.72, 15.64, 16.56, 17.48, 18.4, 24.84, 26.68, 26.68, 28.52, 28.52, 29.44, 30.36, 30.36, 31.28, 31.28, 31.74, 33.12, 33.12, 33.12, 33.12, 33.12, 33.58, 34.04, 34.04, 34.96, 34.96, np.nan, 34.96])
    model_min_length = model_initiation - model_completion
    model_length_percent = model_min_length / 39.
    model_increase_per_day = np.array([1.104, 0.552, 1.564, 1.932, 0.092, 0., 0.16866667, 0.16866667, 0.16866667, 0.16866667, 0.16866667, 0.16866667, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.10514286, 0.092, 0.15333333, 0.15333333, 0.15333333, 0.138, 0.138, 0.184, 0.276, 0.21466667, 0.21466667, 0.21466667, 0., 0.552, 0.70533333, 0.70533333, 0.70533333, 1.104, 0.598, 0.598, 0., 0.299, 0.299, 0.299, 0.299, 0.1472, 0.1472, 0.1472, 0.1472, 0.1472, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.0736, 0.0736, 0.0736, 0.0736, 0.0736, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0.05952941, 0., 0.092, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.05366667, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.04293333, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.03942857, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0.0368, 0., 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.03504762, 0.046, 0.046, 0.0345, 0.0345, 0.0345, 0.0345, 0.0345, 0.0345, 0.0345, 0.0345, 0., 0.046, 0.046, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.04025, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.0644, 0.0644, 0.0644, 0.0644, 0.0644, 0.0644, 0.0644, 0.0644, 0.0644, 0.0644]) 
    for i in xrange(np.size(model_increase_per_day)):
        if model_increase_per_day[i] > 0.4:
            model_increase_per_day[i] = 0.40
    model_pct_increase_per_day = model_increase_per_day / np.max(model_increase_per_day)

    # Kierdorf 2013 data
    kday = np.array([7.5, 21.5, 49.5, 63.3, 73., 154., 168., 185.5, 199., 212.5, 230., 247.5])
    kext = np.array([177.6, 168.4, 180., 131.7, 109.9, 40.4, 35.5, 35.0, 33.2, 27.1, 28.0, 29.3])

    tooth_days = np.array([1., 9., 11., 19., 21., 30., 31., 31., 38., 42., 54., 56., 56., 58., 61., 66., 68., 73., 78., 84., 88., 92., 97., 100., 101., 101., 104., 105., 124., 127., 140., 140., 157., 167., 173., 174., 179., 202., 222., 235., 238., 251., 259., 274.])
    tooth_extension = np.array([9.38, 8.05, 11.32, 9.43, 13.34, 16.19, 13.85, 15.96, 15.32, 14.21, 17.99, 19.32, 19.32, 18.31, 17.53, 18.68, 18.49, 22.08, 23.14, 19.92, 27.97, 24.38, 25.53, 29.07, 27.65, 26.27, 27.55, 24.33, 29.03, 29.07, 30.36, 31.79, 31.37, 31.28, 35.79, 29.81, 31.79, 34.04, 33.21, 34.50, 33.76, 33.40, 36.34, 33.63])
    tooth_35p = np.array([2.07, 1.52, 2.39, 2.67, 4.60, 7.04, 4.55, 5.47, 5.47, 4.83, 8.19, 9.98, 9.75, 9.89, 9.29, 10.40, 8.88, 12.88, 14.49, 11.96, 19.04, 17.48, 16.74, 20.61, 18.86, 17.34, 19.92, 14.67, 22.03, 22.91, 24.47, 26.08, 26.13, 25.94, 34.45, 25.35, 26.91, 30.08, 30.68, 33.49, 28.29, 28.34, 35.83, 33.63])
    tooth_70p = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 2.76, np.nan, np.nan, np.nan, np.nan, 3.68, 4.60, 4.69, 5.11, 4.88, 5.75, 4.74, 7.36, 8.97, 5.15, 13.62, 12.33, 11.13, 14.54, 13.25, 10.81, 13.25, 8.69, 15.78, 17.16, 19.27, 20.56, 19.87, 21.48, 30.27, 20.01, 22.17, 24.56, 26.27, 28.11, 23.55, 23.83, 36.34, 29.49])
    tooth_mat_length = tooth_extension - tooth_70p / np.percentile(tooth_extension, 95)

    tooth_70p_days = np.array([30., 54., 56., 56., 58., 61., 66., 68., 73., 78., 84., 88., 92., 97., 100., 101., 101., 104., 105., 124., 127., 140., 140., 157., 167., 173., 174., 179., 202., 222., 235., 238., 251., 259., 274.])
    tooth_70p_b = np.array([2.76, 3.68, 4.60, 4.69, 5.11, 4.88, 5.75, 4.74, 7.36, 8.97, 5.15, 13.62, 12.33, 11.13, 14.54, 13.25, 10.81, 13.25, 8.69, 15.78, 17.16, 19.27, 20.56, 19.87, 21.48, 30.27, 20.01, 22.17, 24.56, 26.27, 28.11, 23.55, 23.83, 36.34, 29.49])

    p0_var_ext = np.array([30.34, .006102, -10.54])
    p0_var_p35 = np.array([23., .008, 13.])
    p0_var_p70 = np.array([25., .006, 110.])

    days = np.linspace(-50, 350, 401)
    ext_variables, ext_pcov = curve_fit(est_tooth_extension, tooth_days, tooth_extension, p0_var_ext)
    p35_variables, p35_pcov = curve_fit(est_tooth_extension, tooth_days, tooth_35p, p0_var_p35)
    p70_variables, p70_pcov = curve_fit(est_tooth_extension, tooth_70p_days, tooth_70p_b, p0_var_p70)
    p70_variables2, p70_pcov2 = leastsq(func=curve_residuals, x0=p70_variables, args=(p0_var_p70, tooth_70p_days, tooth_70p_b))
    print ext_variables
    print p35_variables
    print p70_variables
    print p70_variables2
    
    extension_erf = est_tooth_extension(days, ext_variables[0], ext_variables[1], ext_variables[2])
    p35_erf = est_tooth_extension(days, p35_variables[0], p35_variables[1], p35_variables[2])
    p70_erf = est_tooth_extension(days, p70_variables2[0], p70_variables2[1], p70_variables2[2])

    diff_extension_erf = np.diff(extension_erf) * 1000
    diff_p35_erf = np.diff(p35_erf) * 1000
    diff_p70_erf = np.diff(p70_erf) * 1000

    ext_zero = ext_variables[1] - ext_variables[2]*spec.erfinv((36. - ext_variables[0])/ext_variables[0])
    p35_zero = p35_variables[1] - p35_variables[2]*spec.erfinv((36. - p35_variables[0])/p35_variables[0])
    p70_zero = p70_variables2[1] - p70_variables2[2]*spec.erfinv((36. - p70_variables2[0])/p70_variables2[0])
    print ext_zero, p35_zero, p70_zero 

    gmod = Model(est_tooth_extension)
    print gmod.eval(x=tooth_70p_b, amplitude=25., slope=.006, offset=110.)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ax2.plot(days[1::4], diff_extension_erf[::4], 'b.', label=r'$ \mathrm{extension} \ \Delta $')
    ax2.plot(days[1::4], diff_p35_erf[::4], 'm.', label=r'$ \mathrm{maturation} \ \Delta $')
    ax2.plot(days[1::4], diff_p70_erf[::4], 'r.', label=r'$ \mathrm{completion} \ \Delta $')
    ax2.set_ylim([0,300])
    ax2.set_xlim([-60,350])
    ax.plot(tooth_days, tooth_extension, marker='o', linestyle='none', color='b', label=r'$ \mathrm{extension}$')
    ax.plot(tooth_days, tooth_35p, marker='o', linestyle='none', color='m', label=r'$ \mathrm{maturation}$')
    ax.plot(tooth_days, tooth_70p, marker='o', linestyle='none', color='r', label=r'$ \mathrm{completion}$')
    ax.plot(days, extension_erf, linestyle='-', color='b', label=r'$ \mathrm{extension,} \ \mathrm{optimized} $')
    ax.plot(days, p35_erf, linestyle='-', color='m', label=r'$ \mathrm{maturation,} \ \mathrm{optimized} $')
    ax.plot(days, p70_erf, linestyle='-', color='r', label=r'$ \mathrm{completion,} \ \mathrm{optimized} $')
    ax.set_ylim([0,40])
    ax.set_xlim([-60,350])
    plt.title('Enamel secretion and maturation progress over time')
    ax.set_xlabel('Days after birth')
    ax.set_ylabel('Progress from cusp tip in mm')
    ax2.set_ylabel('Secretion or maturation speed in um/day')
    ax2.legend(loc='lower right', fancybox=True, framealpha=0.8)
    ax.legend(loc='upper left', fancybox=True, framealpha=0.8)

    plt.show()
    '''
    ax2 = ax.twinx()
    ax2.plot(xs, ys, color='g', label='radiograph extension, smooth')
    ax2.plot(days, increase_per_day, color='y', label='radiograph extension, raw')
    ax2.plot(kday, kext, color='k', mfc='none', marker='o', linestyle='none', label='histology extension')
    plt.show()
    '''



    return 0
if __name__ == '__main__':
    main()
