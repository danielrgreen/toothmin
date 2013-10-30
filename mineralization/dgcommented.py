#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  cells.py
#  
#  Copyright 2013 Greg Green <greg@greg-UX31A>
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
import scipy.ndimage.interpolation as imginterp
import scipy.ndimage.filters as filters
from os.path import abspath, expanduser

# Load image
'''
matplotlib.pyplot.imread reads an image from a file into an array.
fname can be a string path or file-like object; files must be in binary mode.
returned value is a numpy.array; grayscale images return an MxN array.
PNG files are most easily read by python.
'''
    
def load_image(fname):
    return plt.imread(abspath(fname))

# Define x and y, locate edges
'''
shape is an attribute for numpy arrays, returning the array dimensions.
If an array has n rows and m columns, array.shape is (n,m).
n and m would therefore represent y and x coordinates.
'''
'''
Input: img  numpy array with shape=(Nx, Ny)
Output: x    array of x-values of pixels used to fit baseline
spl  list with two splines: one for each edge.
This object can be called as follows:
spl[0](x) = y-value of edge #1 at x
spl[1](x) = y-value of edge #2 at x
'''
'''
numpy.empty(shape, dtype=float, order='C')
shape is int or tuple of int,
dtype (optional) is output,
order (optional) is column F or row C.
np.nan is a macro meaning "not a number;"
np.nan guarantees to unset the signbit, making a "positive" nan.
mask returns 'true' for img > 0 at all coordinates in array;
zeroes in mask are therefore 'false'.
the len function returns the number of items in a sequence or map.
amin(a, axis=None, out=None) returns the mininum along an array or axis.
a is array_like input data. Axis and out are optional.
amin returns ndarray: a new array with the result.
'''

def get_baseline(img):
    Ny, Nx = img.shape
    edge = np.empty((2,Nx), dtype='i4')
    edge[:,:] = np.nan
    mask = (img > 0.)
    for i in xrange(Nx):
        nonzero = np.where(mask[:,i])[0]
        if len(nonzero):
            edge[0,i] = np.min(nonzero)
            edge[1,i] = np.max(nonzero)

    # Clip edges to region with nonzero image
    '''
    numpy.isnan tests element-wise for "not a number" NaN, returns bool array.
    numpy.isnan(x[, out]) where x is an array.
    The output is a new array where NaN = true, otherwise False, for each coord.
    xMax+1 because edge should be above tooth mask?
    linspace creates evenly spaced numbers, carefully handling endpoints.
    numpy.linspace(start, stop[, num, endpoint, retstep])
    start is a number (optional); default is zero.
    stop is a number unless endpoint=False.
    num is an integer, optional, and determines sample number. Default is 50.
    retstep is the spacing between values, default 1.
    If True, retstep returns (samples, step).
    linspace returns ndarray samples:
    num equally spaced samples in the closed interval [start, stop]
    linspace also returns step: a float with the size of spacing.
    See numpy documentation 1.7.0 p.438 for examples.
    '''
    
    isReal = np.where(~np.isnan(edge[0]))[0]
    xMin, xMax = isReal[0], isReal[-1]
    edge = edge[:,xMin:xMax+1]
    x = np.linspace(xMin, xMax, xMax-xMin+1)    
    
    # Fit splines to edges
    '''
    numpy.ones(shape, dtype=None, order='C')
    This returns a new array of a given shape and type filled with ones.
    Rules for np.ones follows numpy.zeros(shape, dtype=float, order='C')
    shape is a single or a sequence of integers. dtype, order are optional.
    np.ones returns ndarray of ones with a given shape, dtype and order.
    len(s) returns the length (number of items) of an object;
    the argument may be a sequence (string, tuple, list) or mapping (dictionary).
    '''
    '''Scipy.interpolate functions essentially smooth discrete data by resampling.
    splev evaluates a spline at any point;
    splint finds the integral between 2 points.
    1D splines can be interpolated from objects using (UnivariateSpline)
    This can used to derive interpolated y values from x values, or
    in our case to smooth data with a smoothing parameter s.
    The default value is s = m - sqrt(2m) where m is the number of data points.
    For no smoothing, set s = 0.
    UnivariateSpline will not pass through all points, whereas
    InterpolatedUnivariateSpling is a subclass that will.
    See Scipy documentation 1.11.0 p.28 and p.259 for documentation.
    '''
    '''
    scipy.interpolate.UnivariateSpline(x, y, w=None, bbox=[None, None], k=3, s=None)
    Fits a spline y=s(x) of degree k to the provided x, y data;
    s specifies the number of knots by giving a smoothing function.
    x and y are array_like, 1D arrays, independent and dependent;
    x must be increasing, y must be of same length as x.
    w is optional, array_like, positive, gives weights for spline fitting;
    If no w is specified, all weights are equal.
    bbox is optional, array_like, specifies boundary of approximation interval;
    No bbox info returns default bbox=[x[0], x[-1]].
    k is optional, an integer, returns the degree of the spline, 5 or less.
    s is optional, a float or None. If zero, spline interpolates through all points.
    Number knots in spline will increase to satisfy this condition:
    s =/> sum((w[i]*(y[i]-s(x[i])))**2,axis=0). Default s=len(w).
    s=len(w)/5 is too rigid, s=len(w)/20 too bumpy. 9 to 11 is just right?
    '''
    
    spl = []
    w = np.ones(len(x))
    w[0] = 10.
    w[-1] = 10.
    for i in xrange(2):
        spl.append( interp.UnivariateSpline(x, edge[i], w=w, s=len(w)/8.) )
    
    return x, spl

# Create markers and calculate distance along spline
'''
xFine are evenly spaced markers on spline, yFine are evaluated at xFine.
numpy.diff calculates the discrete difference along the axis.
The dist output is a sum of x and y distances along the spline.
nMarkers is the integer number of markers: total distance / spacing.
markerDist are evenly spaced numbers from spacing to the spline end.
numpy.empty(shape, dtype=float, order='C') returns a new array:
the array's type and shape are defined, but entries are not initialized.
Here, markerPos is the array [(number of markers), 2], without defined entries.
markerDeriv is the array [number of markers], without defined entries.

'''

def place_markers(x, spl, spacing=5.):
    fineness = 10
    xFine = np.linspace(x[0], x[-1], fineness*len(x))
    yFine = spl(xFine)
    derivFine = np.empty(len(xFine), dtype='f8')
    for i,xx in enumerate(xFine):
        derivFine[i] = spl.derivatives(xx)[1]
    derivFine = filters.gaussian_filter1d(derivFine, fineness*spacing/2.)
    dx = np.diff(xFine)
    dy = np.diff(yFine)
    dist = np.sqrt(dx*dx + dy*dy)
    dist = np.cumsum(dist)
    nMarkers = int(dist[-1] / spacing)
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



def get_image_values(img,x,y,normal,dist,npts):
    '''
    Get the image values along a normal line from a point x.
    Traverse along this line dist pixels with npts
    '''

    # normal is [dx, dy] but may have unknown length
    normal = normal/np.sqrt(normal[0]**2 + normal[1]**2)
    
    # now normal is of length 1, and is in the form [dx,dy]
    xMin = x
    xMax = x + dist*normal[0]
    yMin = y
    yMax = y + dist*normal[1]
    
    #print "original", [xMin, yMin]
    #print "ending", [xMax,yMax]
    #print "distance", np.sqrt((xMax - xMin)**2 + (yMax - yMin)**2)
    
    # make a new array
    xcoords = np.linspace(xMin, xMax, npts)
    ycoords = np.linspace(yMin, yMax, npts)
    
    #print xcoords
    #print ycoords
    
    # travel along points, get image values put them in array
    return imginterp.map_coordinates(img,[xcoords,ycoords])

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
    
    #print samplePos[0,:,:]
    samplePos.shape = (nMarkers*(nSteps+1),2)
    #print samplePos.T[:100]
    resampImg = imginterp.map_coordinates(img.T, samplePos.T)
    resampImg.shape = (nMarkers, nSteps+1)
    #print resampImg.shape
    #print nMarkers, nSteps+1
    
    return resampImg.T[::-1,:]
    #print samplePos[nSteps+1,:]

    #print samplePos
    


# Main section of code in which defined functions are used
'''
Loads image
'''

def main():
    img = load_image('13um1314wkllm1.png')
    
    # Generate a spline for each tooth edge
    '''
    '''
    
    x, spl = get_baseline(img)
    
    # Place makers along the bottom edge
    '''
    '''
    
    markerPos, markerDeriv = place_markers(x, spl[1], spacing=400)
    
    # Calculate y values of edges
    '''
    '''
    
    y = []
    for i in xrange(2):
        y.append(spl[i](x))
    
    # Calculate perpendicular lines to edges
    '''
    Height of line is # in markerPos + #
    '''
    
    DeltaMarker = -np.ones((len(markerDeriv), 2), dtype='f8')
    DeltaMarker[:,0] = markerDeriv[:]
    markerPos2 = markerPos + 80. * DeltaMarker
    
    
    # Goal: take our x positions, step along the y positions, and get the
    # values from the image.
    
    # Plot everything
    '''
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.imshow(img, alpha=0.3)
    
    # Edges
    '''
    '''
    
    for i in xrange(2):
        ax.plot(x, y[i], 'g-')
    
    # Marker positions
    '''
    '''
    
    ax.scatter(markerPos[:,0], markerPos[:,1], c='g', s=8)
    
    # Create perpendicular lines
    '''
    '''
    
    for m1, m2 in zip(markerPos, markerPos2):
        mx = [m1[0], m2[0]]
        my = [m1[1], m2[1]]
        #print "marker length: ", np.sqrt((m2[0]-m1[0])**2 + (m2[1]-m1[1])**2)
        ax.plot(mx, my, 'g-')

    
    #numy = 40
    #numx = len(markerPos)
    #imgdist = 400
    
    alignedimg = get_image_values_2(img, markerPos, DeltaMarker)
    #alignedimg = np.ones([numy,numx])
    # Sample pixel values at regular intervals
    #for m1,normal,index in zip(markerPos,DeltaMarker,range(numx)):
        #tfvals = get_image_values(img,m1[0],m1[1],normal,imgdist,numy)
        #alignedimg[:,index] = get_image_values(img,m1[1],m1[0],normal,imgdist,numy)
    
    ax = fig.add_subplot(2,1,2)
    #ax.set_ylim(0,40)
    ax.imshow(alignedimg, alpha=0.3, aspect='auto')
    
    plt.show()

    return 0

if __name__ == '__main__':
    main()


