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
from os.path import abspath, expanduser

# Load image
# matplotlib.pyplot.imread reads an image from a file into an array. fname can be a string path or file-like object; files must be in binary mode. returned value is a numpy.array; grayscale images return an MxN array. PNG files are best.

def load_image(fname):
	return plt.imread(abspath(fname))

# Define x and y
def get_baseline(img):
	Ny, Nx = img.shape
	
	# Locate edges
	# numpy.empty(shape, dtype=float, order='C')
	# shape is int or tuple of int, dtype (optional) is output, order (optional) is column F or row C.
	# npy.nan is a macro meaning "not a number," and guarantees to unset the signbit, making a "positive" nan. 
	edge = np.empty((2,Nx), dtype='i4')
	edge[:,:] = np.nan
	mask = (img > 0.)
	for i in xrange(Nx):
		nonzero = np.where(mask[:,i])[0]
		if len(nonzero):
			edge[0,i] = np.min(nonzero)
			edge[1,i] = np.max(nonzero)
	
	# Clip edges to region with nonzero image
	isReal = np.where(~np.isnan(edge[0]))[0]
	xMin, xMax = isReal[0], isReal[-1]
	edge = edge[:,xMin:xMax+1]
	x = np.linspace(xMin, xMax, xMax-xMin+1)	
	
	# Fit splines to edges
	spl = []
	w = np.ones(len(x))
	w[0] = 10.
	w[-1] = 10.
	for i in xrange(2):
		spl.append( interp.UnivariateSpline(x, edge[i], w=w, s=len(w)/10.) )
	
	return x, spl

def place_markers(x, spl, spacing=5.):
	xFine = np.linspace(x[0], x[-1], 10*len(x))
	yFine = spl(xFine)
	dx = np.diff(xFine)
	dy = np.diff(yFine)
	dist = np.sqrt(dx*dx + dy*dy)
	dist = np.cumsum(dist)
	
	nMarkers = int(dist[-1] / spacing)
	markerDist = np.linspace(spacing, spacing*nMarkers, nMarkers)
	markerPos = np.empty((nMarkers, 2), dtype='f8')
	markerDeriv = np.empty(nMarkers, dtype='f8')
	cellNo = 0
	for xx, d in zip(xFine[1:], dist):
		if d >= (cellNo+1) * spacing:
			markerPos[cellNo, 0] = xx
			markerPos[cellNo, 1] = spl(xx)
			markerDeriv[cellNo] = spl.derivatives(xx)[1]
			# print markerDeriv[cellNo]
			cellNo += 1
	
	return markerPos, markerDeriv

def curve_dist(curve):
	dx = np.ones(len(curve)-1)
	dy = np.diff(curve)
	idx = (curve == 0.)
	idx = idx[:-1] | idx[1:]
	dy[idx] = 0.
	dx[idx] = 0.
	return np.hstack([[0], np.cumsum(np.sqrt(dx*dx+dy*dy))])

def get_markers(curve, cell_length):
	dist = curve_dist(curve)
	marker_x, marker_y = [], []
	
	startIdx = np.where(dist > 0.)[0][0]
	endIdx = np.where(np.diff(dist))[0][-1]
	
	cellNo = 0
	for i,d in enumerate(dist[startIdx:endIdx+1]):
		if d >= cellNo*cell_length:
			marker_y.append(curve[startIdx + i])
			marker_x.append(startIdx + i)
			cellNo += 1
	
	markers = np.array([marker_x, marker_y])
	return markers

def get_spine(curve):
	pass

def get_slopes(markers):
	diff = np.diff(markers, axis=1)
	diff = diff[:,:-1] + diff[:,1:]
	slope = np.empty(diff.shape, dtype='f8')
	slope[0,:] = diff[1,:]
	slope[1,:] = -diff[0,:]
	return slope

def main():
	img = load_image('13um1314wkllm1.png')
	
	'''
	bottom, top = get_baseline(img)
	x = np.arange(img.shape[1])
	
	markers = get_markers(top, 3.)
	print 'Horizontal cells: %d' % markers.shape[1]
	
	slopes = get_slopes(markers)
	markers_2 = markers[:,1:-1] + 10. * slopes
	'''
	
	# Generate a spline for each tooth edge
	x, spl = get_baseline(img)
	
	# Place makers along the bottom edge
	markerPos, markerDeriv = place_markers(x, spl[1], spacing=5)
	
	# Calculate y values of edges
	y = []
	for i in xrange(2):
		y.append(spl[i](x))
	
	# Calculate perpendicular lines to edges
	DeltaMarker = -np.ones((len(markerDeriv), 2), dtype='f8')
	DeltaMarker[:,0] = markerDeriv[:]
	markerPos2 = markerPos + 40. * DeltaMarker
	
	# Plot everything
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.imshow(img, alpha=0.3)
	
	# Edges
	for i in xrange(2):
		ax.plot(x, y[i], 'g-')
	
	# Marker positions
	ax.scatter(markerPos[:,0], markerPos[:,1], c='g', s=8)
	
	# Perpendicular lines
	for m1, m2 in zip(markerPos, markerPos2):
		mx = [m1[0], m2[0]]
		my = [m1[1], m2[1]]
		ax.plot(mx, my, 'g-')
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

