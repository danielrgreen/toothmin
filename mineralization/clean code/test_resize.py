__author__ = 'darouet'

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize

def resize(array, shape):
    '''
        Takes an array and broadcasts it into a new shape, ignoring NaNs.
        '''

    # Calculate shapes
    oldshape = array.shape
    axis0 = np.repeat(array, shape[0], axis=0)
    axis1 = np.repeat(axis0, shape[1], axis=1)
    print 'axis1', axis1

    # Resize
    axis1ravel = np.ravel(axis1)
    axis1stack = np.reshape(axis1ravel, (shape[0] * shape[1] * oldshape[0], oldshape[1]))
    m_axis1stack = np.ma.masked_array(axis1stack, np.isnan(axis1stack))
    print 'axis1stack', axis1stack
    axis1mean = np.mean(m_axis1stack, axis=1)
    print 'axis1mean', axis1mean
    axis2stack = np.reshape(axis1mean, (oldshape[0] * shape[0], shape[1])).T
    print 'axis2stack', axis2stack
    print 'axis2stack size', axis2stack.size
    print 'shape[0]*shape[1]*oldshape[0]', shape[0]*shape[1]*oldshape[0]

    axis2reshape = np.reshape(axis2stack, (shape[0]*shape[1], oldshape[0]))
    print 'axis2reshape', axis2reshape

    m_axis2reshape = np.ma.masked_array(axis2reshape, np.isnan(axis2reshape))
    axis2reshape_mean = np.mean(m_axis2reshape, axis=1).reshape(shape[1], shape[0])
    new_array = axis2reshape_mean.T
    print "new array", new_array
    # Add back in NaNs, threshold > 50% NaN
    nan_map = np.zeros(oldshape)
    nan_map[np.isnan(array)] = 1.
    sm_nan_map = imresize(nan_map, shape, interp='bilinear', mode='F')
    new_array[sm_nan_map >= 0.5] = np.nan

    return new_array

def resize2(array, shape):
    old_shape = array.shape

    array = np.repeat(array, shape[0], axis=0)
    array = np.repeat(array, shape[1], axis=1)

    array = np.reshape(array, (old_shape[0]*shape[0]*shape[1], old_shape[1]))
    array = np.mean(np.ma.masked_array(array, np.isnan(array)), axis=1)

    array = np.reshape(array, (old_shape[0], shape[0]*shape[1]))
    array = np.mean(np.ma.masked_array(array, np.isnan(array)), axis=0)

    array = np.reshape(array, shape)

    return array

def resize_w_nans(array, shape):
    arr_nan = np.isnan(array).astype('f8')

    arr_sm = self._resize2(array, shape)
    arr_sm_nan = self._resize2(arr_nan, shape)
    arr_sm[arr_sm_nan > 0.5] = np.nan

    return arr_sm

def main():

    x = np.arange(120.).reshape((12,10)) + 1
    x[x==2.] = np.nan
    x[x==1.] = np.nan
    x[x==3.] = np.nan
    x[x==11.] = np.nan
    x[x==21.] = np.nan
    x[x==22.] = np.nan
    x[x==23.] = np.nan
    x[x==4.] = np.nan
    x[x==24.] = np.nan
    x[x==25.] = np.nan
    x[x==31.] = np.nan
    x[x==32.] = np.nan
    x[x==33.] = np.nan
    x[x==41.] = np.nan
    x[x==100.] = np.nan

    print x
    max = np.nanmax(x)
    shape = (7,4)

    y = resize(x, shape)
    print y

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.imshow(x, origin='lower', aspect='auto', interpolation='none', vmax=max)
    ax2.imshow(y, origin='lower', aspect='auto', interpolation='none', vmax=max)

    plt.show()

    return 0

if __name__ == '__main__':
    main()