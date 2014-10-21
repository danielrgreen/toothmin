# Simple imresize with NaNs

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

    # Resize
    axis1ravel = np.ravel(axis1)
    axis1stack = np.reshape(axis1ravel, (shape[0]*shape[1]*oldshape[0],oldshape[1]))
    m_axis1stack = np.ma.masked_array(axis1stack,np.isnan(axis1stack))
    axis1mean = np.mean(m_axis1stack, axis = 1)
    axis2stack = np.reshape(axis1mean, (shape[0]*shape[1], oldshape[0]))
    m_axis2stack = np.ma.masked_array(axis2stack,np.isnan(axis2stack))
    axis2mean = np.mean(m_axis2stack, axis=1)
    new_array = np.reshape(axis2mean, shape)

    # Add back in NaNs
    nan_map = np.zeros(oldshape)
    nan_map[np.isnan(array)] = 1.
    sm_nan_map = imresize(nan_map, shape, interp='bilinear', mode='F')
    new_array[sm_nan_map >= 0.5] = np.nan
            
    return new_array

def main():

    array = np.arange(80.).reshape(8,10)
    array[array==2.] = np.nan
    array[array==65.] = np.nan
    array[array==66.] = np.nan
    array[array==55.] = np.nan
    shape = (5,6)
    new_array = resize(array, shape)

    print 'original array = ', array
    print 'new array = ', new_array

    return 0
if __name__ == '__main__':
    main()
