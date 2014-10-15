# Simple imresize with NaNs

import numpy as np


def resize(array, shape):
    '''
    Takes an array and broadcasts it into a new shape.
    '''
    oldshape = array.shape
    axis0 = np.repeat(array, shape[0], axis=0)
    axis1 = np.repeat(axis0, shape[1], axis=1)

    axis1ravel = np.ravel(axis1)
    axis1stack = np.reshape(axis1ravel, (shape[0]*shape[1]*oldshape[0],oldshape[1]))
    axis1mean = np.mean(axis1stack, axis = 1)
    axis2stack = np.reshape(axis1mean, (shape[0]*shape[1], oldshape[0]))
    axis2mean = np.mean(axis2stack, axis=1)
    new_array = np.reshape(axis2mean, shape)

    return new_array

def main():

    array = np.arange(80.).reshape(8,10)
    array[array==2.] = np.nan
    array[array==65.] = np.nan
    shape = (5,6)
    new_array = resize(array, shape)

    print 'original array = ', array
    print 'new array = ', new_array

    return 0
if __name__ == '__main__':
    main()
