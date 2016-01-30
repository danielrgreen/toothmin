__author__ = 'darouet'

import numpy as np

def like_score(size, sigma, diff):
    '''
    returns a likelihood score for data of some size, with an expected sigma
    and average diff between model and data.
    :param size:    array, number of data points evaluated
    :param sigma:   float, expected error of each data point
    :param diff:    float, average model-data difference
    :return:
    '''

    diff_array = np.ones(size)*diff
    sigma_array = np.ones(size)*sigma

    return np.sum((diff_array/sigma_array)**2)

def main():

    for x in np.linspace(0.1, 10., 100):
        print like_score(10., 0.25, x)



    return 0
if __name__ == '__main__':
    main()