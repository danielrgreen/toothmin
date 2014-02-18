#
# Copyright Daniel Green, 2013
#
# All who need to make use of this program or modify it
# should do so according to their need.
#
#
#
#
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg

def sample_rain(fname1):
    '''
    reads 12 months of rainfall from an excel file into a numpy array.
    
    inputs:     filename
    outputs:    numpy vector
    '''
    
    # reads the file data for sample
    s = open(fname1, 'r')
    txt = s.read()
    s.close()
    
    # turns rainfall data into array of floats
    sample = [float(x) for x in txt.split(',')]
    sample = np.log(sample)
    sample = np.array(sample)
    
    return sample

def load_place(fname2):
    '''
    Loads rainfall data from csv format, where every year
    is one row and every month a column, into a numpy array.
    
    inputs:     csv filename
    outputs:    numpy array with years x months rainfall
    
    shape = (year, month)
    '''
    
    # reads file data for site
    f = open(fname2, 'r')
    txt = f.read()
    f.close()
    
    # takes rainfall data and places it into a year x month array
    lines = [l.lstrip().rstrip() for l in txt.splitlines()]
    rainfall = []
    for l in lines:
        year = [float(x) for x in l.split(',')]
        rainfall.append(year)
    rainfall = np.array(rainfall)

    # Don't allow rainfall to go below 1 mm/month
    rainfall = np.sqrt(rainfall*rainfall + 1)

    return rainfall

def calc_rainfall_stats(rainfall):
    '''
    Takes the years of monthly rainfall records and calculates
    monthly variance, covariation.
    
    inputs: numpy array with years x months rainfall
    outputs: log covariance matrix and monthly rainfall means
    '''
    
    # Log transform site rainfall
    log_rainfall = np.log(rainfall)
    n_years = rainfall.shape[0]
    
    # Find mean and variance for each month
    mean_log = np.mean(log_rainfall, axis=0)
    std_log = np.std(log_rainfall, axis=0)
    cov_log = np.cov(log_rainfall, rowvar=0)
        
    return mean_log, cov_log

def gen_rain(mean_log, cov_log):
    '''
    Randomly generates additional years of rainfall data
    based on log mean and covariance matrix.
    
    inputs:     monthly rainfall mean, covariance matrix
    outputs:    additional years rainfall data
    '''

    # Generate random monthly rains from distribution
    x = np.random.multivariate_normal(mean_log, cov_log, size=120)

    return x

def hotelling(x, sample):
    '''
    Calculates t2 and F distribution probability that a sample
    came from the rainfall distribution x at a site.
    
    inputs:     x (rainfall at a site)
                sample (one year rainfall)
    outputs:    t2 (hotelling's t2 statistic)
                F (F statistic)
    '''

    n = 30. # Number of years data collected for comparison site
    k = 12. # Number of time points per year; 12 months per year
    rainfall_means = np.mean(x, axis=0)
    rainfall_means = np.array(rainfall_means) # From comparison site
    diffs = rainfall_means - sample # Mean differences per month
    transpose = np.transpose(diffs) # Transpose of the differences
    both = np.array(x - sample) # 
    both_cov = np.cov(both, rowvar=0)
    both_cov_inv = linalg.inv(both_cov)
    
    
    t2 = n * np.dot(diffs, np.dot(both_cov_inv, transpose))
    F = (n - k) / ((n - 1.) * k) * t2
    
    return t2, F

def evaluate_site(fname1, fname2):

    sample = sample_rain(fname1)

    rainfall = load_place(fname2)

    mean_log, cov_log = calc_rainfall_stats(rainfall)

    x = gen_rain(mean_log, cov_log)

    t2, F = hotelling(x, sample)
    p = t2, F

    return x, p

def main():
    
    x1, p1 = evaluate_site('samples.csv', 'rainaddis.csv')
    x2, p2 = evaluate_site('samples.csv', 'rainmarsabit.csv')
    x3, p3 = evaluate_site('samples.csv', 'rainmoyale.csv')
    x4, p4 = evaluate_site('samples.csv', 'rainkitale.csv')
    x5, p5 = evaluate_site('samples.csv', 'raineldoret.csv')
    x6, p6 = evaluate_site('samples.csv', 'rainlodwar.csv')
    x7, p7 = evaluate_site('samples.csv', 'rainwajir.csv')

    print p1
    print p2
    print p3
    print p4
    print p5
    print p6
    print p7

    sample = sample_rain('samples.csv')
    s = np.exp(sample)
    
    # Plot randomly generated rains
    fig = plt.figure()
    
    ax = fig.add_subplot(3,3,1)
    
    # plot rainfall squared (expanding the log normal)
    ax.plot(np.arange(12)+1, s, c='r', alpha=1)
    for xx in np.exp(x1):
        ax.plot(np.arange(12)+1, xx, c='b', alpha=0.1)
    ax.set_ylim(0., 500.)
    ax.set_xlim(1., 12.)
    ax.set_title(r'Addis')


    ax = fig.add_subplot(3,3,2)
    
    # plot rainfall squared (expanding the log normal)
    ax.plot(np.arange(12)+1, s, c='r', alpha=1)
    for xx in np.exp(x2):
        ax.plot(np.arange(12)+1, xx, c='b', alpha=0.1)
    ax.set_ylim(0., 500.)
    ax.set_xlim(1., 12.)
    ax.set_title(r'Marsabit')

    ax = fig.add_subplot(3,3,3)
    
    # plot rainfall squared (expanding the log normal)
    ax.plot(np.arange(12)+1, s, c='r', alpha=1)
    for xx in np.exp(x3):
        ax.plot(np.arange(12)+1, xx, c='b', alpha=0.1)
    ax.set_ylim(0., 500.)
    ax.set_xlim(1., 12.)
    ax.set_title(r'Moyale')

    ax = fig.add_subplot(3,3,4)
    
    # plot rainfall squared (expanding the log normal)
    ax.plot(np.arange(12)+1, s, c='r', alpha=1)
    for xx in np.exp(x4):
        ax.plot(np.arange(12)+1, xx, c='b', alpha=0.1)
    ax.set_ylim(0., 500.)
    ax.set_xlim(1., 12.)
    ax.set_title(r'Kitale')

    
    ax = fig.add_subplot(3,3,5)
    
    # plot rainfall squared (expanding the log normal)
    ax.plot(np.arange(12)+1, s, c='r', alpha=1)
    for xx in np.exp(x5):
        ax.plot(np.arange(12)+1, xx, c='b', alpha=0.1)
    ax.set_ylim(0., 500.)
    ax.set_xlim(1., 12.)
    ax.set_title(r'Eldoret')


    ax = fig.add_subplot(3,3,6)
    
    # plot rainfall squared (expanding the log normal)
    ax.plot(np.arange(12)+1, s, c='r', alpha=1)
    for xx in np.exp(x6):
        ax.plot(np.arange(12)+1, xx, c='b', alpha=0.1)
    ax.set_ylim(0., 500.)
    ax.set_xlim(1., 12.)
    ax.set_title(r'Lodwar')


    ax = fig.add_subplot(3,3,7)
    
    # plot rainfall squared (expanding the log normal)
    ax.plot(np.arange(12)+1, s, c='r', alpha=1)
    for xx in np.exp(x7):
        ax.plot(np.arange(12)+1, xx, c='b', alpha=0.1)
    ax.set_ylim(0., 600.)
    ax.set_xlim(1., 12.)
    ax.set_title(r'Wajir')

    fig.subplots_adjust(hspace=.5)
    fig.savefig('rainpick20130927.png', dpi=500, edgecolor='none')
    plt.show()

   
   
   
   
    return 0

if __name__ == '__main__':
    main()
