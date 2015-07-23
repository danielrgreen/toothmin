__author__ = 'darouet'

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import nlopt
from time import time

def est_tooth_extension(ext_param, **kwargs): # days, amplitude, slope, offset
    '''
    '''
    p35_model_extension, tooth_35p, p70_model_extension, tooth_70p = extension(*ext_param, **kwargs)
    score = compare(p35_model_extension, tooth_35p, p70_model_extension, tooth_70p)

    return score, p35_model_extension, p70_model_extension

def extension(p35_amplitude, p35_slope, p35_offset, p70_amplitude, p70_slope, p70_offset, M1_days, tooth_35p, tooth_70p, tooth_70p_days):

    M1_height_max = 35. # in millimeters
    p35_model_extension = (p35_amplitude * spec.erf(p35_slope * (M1_days - p35_offset))) + (M1_height_max - p35_amplitude)
    p70_model_extension = (p70_amplitude * spec.erf(p70_slope * (tooth_70p_days - p70_offset))) + (M1_height_max - p70_amplitude)

    #k_height_max = 32. # in millimeters
    #kext_model_rate = (spec.erfinv((kext_amplitude + kday - k_height_max) / kext_amplitude) + kext_offset * kext_slope) / kext_slope

    return p35_model_extension, tooth_35p, p70_model_extension, tooth_70p

def compare(p35_model_extension, tooth_35p, p70_model_extension, tooth_70p):

    sigma = 6. # in mm

    p35_score = (p35_model_extension - tooth_35p)**2. / sigma**2
    p35_score[~np.isfinite(p35_score)] = 10000000.
    p35_score = (1. / (2. * np.pi * sigma**2)) * np.exp(-.5*(p35_score))
    p35_score = np.product(p35_score)

    p70_score = (p70_model_extension - tooth_70p)**2. / sigma**2.
    p70_score[~np.isfinite(p70_score)] = 10000000.
    p70_score = (1. / (2. * np.pi * sigma**2.)) * np.exp(-.5*(p70_score))
    p70_score = np.product(p70_score)

    #k_score = (kext_model_rate - kext)**2. / sigma**2.
    #k_score[~np.isfinite(k_score)] = 10000000.
    #k_score = (1. / (2. * np.pi * sigma**2.)) * np.exp(-.5*(k_score))
    #k_score = np.product(k_score)

    data_score = p35_score * p70_score #+ k_score
    #prior_score = prior(m2_m1_converted, M1_initiation, M2_initiation)
    print -1 * data_score #* prior_score

    return -1 * data_score #* prior_score

def prior(m2_m1_converted, M1_initiation, M2_initiation):

    M1_initiation_expected = -49.
    M2_initiation_expected = 84.
    sigma = 12.

    M1_initiation_score = ((M1_initiation - M1_initiation_expected)**2.) / sigma**2.
    if np.isfinite(M1_initiation_score) != True:
        M1_initiation_score = 1000.
    M1_initiation_score = (1. / (2. * np.pi * sigma**2.)) * np.exp(-.5*(M1_initiation_score))

    M2_initiation_score = ((M2_initiation - M2_initiation_expected)**2.) / sigma**2.
    if np.isfinite(M2_initiation_score) != True:
        M2_initiation_score = 1000.
    M2_initiation_score = (1. / (2. * np.pi * sigma**2.)) * np.exp(-.5*(M2_initiation_score))
    prior_score = M1_initiation_score * M2_initiation_score

    return prior_score

def convert(m2_m1_conversion, M1_amplitude, M1_slope, M1_offset, M2_amplitude, M2_slope, M2_offset):
    '''
    '''
    M2_height = M2_amplitude*spec.erf(M2_slope*(m2_m1_conversion-M2_offset))+(41.-M2_amplitude) # max at 40.5 optimized with synchrotron data set on nlopt
    M2_percent = M2_height / 41.
    M1_max_height = 35.
    M1_height = M2_percent * M1_max_height
    m2_m1_converted = (spec.erfinv((M1_amplitude+M1_height-M1_max_height)/M1_amplitude) + (M1_offset*M1_slope)) / M1_slope

    return m2_m1_converted

def final_convert(m2_m1_conversion, M1_amplitude, M1_slope, M1_offset, M2_amplitude, M2_slope, M2_offset):

    M1_height = M1_amplitude*spec.erf(M1_slope*(m2_m1_conversion+M1_offset))+(35-M1_amplitude) # max at 40.5 optimized with synchrotron data set on nlopt
    M1_percent = M1_height / 35
    M2_max_height = 41.
    M2_height = M1_percent * M2_max_height
    m2_m1_converted = (spec.erfinv((M2_amplitude+M2_height-M2_max_height)/M2_amplitude) + (M2_offset*M2_slope)) / M2_slope

    return m2_m1_converted

def optimize_curve(M1_days, M1_data_extension, tooth_35p, tooth_70p, tooth_70p_days, kday, kext, M2_days, M2_data_extension, **fit_kwargs):
    #m2_m1_conversion = np.array([84., 202., 263., 450., 500.])
    #fit_kwargs['m2_m1_conversion'] = m2_m1_conversion
    #fit_kwargs['M1_data_extension'] = M1_data_extension
    fit_kwargs['M1_days'] = M1_days
    #fit_kwargs['M2_data_extension'] = M2_data_extension
    #fit_kwargs['M2_days'] = M2_days
    fit_kwargs['tooth_35p'] = tooth_35p
    fit_kwargs['tooth_70p'] = tooth_70p
    #fit_kwargs['kday'] = kday
    #fit_kwargs['kext'] = kext
    fit_kwargs['tooth_70p_days'] = tooth_70p_days



    # Model a M1 combined with different M2 possibilities
    #m2_m1_params = np.array([56.031, .003240, 1.1572, 41., 21.820, .007889, 29.118, 35.]) # No limits, 'a', 2000k

    # Model c M1 combined with different M2 possibilities
    #m2_m1_params = np.array([56.031, .003240, 1.1572, 41., 29.764, .005890, -19.482, 35.]) # No limits, 'a', 2000k



    t1 = time()

    f_objective = lambda x, grad: est_tooth_extension(x, **fit_kwargs)[0]

    local_opt = nlopt.opt(nlopt.LN_COBYLA, 6)
    local_opt.set_xtol_abs(.01)
    local_opt.set_lower_bounds([20., .004, 30., 18., .0035, 50.])
    local_opt.set_upper_bounds([50., .0075, 60., 60., .008, 90.])
    local_opt.set_min_objective(f_objective)

    global_opt = nlopt.opt(nlopt.G_MLSL_LDS, 6)
    global_opt.set_maxeval(300000)
    global_opt.set_lower_bounds([20., .004, 30., 18., .0035, 50.])
    global_opt.set_upper_bounds([50., .0075, 60., 60., .008, 90.])
    global_opt.set_min_objective(f_objective)
    global_opt.set_local_optimizer(local_opt)
    global_opt.set_population(6)

    print 'Running global optimizer ...'
    x_opt = global_opt.optimize([40., .005, 40., 40., .0045, 60])

    minf = global_opt.last_optimum_value()
    print "minimum value = ", minf
    print "result code = ", global_opt.last_optimize_result()
    print 'result at ', x_opt

    t2 = time()
    run_time = t2-t1

    M1_params = np.array([21.820, .007889, 29.118])
    M1_height_max = 35.

    days = np.linspace(-100, 350, 451)
    p35_model_extension, tooth_35p, p70_model_extension, tooth_70p = extension(x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], days, tooth_35p, tooth_70p, days)
    M1_model_extension = (M1_params[0] * spec.erf(M1_params[1] * (days - M1_params[2]))) + (M1_height_max - M1_params[0])
    p35_diff_extension = np.diff(p35_model_extension) * 1000.
    p70_diff_extension = np.diff(p70_model_extension) * 1000.
    M1_diff_extension = np.diff(M1_model_extension) * 1000.

    local_method = 'cobyla'
    global_method = 'msds'
    textstr = '%.3f, %.6f, %.3f, \n%.3f, %.6f, %.3f, \nmin = %.3g, time = %.1f seconds \n%s, %s' % (x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], minf, run_time, local_method, global_method)
    print textstr

    fig = plt.figure()
    #fig = plt.figure(figsize=(3.3194, 2.4028))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax2.plot(days[1::4], M1_diff_extension[::4], 'b.', label=r'$ \mathrm{extension} \ \mathrm{rate} $', alpha=.5)
    ax2.plot(days[1::4], p35_diff_extension[::4], 'm.', label=r'$ \mathrm{maturation} \ \mathrm{rate} $', alpha=.5)
    ax2.plot(days[1::4], p70_diff_extension[::4], 'r.', label=r'$ \mathrm{completion} \ \mathrm{rate} $', alpha=.5)
    #ax2.plot(days[1::4], kext_model_rate[::4], 'g.', label=r' \mathrm{histology} \ \Delta $', alpha=.5)
    ax2.plot(kday, kext, marker='D', fillstyle='none', linestyle='none', color='g', label=r'$ \mathrm{Kierdorf} \ \mathrm{rate} $')
    ax2.set_ylim([0,250])
    ax2.set_xlim([-50,350])
    ax1.plot(M1_days, M1_data_extension, marker='o', linestyle='none', color='b', label=r'$ \mathrm{extension} \ \mathrm{(observed)} $')
    ax1.plot(days, M1_model_extension, linestyle='-', color='b', label=b'$ \mathrm{extension} \ \mathrm{(fitted)} $')
    ax1.plot(M1_days, tooth_35p, marker='o', linestyle='none', color='m', label=r'$ \mathrm{maturation} \ \mathrm{(observed)} $')
    ax1.plot(days, p35_model_extension, linestyle='-', color='m', label=r'$ \mathrm{maturation} \ \mathrm{(fitted)} $')
    ax1.plot(tooth_70p_days, tooth_70p, marker='o', linestyle='none', color='r', label=r'$ \mathrm{completion} \ \mathrm{(observed)} $')
    ax1.plot(days, p70_model_extension, linestyle='-', color='r', label=r'$ \mathrm{completion} \ \mathrm{(fitted)} $')
    ax1.set_ylim([0,40])
    ax1.set_xlim([-50,350])
    #plt.title('M1 extension, maturation onset and completion over time', fontsize=6)
    #ax1.set_xlabel('Days after birth', fontsize=6)
    #ax1.set_ylabel('Progress from cusp tip in mm', fontsize=6)
    #ax2.set_ylabel('$ \mathrm{Extension,} \ \mathrm{maturation} \ \mathrm{and} \ \mathrm{completion,} \ \mu \mathrm{m/day} $')
    #ax2.set_ylabel('Extension, maturation and completion, ' + '$ \mu $' + 'm/day', fontsize=6)
    #ax2.legend(loc='lower right', fancybox=True, framealpha=0.8)
    #ax.legend(loc='upper right', fancybox=True, framealpha=0.8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='center right', fontsize=6)
    ax1.text(100, 2, textstr, fontsize=6)

    plt.show()

def main():

    # M2 extension data histology only
    #M2_days = np.array([203., 223., 265., 285., 510.])
    #M2_data_extension = np.array([24.59, 26.23, 30.60, 32.50, 41.79])
    # M2 extension data full set including synchrotron + histology
    #M2_days = np.array([88., 92., 97., 100., 101., 101., 105., 124., 127., 140., 140., 159., 167., 174., 179., 202., 203., 222., 223., 251., 265., 285., 510., 510., 510., 510.])
    #M2_data_extension = np.array([2.22, 4.43, 6.23, 3.14, 7.16, 3.19, 6.74, 6.23, 9.61, 11.8, 12.3, 18.4, 16.9, 17.9, 18.9, 21, 24.59, 20.6, 26.23, 20.1, 30.60, 32.50, 41.79, 39.61, 39.39, 42.3])
    # M2 extension data partial set including synchrotron but not histology
    M2_days = np.array([88., 92., 97., 100., 101., 101., 105., 124., 127., 140., 140., 159., 167., 174., 179., 202., 222., 251., 510., 510., 510., 510.])
    M2_data_extension = np.array([2.22, 4.43, 6.23, 3.14, 7.16, 3.19, 6.74, 6.23, 9.61, 11.8, 12.3, 18.4, 16.9, 17.9, 18.9, 21, 20.6, 20.1, 41.79, 39.61, 39.39, 42.3])
    # M1 data, extension
    M1_days = np.array([1., 9., 11., 19., 21., 30., 31., 31., 38., 42., 54., 56., 56., 58., 61., 66., 68., 73., 78., 84., 88., 92., 97., 100., 101., 101., 104., 105., 124., 127., 140., 140., 157., 167., 173., 174., 179., 202., 222., 235., 238., 251., 259., 274.])
    M1_data_extension = np.array([9.38, 8.05, 11.32, 9.43, 13.34, 16.19, 13.85, 15.96, 15.32, 14.21, 17.99, 19.32, 19.32, 18.31, 17.53, 18.68, 18.49, 22.08, 23.14, 19.92, 27.97, 24.38, 25.53, 29.07, 27.65, 26.27, 27.55, 24.33, 29.03, 29.07, 30.36, 31.79, 31.37, 31.28, 35.79, 29.81, 31.79, 34.04, 33.21, 34.50, 33.76, 33.40, 36.34, 33.63])
    # M1 mineralization data.
    tooth_35p = np.array([2.07, 1.52, 2.39, 2.67, 4.60, 7.04, 4.55, 5.47, 5.47, 4.83, 8.19, 9.98, 9.75, 9.89, 9.29, 10.40, 8.88, 12.88, 14.49, 11.96, 19.04, 17.48, 16.74, 20.61, 18.86, 17.34, 19.92, 14.67, 22.03, 22.91, 24.47, 26.08, 26.13, 25.94, 34.45, 25.35, 26.91, 30.08, 30.68, 33.49, 28.29, 28.34, 35.83, 33.63])
    tooth_70p = np.array([2.76, 3.68, 4.60, 4.69, 5.11, 4.88, 5.75, 4.74, 7.36, 8.97, 5.15, 13.62, 12.33, 11.13, 14.54, 13.25, 10.81, 13.25, 8.69, 15.78, 17.16, 19.27, 20.56, 19.87, 21.48, 30.27, 20.01, 22.17, 24.56, 26.27, 28.11, 23.55, 23.83, 36.34, 29.49])
    tooth_70p_days = np.array([30., 54., 56., 56., 58., 61., 66., 68., 73., 78., 84., 88., 92., 97., 100., 101., 101., 104., 105., 124., 127., 140., 140., 157., 167., 173., 174., 179., 202., 222., 235., 238., 251., 259., 274.])
    # Kierdorf 2013 data
    kday = np.array([7.5, 21.5, 49.5, 63.3, 73., 154., 168., 185.5, 199., 212.5, 230., 247.5])
    kext = np.array([177.6, 168.4, 180., 131.7, 109.9, 40.4, 35.5, 35.0, 33.2, 27.1, 28.0, 29.3])

    print tooth_35p.shape
    print tooth_70p.shape
    print tooth_70p_days.shape

    optimize_curve(M1_days, M1_data_extension, tooth_35p, tooth_70p, tooth_70p_days, kday, kext, M2_days, M2_data_extension)

    '''
    m2days = np.array([70., 80., 202., 263., 450., 500.])
    m1days = convert(m2days)

    for i in xrange(m2days.size):
        print m2days[i], m1days[i]
    print 'switch length = ', m1days[3]-m1days[2]
    '''

    return 0
if __name__ == '__main__':
    main()