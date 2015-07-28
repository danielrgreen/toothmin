__author__ = 'darouet'

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import nlopt
from time import time

def est_tooth_extension(ext_param, **kwargs): # days, amplitude, slope, offset
    '''
    '''
    M1_model_extension, M1_data_extension, M2_model_extension, M2_data_extension, m2_m1_converted, M1_initiation, M2_initiation = extension(*ext_param, **kwargs)
    score = compare(M1_model_extension, M1_data_extension, M2_model_extension, M2_data_extension, m2_m1_converted, M1_initiation, M2_initiation)

    return score, M1_model_extension, M2_model_extension

def extension(M1_amplitude, M1_slope, M1_offset, M2_amplitude, M2_slope, M2_offset, M1_days, M1_data_extension, M2_days, M2_data_extension, m2_m1_conversion):

    M1_height_max = 35. # in millimeters
    M1_model_extension = (M1_amplitude * spec.erf(M1_slope * (M1_days - M1_offset))) + (M1_height_max - M1_amplitude)
    M1_initiation = (spec.erfinv((M1_amplitude - M1_height_max) / M1_amplitude) + M1_offset * M1_slope) / M1_slope

    M2_height_max = 41. # in millimeters
    M2_model_extension = (M2_amplitude * spec.erf(M2_slope * (M2_days - M2_offset))) + (M2_height_max - M2_amplitude)
    M2_initiation = (spec.erfinv((M2_amplitude - M2_height_max) / M2_amplitude) + M2_offset * M2_slope) / M2_slope

    m2_m1_converted = convert(m2_m1_conversion, M1_amplitude, M1_slope, M1_offset, M2_amplitude, M2_slope, M2_offset)

    return M1_model_extension, M1_data_extension, M2_model_extension, M2_data_extension, m2_m1_converted, M1_initiation, M2_initiation

def compare(M1_model_extension, M1_data_extension, M2_model_extension, M2_data_extension, m2_m1_converted, M1_initiation, M2_initiation):

    sigma = 6. # in mm

    M1_score = (M1_model_extension - M1_data_extension)**2. / sigma**2
    M1_score[~np.isfinite(M1_score)] = 10000000.
    M1_score = (1. / (2. * np.pi * sigma**2)) * np.exp(-.5*(M1_score))
    M1_score = np.product(M1_score)

    M2_score = (M2_model_extension - M2_data_extension)**2. / sigma**2.
    M2_score[~np.isfinite(M2_score)] = 10000000.
    M2_score = (1. / (2. * np.pi * sigma**2.)) * np.exp(-.5*(M2_score))
    M2_score = np.product(M2_score)

    data_score = M1_score * M2_score
    prior_score = prior(m2_m1_converted, M1_initiation, M2_initiation)
    print data_score * prior_score * -1.

    return data_score * prior_score * -1.

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

def optimize_curve(M1_days, M1_data_extension, M2_days, M2_data_extension, **fit_kwargs):
    blood_day_measures = np.array([(57., -5.71), (199., -4.96), (203., -10.34), (207., -12.21), (211., -13.14), (215., -13.49), (219., -13.16), (239., -13.46), (261., -13.29), (281., -4.87), (289., -4.97), (297., -4.60), (309., -4.94)])
    blood_days = np.array([i[0] for i in blood_day_measures])
    blood_measures = np.array([i[1] for i in blood_day_measures])
    water_iso_day_measures = np.array([(198., -6.6), (199., -19.4), (219., -19.3), (261., -19.4)])
    water_iso_days = np.array([i[0] for i in water_iso_day_measures])
    water_iso_measures = np.array([i[1] for i in water_iso_day_measures])
    m2_m1_conversion = np.array([84., 199., 261., 450., 500.])
    fit_kwargs['m2_m1_conversion'] = m2_m1_conversion
    fit_kwargs['M1_data_extension'] = M1_data_extension
    fit_kwargs['M1_days'] = M1_days
    fit_kwargs['M2_data_extension'] = M2_data_extension
    fit_kwargs['M2_days'] = M2_days

    t1 = time()

    f_objective = lambda x, grad: est_tooth_extension(x, **fit_kwargs)[0]

    local_opt = nlopt.opt(nlopt.LN_COBYLA, 6)
    local_opt.set_xtol_abs(.01)
    local_opt.set_lower_bounds([15., .0035, -50., 45., .003, 20.])
    local_opt.set_upper_bounds([80., .0055, -10., 65., .0045, 40.])
    local_opt.set_min_objective(f_objective)

    global_opt = nlopt.opt(nlopt.G_MLSL_LDS, 6)
    global_opt.set_maxeval(180)
    global_opt.set_lower_bounds([15., .0035, -50., 45., .003, 20.])
    global_opt.set_upper_bounds([80., .0055, -10., 65., .0045, 40.])
    global_opt.set_min_objective(f_objective)
    global_opt.set_local_optimizer(local_opt)
    global_opt.set_population(6)

    print 'Running global optimizer ...'
    x_opt = global_opt.optimize([40., .005, -45., 50., .004, 29.])

    minf = global_opt.last_optimum_value()
    print "minimum value = ", minf
    print "result code = ", global_opt.last_optimize_result()

    t2 = time()
    run_time = t2-t1

    M1_initiation = (spec.erfinv((x_opt[0] - 35) / x_opt[0]) + x_opt[2] * x_opt[1]) / x_opt[1]
    M2_initiation = (spec.erfinv((x_opt[3] - 35) / x_opt[3]) + x_opt[5] * x_opt[4]) / x_opt[4]

    m2_m1_converted = convert(m2_m1_conversion, *x_opt)
    switch_length =  m2_m1_converted[2]-m2_m1_converted[1]

    days = np.linspace(-100, 550, 651)
    # Plot test result
    testp = np.array([27.792, 0.0066885, -4.4215, 52.787, 0.003353, 17.1276])
    M1_model_extension, M1_data_extension, M2_model_extension, M2_data_extension, m2_m1_converted, M1_initiation, M2_initiation = extension(testp[0], testp[1], testp[2], testp[3], testp[4], testp[5], days, M1_data_extension, days, M2_data_extension, m2_m1_conversion)
    # Plot optimized result
    #M1_model_extension, M1_data_extension, M2_model_extension, M2_data_extension, m2_m1_converted, M1_initiation, M2_initiation = extension(x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], days, M1_data_extension, days, M2_data_extension, m2_m1_conversion)
    M1_diff_extension = np.diff(M1_model_extension) * 1000.
    M2_diff_extension = np.diff(M2_model_extension) * 1000.
    prior_score = prior(m2_m1_converted, M1_initiation, M2_initiation)
    print 'prior score, data score = ', prior_score, minf/prior_score

    local_method = 'cobyla'
    global_method = 'msds'
    textstr = '%.3f, %.6f, %.3f, \n%.3f, %.6f, %.3f, \nmin = %.3g, time = %.1f seconds \nm1start = %.2f, m2start = %.2f \nM1_est_switch_len = %.2f \n%s, %s' % (x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], minf, run_time, M1_initiation, M2_initiation, switch_length, local_method, global_method)
    #textstr2 =
    #textstr3 =
    #textstr4 =
    print textstr

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax2.plot(days[1::4], M1_diff_extension[::4], 'b.', label=r'$M1 \mathrm{extension} \ \Delta $', alpha=.5)
    ax2.plot(days[1::4], M2_diff_extension[::4], 'g.', label=r'$M2 \mathrm{extension} \ \Delta $', alpha=.5)
    ax2.set_ylim([0,250])
    ax2.set_xlim([-100,550])
    ax1.plot(M1_days, M1_data_extension, marker='o', linestyle='none', color='b', label=r'$M1 \mathrm{extension} \ \mathrm{(observed)} $')
    ax1.plot(days, M1_model_extension, linestyle='-', color='b', label=r'$M1 \mathrm{extension,} \ \mathrm{optimized} $')
    ax1.plot(M2_days, M2_data_extension, marker='o', linestyle='none', color='g', label=r'$M2 \mathrm{extension} \ \mathrm{(observed)} $')
    ax1.plot(days, M2_model_extension, linestyle='-', color='g', label=r'$M2 \mathrm{extension,} \ \mathrm{optimized} $')
    ax1.set_ylim([0,45])
    ax1.set_xlim([-100,550])
    plt.title('M1 and M2 extension over time: synchrotron')
    ax1.set_xlabel('Days after birth')
    ax1.set_ylabel('Progress from cusp tip in mm')
    ax2.set_ylabel('Secretion or maturation speed in um/day')
    #ax2.legend(loc='lower right', fancybox=True, framealpha=0.8)
    #ax.legend(loc='upper right', fancybox=True, framealpha=0.8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='center right', fontsize=10)
    ax1.text(180, 5, textstr, fontsize=8)

    plt.show()

    final = final_convert(m2_m1_converted, *x_opt)
    print 'm1 switch length = ', m2_m1_converted[2]-m2_m1_converted[1]
    print 'm2 switch length = ', final[2]-final[1]

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

    optimize_curve(M1_days, M1_data_extension, M2_days, M2_data_extension)

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