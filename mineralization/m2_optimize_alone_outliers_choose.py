__author__ = 'darouet'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.special as spec
import nlopt
from time import time

score_param_list = []

def getkey(item):
    return item[0]

def est_tooth_extension(ext_param, **kwargs): # days, amplitude, slope, offset
    '''
    '''
    M1_model_extension, M1_data_extension, M1_initiation = extension(*ext_param, **kwargs)
    score = compare(M1_model_extension, M1_data_extension, M1_initiation)

    temp_list = np.array([score, ext_param[0], ext_param[1], ext_param[2]])
    score_param_list.append(temp_list)

    return score, M1_model_extension, M1_initiation

def extension(M1_amplitude, M1_slope, M1_offset, M1_days, M1_data_extension):

    M1_height_max = 41. # in millimeters
    M1_model_extension = (M1_amplitude * spec.erf(M1_slope * (M1_days - M1_offset))) + (M1_height_max - M1_amplitude)
    M1_initiation = (spec.erfinv((M1_amplitude - M1_height_max) / M1_amplitude) + M1_offset * M1_slope) / M1_slope

    return M1_model_extension, M1_data_extension, M1_initiation

def compare(M1_model_extension, M1_data_extension, M1_initiation):

    sigma = 12. # in mm

    M1_score = (M1_model_extension - M1_data_extension)**2. / sigma**2
    #M1_score[~np.isfinite(M1_score)] = 10000000.
    M1_score = (1. / (2. * np.pi * sigma**2)) * np.exp(-.5*(M1_score))
    M1_score = np.product(M1_score)

    data_score = M1_score
    prior_score = prior(M1_initiation)
    #print data_score * prior_score * -1.

    return data_score * prior_score * -1.

def prior(M1_initiation):

    M1_initiation_expected = 86.
    sigma = 12.

    M1_initiation_score = ((M1_initiation - M1_initiation_expected)**2.) / sigma**2.
    #if np.isfinite(M1_initiation_score) != True:
    #    M1_initiation_score = 1000.
    M1_initiation_score = (1. / (2. * np.pi * sigma**2.)) * np.exp(-.5*(M1_initiation_score))

    prior_score = M1_initiation_score

    return prior_score

def final_convert(m2_m1_conversion, M1_amplitude, M1_slope, M1_offset, M2_amplitude, M2_slope, M2_offset):

    M1_height = M1_amplitude*spec.erf(M1_slope*(m2_m1_conversion-M1_offset))+(35.-M1_amplitude)
    M1_percent = M1_height / 35.
    M2_max_height = 41.
    M2_height = M1_percent * M2_max_height
    m2_m1_converted = (spec.erfinv((M2_amplitude+M2_height-M2_max_height)/M2_amplitude) + (M2_offset*M2_slope)) / M2_slope

    return m2_m1_converted

def optimize_curve(M1_days, M1_data_extension, **fit_kwargs):
    m2_m1_conversion = np.array([84., 202., 263., 450., 500.])
    #fit_kwargs['m2_m1_conversion'] = m2_m1_conversion
    fit_kwargs['M1_data_extension'] = M1_data_extension
    fit_kwargs['M1_days'] = M1_days

    t1 = time()
    trials = 1

    f_objective = lambda x, grad: est_tooth_extension(x, **fit_kwargs)[0]

    local_opt = nlopt.opt(nlopt.LN_COBYLA, 3)
    local_opt.set_xtol_abs(.01)
    local_opt.set_lower_bounds([-60., .001, -260.])
    local_opt.set_upper_bounds([190., .009, 222.])
    local_opt.set_min_objective(f_objective)

    global_opt = nlopt.opt(nlopt.G_MLSL_LDS, 3)
    global_opt.set_maxeval(trials)
    global_opt.set_lower_bounds([-60., .001, -260.])
    global_opt.set_upper_bounds([190., .009, 222.])
    global_opt.set_min_objective(f_objective)
    global_opt.set_local_optimizer(local_opt)
    global_opt.set_population(3)

    print 'Running global optimizer ...'
    x_opt = global_opt.optimize([20., .0055, 20.])

    minf = global_opt.last_optimum_value()
    print "minimum value = ", minf
    print "result code = ", global_opt.last_optimize_result()

    t2 = time()
    run_time = t2-t1

    M1_initiation = (spec.erfinv((x_opt[0] - 35.) / x_opt[0]) + x_opt[2] * x_opt[1]) / x_opt[1]

    #m2_m1_converted = convert(m2_m1_conversion, *x_opt)
    #switch_length =  m2_m1_converted[2]-m2_m1_converted[1]

    days = np.linspace(-100, 550, 651)
    # Plot test result

    #testp = np.array([27.792, 0.0066885, -4.4215, 52.787, 0.003353, 17.1276])
    #M1_model_extension, M1_data_extension, M1_initiation = extension(testp[0], testp[1], testp[2], testp[3], testp[4], testp[5], days, M1_data_extension, days, M2_data_extension, m2_m1_conversion)

    # Plot optimized result
    M1_model_extension, M1_data_extension, M1_initiation = extension(x_opt[0], x_opt[1], x_opt[2], days, M1_data_extension)

    a1, s1, o1 = 67.974, 0.003352, -25.414
    a2, s2, o2 = 74., .0031, -48.
    a3, s3, o3 = 62., .0037, 0.

    M1_model_extension1, M1_data_extension1, M1_initiation1 = extension(a1, s1, o1, days, M1_data_extension)
    M1_model_extension2, M1_data_extension2, M1_initiation2 = extension(a2, s2, o2, days, M1_data_extension)
    M1_model_extension3, M1_data_extension3, M1_initiation3 = extension(a3, s3, o3, days, M1_data_extension)
    M1_diff_extension = np.diff(M1_model_extension) * 1000.
    prior_score = prior(M1_initiation)
    print 'prior score, data score = ', prior_score, minf/prior_score

    print M1_data_extension1

    local_method = 'cobyla'
    global_method = 'msds'
    textstr = '%.3f, %.6f, %.3f, \nmin = %.3g, time = %.1f seconds \nm1start = %.2f \n%s, %s' % (x_opt[0], x_opt[1], x_opt[2], minf, run_time, M1_initiation, local_method, global_method)
    textstr2 = 'blue = %.3f, %.6f, %.3f \ngreen = %.3f, %.6f, %.3f \nred = %.3f, %.6f, %.3f \n' % (a1, s1, o1, a2, s2, o2, a3, s3, o3)
    #textstr2 =
    #textstr3 =
    #textstr4 =
    print textstr

    model, data, init = extension(x_opt[0], x_opt[1], x_opt[2], M1_days, M1_data_extension)
    model_days = (spec.erfinv((x_opt[0]+data-41.)/x_opt[0]) + (x_opt[2]*x_opt[1])) / x_opt[1]
    day_diffs = model_days - M1_days
    m_day_diffs = np.ma.masked_array(day_diffs, ~np.isfinite(day_diffs))
    m_day_diffs = m_day_diffs[np.isfinite(m_day_diffs)]
    day_percentiles = np.percentile(m_day_diffs, (5, 50, 95))
    mean = np.mean(m_day_diffs)
    variance = np.var(m_day_diffs)
    sigma = np.sqrt(variance)
    print day_percentiles
    print mean, sigma, variance

    print m_day_diffs

    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.hist(m_day_diffs, bins=np.linspace(min(m_day_diffs), max(m_day_diffs), 10))
    #plt.show()


    # M2 extension data histology only
    hist_days = np.array([203., 223., 265., 285., 510.]) # ************* Last is 510 ***********
    hist_extension = np.array([24.59, 26.23, 30.60, 32.50, 41.79])

    # M2 synchrotron outliers
    outlier_days = np.array([251.])
    outlier_extension = np.array([20.1])

    score_param_list.sort(key=getkey, reverse=True)
    score_param_array_template = np.array(score_param_list)
    score_param_array = np.ones(score_param_array_template.shape[0]*score_param_array_template.shape[1]).reshape(score_param_array_template.shape)
    for a,b in enumerate(score_param_array_template):
        for i,j in enumerate(b):
            score_param_array[a,i] = j

    t_save = time()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax2.plot(days[1::4], M1_diff_extension[::4], 'g.', label=r'$M2 \ \mathrm{extension} \ \Delta $', alpha=.5)
    ax2.set_ylim([0,250])
    ax2.set_xlim([-100,550])
    ax1.plot(M1_days, M1_data_extension, marker='o', linestyle='none', color='b', label=r'$M2 \ \mathrm{extension} \ \mathrm{(observed)} $')
    ax1.plot(days, M1_model_extension1, linestyle='-', color='b', label=r'$M2 \ \mathrm{extension,} \ \mathrm{optimized} $', lw=2)
    ax1.plot(days, M1_model_extension2, linestyle='-', color='g', label=r'$M2 \ \mathrm{extension,} \ \mathrm{optimized} $')
    ax1.plot(days, M1_model_extension3, linestyle='-', color='r', label=r'$M2 \ \mathrm{extension,} \ \mathrm{optimized} $')
    ax1.plot(hist_days, hist_extension, marker='o', linestyle='none', color='r', label=r'$962 \ \mathrm{ extension} \ \mathrm{(observed)} $')
    ax1.plot(outlier_days, outlier_extension, marker='o', linestyle='none', color='g', mfc='none')
    min_all, max_all = min(score_param_array[:,0]), max(score_param_array[:,0])
    #for a,b in enumerate(score_param_array[:2000:10,:]):
    #    score = b[0]
    #    range_score = max_all - min_all
    #    r_pct = (score-min_all)/range_score
    #    g_pct = 1. - r_pct
    #    m1, d1, i1 = extension(b[1], b[2], b[3], days, M1_data_extension)
    #    ax1.plot(days, m1, linestyle='-', color=(r_pct, 0, g_pct), alpha=.05)
    ax1.set_ylim([0,45])
    ax1.set_xlim([0,550])
    plt.title('M2 extension over time: synchrotron data')
    ax1.set_xlabel('Days after birth')
    ax1.set_ylabel('Progress from cusp tip in mm')
    ax2.set_ylabel('Secretion or maturation speed in um/day')
    #ax2.legend(loc='lower right', fancybox=True, framealpha=0.8)
    #ax.legend(loc='upper right', fancybox=True, framealpha=0.8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='center right', fontsize=10)
    ax1.text(280, 5, textstr2, fontsize=8)
    #fig.savefig('m2_opt_86_outlier_100k_{0}.svg'.format(t_save))

    plt.show()
    '''
    score_param_array_template = np.array(score_param_list)
    score_param_array = np.ones(score_param_array_template.shape[0]*score_param_array_template.shape[1]).reshape(score_param_array_template.shape)
    for a,b in enumerate(score_param_array_template):
        for i,j in enumerate(b):
            score_param_array[a,i] = j

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.scatter(score_param_array[:,2], score_param_array[:,1], c=score_param_array[:,0]*-1, marker='8', s=50, lw=0, cmap='jet')
    ax1.set_ylim(10, 190)
    ax1.set_xlim(.001, .0085)
    ax1.set_xlabel('Slope')
    ax1.set_ylabel('Amplitude')
    ax2 = fig.add_subplot(2,1,2)
    ax2.scatter(score_param_array[:,2], score_param_array[:,3], c=score_param_array[:,0]*-1, marker='8', s=50, lw=0, cmap='jet')
    ax2.set_ylim(-260, 205)
    ax2.set_xlim(.001, .0085)
    ax2.set_xlabel('Slope')
    ax2.set_ylabel('Offset')
    plt.show()

    return 0

    #fig = plt.figure()
    #ax1 = fig.add_subplot(3,1,1)
    #ax1.hist(score_param_array[:,1], bins=np.linspace(min(score_param_array[:,1]), max(score_param_array[:,1]), 100), alpha=.6)
    #ax2 = fig.add_subplot(3,1,2)
    #ax2.hist(score_param_array[:,2], bins=np.linspace(min(score_param_array[:,2]), max(score_param_array[:,2]), 100), alpha=.6)
    #ax3 = fig.add_subplot(3,1,3)
    #ax3.hist(score_param_array[:,3], bins=np.linspace(min(score_param_array[:,3]), max(score_param_array[:,3]), 100), alpha=.6)
    #fig.savefig('test_histo_M2_times_{0}.svg'.format(t_save), dpi=300, bbox_inches='tight')
    #plt.show()
    '''


def main():

    # M2 extension data histology only
    #M2_days = np.array([203., 223., 265., 285., 510.])
    #M2_data_extension = np.array([24.59, 26.23, 30.60, 32.50, 41.79])
    # M2 extension data full set including synchrotron + histology
    #M2_days = np.array([88., 92., 97., 100., 101., 101., 105., 124., 127., 140., 140., 159., 167., 174., 179., 202., 203., 222., 223., 251., 265., 285., 510., 510., 510., 510.])
    #M2_data_extension = np.array([2.22, 4.43, 6.23, 3.14, 7.16, 3.19, 6.74, 6.23, 9.61, 11.8, 12.3, 18.4, 16.9, 17.9, 18.9, 21, 24.59, 20.6, 26.23, 20.1, 30.60, 32.50, 41.79, 39.61, 39.39, 42.3])
    # M2 extension data partial set including synchrotron but not histology or outliers
    M2_days = np.array([88., 92., 97., 100., 101., 101., 105., 124., 127., 140., 140., 159., 167., 174., 179., 202., 222., 510., 510., 510., 510.]) # 251.,
    M2_data_extension = np.array([2.22, 4.43, 6.23, 3.14, 7.16, 3.19, 6.74, 6.23, 9.61, 11.8, 12.3, 18.4, 16.9, 17.9, 18.9, 21., 20.6, 41.79, 39.61, 39.39, 42.3]) # 20.1,
    # M1 data, extension
    #M1_days = np.array([1., 9., 11., 19., 21., 30., 31., 31., 38., 42., 54., 56., 56., 58., 61., 66., 68., 73., 78., 84., 88., 92., 97., 100., 101., 101., 104., 105., 124., 127., 140., 140., 157., 167., 173., 174., 179., 202., 222., 235., 238., 251., 259., 274.])
    #M1_data_extension = np.array([9.38, 8.05, 11.32, 9.43, 13.34, 16.19, 13.85, 15.96, 15.32, 14.21, 17.99, 19.32, 19.32, 18.31, 17.53, 18.68, 18.49, 22.08, 23.14, 19.92, 27.97, 24.38, 25.53, 29.07, 27.65, 26.27, 27.55, 24.33, 29.03, 29.07, 30.36, 31.79, 31.37, 31.28, 35.79, 29.81, 31.79, 34.04, 33.21, 34.50, 33.76, 33.40, 36.34, 33.63])

    optimize_curve(M2_days, M2_data_extension)

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