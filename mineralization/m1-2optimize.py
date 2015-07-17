__author__ = 'darouet'

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import nlopt
from time import time

def est_tooth_extension(ext_param, **kwargs): # days, amplitude, slope, offset
    '''
    '''
    model_extension, data_extension, m1days, zero_pt = extension(*ext_param, **kwargs)

    return compare(model_extension, data_extension, m1days, zero_pt), model_extension

def extension(amplitude, slope, offset, days, data_extension, m2days):
    height_max = 42. # in millimeters
    #offset = 84.
    model_extension = (amplitude * spec.erf(slope * (days - offset))) + (height_max - amplitude)
    zero_pt = (spec.erfinv((amplitude - height_max) / amplitude) + offset * slope) / slope
    #print 'zero_pt = ', zero_pt
    m1days = convert(m2days, amplitude, slope, offset)

    return model_extension, data_extension, m1days, zero_pt

def compare(model_extension, data_extension, m1days, zero_pt):
    sigma = 6. # in mm
    score = (model_extension - data_extension) / sigma
    score[~np.isfinite(score)] = 0.
    score = np.sum((score**2)**.5)
    #print 'score = ', score
    total_score = prior(m1days, zero_pt)
    return score+total_score

def prior(m1days, zero_pt):
    score1 = ((m1days[0] + 50)**2)**.5 / 5.
    if np.isfinite(score1) != True:
        score1 = 1000.
    score2 = ((m1days[1] - 30)**2)**.5 / 20.
    if np.isfinite(score2) != True:
        score2 = 1000.
    #score4 = (((m1days[3] - m1days[2]) - 50)**2)**.5 / 20.
    #if np.isfinite(score4) != True:
    #    score3 = 1000.
    #print 'score4 = ', score4
    score5 = ((zero_pt - 84)**2)**.5 / 5
    total_score = score1+score2+score5 #+score3 #+score4
    #print 'm1days = ', m1days[0], m1days[1], m1days[2]
    #print 'total score = ', total_score

    return total_score

def convert(m2days, amplitude, slope, offset):
    '''
    '''
    #m2height = 75.123*spec.erf(.0028302*(m2days+70.17))+(42-75.12266) # max at 42
    #m2height = 44.182*spec.erf(.003736412*(m2days+53.0767))+(40.5-44.182) # max at 40.5 optimized with full data set on nlopt
    #m2height = 46.625*spec.erf(.0032506*(m2days+53.0767))+(42.46-46.625) # max at 40.5 optimized with synchrotron data set on nlopt
    m2_height = amplitude*spec.erf(slope*(m2days+offset))+(41-amplitude) # max at 40.5 optimized with synchrotron data set on nlopt
    m2_percent = m2_height / 41
    m1_max_height = 35.
    m1_height = m2_percent * m1_max_height
    m1_amplitude = 36.428
    m1_offset = -57.637
    m1_slope = .004846
    m1days = (spec.erf((m1_amplitude+m1_height-m1_max_height)/m1_amplitude) + (m1_offset*m1_slope)) / m1_slope

    return m1days

def optimize_curve(tooth_days, tooth_extension, **fit_kwargs):
    data_extension = tooth_extension
    days = tooth_days
    m2days = np.array([84., 202., 263., 450., 500.])
    fit_kwargs['m2days'] = m2days
    fit_kwargs['data_extension'] = data_extension
    fit_kwargs['days'] = days

    t1 = time()

    f_objective = lambda x, grad: est_tooth_extension(x, **fit_kwargs)[0]

    local_opt = nlopt.opt(nlopt.LN_COBYLA, 3)
    local_opt.set_xtol_abs(.01)
    local_opt.set_lower_bounds([25., .001, -40.])
    local_opt.set_upper_bounds([160., .01, 40.])
    local_opt.set_min_objective(f_objective)

    global_opt = nlopt.opt(nlopt.G_MLSL_LDS, 3)
    global_opt.set_maxeval(200000)
    global_opt.set_lower_bounds([25., .001, -40.])
    global_opt.set_upper_bounds([160., .01, 40.])
    global_opt.set_min_objective(f_objective)
    global_opt.set_local_optimizer(local_opt)
    global_opt.set_population(3)

    print 'Running global optimizer ...'
    x_opt = global_opt.optimize([40., .005, -10.])

    minf = global_opt.last_optimum_value()
    print "optimum at", x_opt
    print "minimum value = ", minf
    print "result code = ", global_opt.last_optimize_result()

    t2 = time()
    run_time = t2-t1

    m1days = convert(m2days, *x_opt)
    for i in xrange(m2days.size):
        print m2days[i], m1days[i]
    switch_length =  m1days[2]-m1days[1]
    print 'switch length = ', switch_length

    days = np.linspace(50, 550, 501)
    extension_erf, data_extension, m1days, zero_pt = extension(x_opt[0], x_opt[1], x_opt[2], days, data_extension, m2days)
    diff_extension_erf = np.diff(extension_erf) * 1000
    days = np.linspace(50, 550, 501)
    second_score = prior(m1days, zero_pt)
    print 'percent prior score = ', (minf/second_score)*100

    local_method = 'cobyla'
    global_method = 'msds'
    textstr = '%.3f, %.6f, %.3f, \nmin = %.2f, time = %.1f seconds \nm1start = %.2f, m2start = %.2f \nM1_est_switch_len = %.2f \n%s, %s' % (x_opt[0], x_opt[1], x_opt[2], minf, run_time, m1days[0], zero_pt, switch_length, local_method, global_method)
    print textstr

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ax2.plot(days[1::4], diff_extension_erf[::4], 'b.', label=r'$ \mathrm{extension} \ \Delta $', alpha=.5)
    ax2.set_ylim([0,200])
    ax2.set_xlim([50,550])
    ax.plot(tooth_days, tooth_extension, marker='o', linestyle='none', color='b', label=r'$ \mathrm{extension} \ \mathrm{(observed)} $')
    ax.plot(days, extension_erf, linestyle='-', color='b', label=r'$ \mathrm{extension,} \ \mathrm{optimized} $')
    ax.set_ylim([0,45])
    ax.set_xlim([50,550])
    plt.title('Enamel secretion and maturation progress over time')
    ax.set_xlabel('Days after birth')
    ax.set_ylabel('Progress from cusp tip in mm')
    ax2.set_ylabel('Secretion or maturation speed in um/day')
    #ax2.legend(loc='lower right', fancybox=True, framealpha=0.8)
    #ax.legend(loc='upper right', fancybox=True, framealpha=0.8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='center right')
    ax.text(200, 10, textstr, fontsize=8)

    plt.show()



def main():

    # M2 extension data full set including synchrotron + histology
    #tooth_days = np.array([88., 92., 97., 100., 101., 101., 105., 124., 127., 140., 140., 159., 167., 174., 179., 202., 203., 222., 223., 251., 265., 285., 510., 510., 510., 510.])
    #tooth_extension = np.array([2.22, 4.43, 6.23, 3.14, 7.16, 3.19, 6.74, 6.23, 9.61, 11.8, 12.3, 18.4, 16.9, 17.9, 18.9, 21, 24.59, 20.6, 26.23, 20.1, 30.60, 32.50, 41.79, 39.61, 39.39, 42.3])
    # M2 extension data partial set including synchrotron but not histology
    tooth_days = np.array([88., 92., 97., 100., 101., 101., 105., 124., 127., 140., 140., 159., 167., 174., 179., 202., 222., 251., 510., 510., 510., 510.])
    tooth_extension = np.array([2.22, 4.43, 6.23, 3.14, 7.16, 3.19, 6.74, 6.23, 9.61, 11.8, 12.3, 18.4, 16.9, 17.9, 18.9, 21, 20.6, 20.1, 41.79, 39.61, 39.39, 42.3])

    #evaluation_pts = np.array([84., 202., 263.])

    optimize_curve(tooth_days, tooth_extension)

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