# Daniel Green, Gregory Green, 2014
# drgreen@fas.harvard.edu
# Human Evolutionary Biology
# Center for Astrophysics
# Harvard University
#
# Mineralization Model Re-Size:
# this code takes a larger mineralization model
# and produces images demonstrating mineral density
# increase over time, total density over time, or
# calculates final isotope distributions at full
# or partial resolution.
# 

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.special as spec

from time import time

def tooth_timing_convert_curv2lin(conversion_times, a1, s1, o1, max1, s2, o2, max2):

    t1_ext = a1*spec.erf(s1*(conversion_times-o1))+(max1-a1)
    t1_pct = t1_ext / max1
    t2_ext = t1_pct * max2
    converted_times = (t2_ext-o2)/s2

    return converted_times

def tooth_timing_convert_lin2curv(conversion_times, s1, o1, max1, a2, s2, o2, max2):

    t1_ext = (s1*conversion_times)+o1
    t1_pct = t1_ext / max1
    t2_ext = t1_pct * max2
    converted_times = (spec.erfinv((a2+t2_ext-max2)/a2) + (o2*s2)) / s2

    return converted_times

def tooth_timing_convert(conversion_times, a1, s1, o1, max1, a2, s2, o2, max2):
    '''
    Takes an array of events in days occurring in one tooth, calculates where
    these will appear spatially during tooth extension, then maps these events
    onto the spatial dimensions of a second tooth, and calculates when similar
    events would have occurred in days to produce this mapping in the second
    tooth.

    Inputs:
    conversion_times:   a 1-dimensional numpy array with days to be converted.
    a1, s1, o1, max1:   the amplitude, slope, offset and max height of the error
                        function describing the first tooth's extension, in mm,
                        over time in days.
    a2, s2, o2, max2:   the amplitude, slope, offset and max height of the error
                        function describing the second tooth's extension, in mm,
                        over time in days.
    Returns:            converted 1-dimensional numpy array of converted days.

    '''
    t1_ext = a1*spec.erf(s1*(conversion_times-o1))+(max1-a1)
    t1_pct = t1_ext / max1
    t2_ext = t1_pct * max2
    converted_times = (spec.erfinv((a2+t2_ext-max2)/a2) + (o2*s2)) / s2

    return converted_times

def spline_input_signal(iso_values, value_days, smoothness):
    '''
    Takes a series of iso_values, each lasting for a number of days called value_days,
    and interpolates to create a water history of the appropriate length iso_values*value_days.
    Has blood and water data from sheep 962 arranged from birth and outputs a
    day-by-day spline-smoothed version.
    '''

    spline_data_days = np.arange(np.size(iso_values))*value_days
    spline_output = InterpolatedUnivariateSpline(spline_data_days, iso_values, k=smoothness)
    days = np.arange(value_days*np.size(iso_values))
    water_spl = spline_output(days)

    return water_spl[:584]

def main():

    m1_m2_params = np.array([21.820, .007889, 29.118, 35., 67.974, 0.003352, -25.414, 41.]) # 'synch86', outlier, 100k
    m2_m1_params = np.array([67.974, 0.003352, -25.414, 41., 21.820, .007889, 29.118, 35.]) # 'synch86', outlier, 100k
    m2_m2_params_curv2lin = np.array([67.974, 0.003352, -25.414, 41., (41./416.), -8.3, 41.]) # 'synch86', outlier, 100k

    daily_d18O_360 = 10.*np.sin((2*np.pi/360.)*(np.arange(600.)))-11.
    daily_d18O_180 = 10.*np.sin((2*np.pi/180.)*(np.arange(600.)))-11.
    daily_d18O_090 = 10.*np.sin((2*np.pi/90.)*(np.arange(600.)))-11.
    daily_d18O_045 = 10.*np.sin((2*np.pi/45.)*(np.arange(600.)))-11.

    days = np.arange(84., 684.)
    m2_events = np.arange(0., 600., 50.)+84
    converted_days = tooth_timing_convert(days, *m2_m1_params)
    converted_days = converted_days-converted_days[0]
    m1_events = tooth_timing_convert(m2_events, *m2_m1_params)

    M2_test1 = np.ones(days.size)
    M2_test1[:] = 5.
    M2_test1[50:100] = 15.
    M2_test1[150:200] = 25.
    M2_test1[250:300] = 35.
    M2_test1[350:400] = 45.
    M2_test1[450:500] = 55.

    M1_test1_tmp = np.ones(converted_days.size)
    for k,d in enumerate(converted_days):
        print k,d
        d = int(d)
        M1_test1_tmp[d:] = M2_test1[k]
    M1_test1 = M1_test1_tmp
    #M1_test1 = M1_test1[84:]

    print 'days =', days
    print 'converted days =', converted_days
    print 'm2 = ', M2_test1
    print 'm1 = ', M1_test1

    t_save = time()

    print days.size, M1_test1.size, M2_test1.size, days[:-84].size

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1text = 'M2->M1, M2_days start@84, M2/M1 plotted w/diff day_arrays'
    ax1.text(0, 50, ax1text, fontsize=8)
    ax1.plot(days, M2_test1, 'k--', linewidth=1.0)
    ax1.plot(converted_days, M1_test1, 'b-', linewidth=1.0)
    ax1.set_ylim(-5, 65)
    ax1.set_xlim(-50, 600)

    ax1 = fig.add_subplot(2,1,2)
    ax1text = 'M2->M1, M2_days start@84, M2/M1 plotted on same'
    ax1.text(0, 50, ax1text, fontsize=8)
    ax1.plot(np.arange(np.size(M2_test1)), M2_test1, 'k--', linewidth=1.0)
    ax1.plot(np.arange(np.size(M1_test1)), M1_test1, 'b-', linewidth=1.0)
    ax1.set_ylim(-5, 65)
    ax1.set_xlim(-50, 600)

    fig.savefig('M2-M1_convert_testing_{0}.svg'.format(t_save), dpi=300, bbox_inches='tight')
    plt.show()

    # Converting back
    m2_events_reverse = tooth_timing_convert(m1_events, *m1_m2_params)
    print 'm2 events'
    print m2_events
    print 'm1 events'
    print m1_events
    print 'm2 events reverse'
    print m2_events_reverse


    return 0

if __name__ == '__main__':
    main()




