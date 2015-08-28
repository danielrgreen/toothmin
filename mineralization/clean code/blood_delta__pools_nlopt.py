#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  blood_delta.py
#  
#  Copyright 2015 Daniel Green
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#

import numpy as np
import matplotlib.pyplot as plt
import nlopt
from time import time


def calc_water_step2(w_blocks, block_length):
    '''
    Expand blocks of water isotope value into one array.

    :param w_blocks: Isotope value in each block.
    :param block_length: Length (in days) of each block.
    :return: The isotope value on each day.
    '''

    return np.repeat(np.array(w_blocks), block_length)

    '''
    w = np.empty(len(w_blocks) * block_length, dtype='f8')

    for k,val in enumerate(w_blocks):
        w[k*block_length:(k+1)*block_length] = val

    return w
    '''

def calc_water_step(n_days, **kwargs):
    '''

    :param n_days:      Number of days to calculate water history
    :param kwargs:      Isotope values every 20 days in water history
    :return:            Stepped water isotope history as a 1D float vector.
    '''
    w0 = kwargs.get('w0', -6.467)
    w1 = kwargs.get('w1', -6.467)
    w2 = kwargs.get('w2', -6.467)
    w3 = kwargs.get('w3', -6.467)
    w4 = kwargs.get('w4', -6.467)
    w5 = kwargs.get('w5', -6.467)
    w6 = kwargs.get('w6', -6.467)
    w7 = kwargs.get('w7', -6.467)
    w8 = kwargs.get('w8', -6.467)
    w9 = kwargs.get('w9', -6.467)
    w10 = kwargs.get('w10', -19.367)
    w11 = kwargs.get('w11', -19.367)
    w12 = kwargs.get('w12', -19.367)
    w13 = kwargs.get('w13', -6.467)
    w14 = kwargs.get('w14', -6.467)
    w15 = kwargs.get('w15', -6.467)
    w16 = kwargs.get('w16', -6.467)
    w17 = kwargs.get('w17', -6.467)
    w18 = kwargs.get('w18', -6.467)
    w19 = kwargs.get('w19', -6.467)
    w20 = kwargs.get('w20', -6.467)
    water_step = np.ones(n_days)
    water_step[0:20] = w0
    water_step[20:40] = w1
    water_step[40:60] = w2
    water_step[60:80] = w3
    water_step[80:100] = w4
    water_step[100:105] = w5
    water_step[105:140] = w6
    water_step[140:160] = w7
    water_step[160:180] = w8
    water_step[180:199] = w9
    water_step[199:220] = w10
    water_step[220:240] = w11
    water_step[240:261] = w12
    water_step[261:280] = w13
    water_step[280:300] = w14
    water_step[300:320] = w15
    water_step[320:340] = w16
    water_step[340:360] = w17
    water_step[360:380] = w18
    water_step[380:400] = w19
    water_step[400:n_days] = w20

    return water_step

def d2R(delta, standard=0.0020052):
    '''
    Convert isotope delta to Ratio.

    :param delta: delta of isotope
    :param standard: Ratio in standard (default: d18O/d16O in SMOW)
    :return: Ratio of isotope
    '''
    return (delta/1000. + 1.) * standard

def R2d(Ratio, standard=0.0020052):
    '''
    Convert isotope Ratio to delta.

    :param Ratio: Ratio of isotope
    :param standard: Ratio in standard (default: d18O/d16O in SMOW)
    :return: delta of isotope
    '''
    return (Ratio/standard - 1.) * 1000.

def blood_d_equilibrium(blood_params, d_O2, d_water, d_feed, **kwargs):
    '''

    :param d_O2:
    :param d_water:
    :param d_feed:
    :param kwargs:
    :return:
    '''
     # Get defaults
    airfrac = (1.-blood_params[0]) * (blood_params[1])
    feed_frac = (1.-blood_params[0]) * (1. - blood_params[1])

    f_H2O = kwargs.get('f_H20', blood_params[0]) # was 0.62
    f_O2 = kwargs.get('f_O2', airfrac) # was 0.24
    alpha_O2 = kwargs.get('alpha_O2', blood_params[2]) # Was 0.992
    f_feed = kwargs.get('f_feed', feed_frac) # was 0.14

    f_H2O_en = kwargs.get('f_H2O_en', blood_params[0])  # Fraction effluent H2O unfractionated. Was 0.62
    alpha_H2O_ef = kwargs.get('alpha_H2O_ef', blood_params[2]) # Was 0.992
    f_H2O_ef = kwargs.get('f_H2O_ef', feed_frac) # was 0.14
    alpha_CO2_H2O = kwargs.get('alpha_CO2_H2O', blood_params[3]) # Was 1.0383
    f_CO2 = kwargs.get('f_CO2', airfrac) # was 0.24


    # Calculate equilibrium on each day
    R_eq = (
        d2R(d_water) * f_H2O
      + d2R(d_O2) * alpha_O2 * f_O2
      + d2R(d_feed) * f_feed
    )

    R_eq /= (
        f_H2O_en
        + (alpha_H2O_ef * f_H2O_ef)
        + (alpha_CO2_H2O * f_CO2)
    )

    return R2d(R_eq) #+ ev_enrichment

def cerling_delta(blood_params, **kwargs):

    d_feed = kwargs.get('feed', 25.5)
    d_O2 = kwargs.get('air', 23.5)

    l1 = blood_params[4]
    l2 = blood_params[5]

    f1 = blood_params[6]
    f2 = 1 - blood_params[6]

    d_water = calc_water_step(400., **kwargs)
    days = np.arange(d_water.size)
    d_eq = blood_d_equilibrium(blood_params, d_O2, d_water, d_feed, **kwargs)
    dt = np.ones(days.size) * d_eq[0]
    for d,i in enumerate(dt):
        dt[d] = (dt[d-1] - d_eq[d]) * ((f1*np.exp(-l1)) + (f2*np.exp(-l2))) + d_eq[d] # ****WRONG****

    #for j,k in enumerate(dt):
    #    print j,k

    blood_day_measures = np.array([(57., -5.71), (199., -4.96), (203., -10.34), (207., -12.21), (211., -13.14), (215., -13.49), (219., -13.16), (239., -13.46), (261., -13.29), (281., -4.87), (289., -4.97), (297., -4.60), (309., -4.94)])
    blood_days = np.array([i[0] for i in blood_day_measures])
    blood_measures = np.array([i[1] for i in blood_day_measures])

    return d_water, dt, blood_days, blood_measures

def compare(blood_days, blood_measures, delta):

    sigma = .5 # d18O
    blood_days_new = np.ones(blood_days.size, dtype='int')
    for n,b in enumerate(blood_days):
        blood_days_new[n] = int(b)
    delta_compare = delta[blood_days_new]

    #print "real = ", blood_measures
    #print "model = ", delta_compare

    param_score = (blood_measures[1:] - delta_compare[1:])**2. / sigma**2
    #M1_score[~np.isfinite(M1_score)] = 10000000.
    param_score = (1. / (2. * np.pi * sigma**2)) * np.exp(-.5*(param_score))
    param_score = np.product(param_score)

    return param_score * -1

def est_blood_params(blood_params, **kwargs):

    water_step, delta, blood_days, blood_measures = cerling_delta(blood_params, **kwargs)
    score = compare(blood_days, blood_measures, delta)

    return score, delta, blood_days, blood_measures, water_step

def optimize_blood_params(blood_days, blood_measures, **fit_kwargs):
    fit_kwargs['blood_measures'] = blood_measures
    fit_kwargs['blood_days'] = blood_days

    t1 = time()

    f_objective = lambda x, grad: est_blood_params(x, **fit_kwargs)[0]

    local_opt = nlopt.opt(nlopt.LN_COBYLA, 7)
    local_opt.set_xtol_abs(.01)
    local_opt.set_lower_bounds([.40, .52, .960, 1.0160, 2.8095, .001, .99])
    local_opt.set_upper_bounds([.85, .85, .999, 1.0520, 2.8095, 2., .99])
    local_opt.set_min_objective(f_objective)

    global_opt = nlopt.opt(nlopt.G_MLSL_LDS, 7)
    global_opt.set_maxeval(5000)
    global_opt.set_lower_bounds([.40, .52, .960, 1.0160, 2.8095, .001, .99])
    global_opt.set_upper_bounds([.85, .85, .999, 1.0520, 2.8095, 2., .99])
    global_opt.set_min_objective(f_objective)
    global_opt.set_local_optimizer(local_opt)
    global_opt.set_population(7)

    print 'Running global optimizer ...'
    x_opt = global_opt.optimize([.60, .66, .992, 1.0383, 2.8095, .01, .99])

    minf = global_opt.last_optimum_value()
    print "minimum value = ", minf
    print "result code = ", global_opt.last_optimize_result()

    print "water input fraction = ", x_opt[0]
    print "air input fraction = ", (1. - x_opt[0]) * (x_opt[1])
    print "air input alpha = ", x_opt[2]
    print "feed input fraction = ", (1. - x_opt[0]) * (1. - x_opt[1])
    print "water efflux alpha = ", x_opt[2]
    print "CO2 efflux alpha = ", x_opt[3]


    print "first lambda = ", x_opt[4]
    print "second lambda = ", x_opt[5]
    print "first lambda fraction = ", x_opt[6]
    print "second lambda fraction = ", 1 - x_opt[6]

    first_thalf = np.log(2)/x_opt[4]
    second_thalf = np.log(2)/x_opt[5]

    t2 = time()
    run_time = t2-t1

    score, delta, blood_days, blood_measures, water_step = est_blood_params(x_opt, **fit_kwargs)
    days = np.arange(delta.size)

    textstr = '%.3f, %.3f, %.3f, %.3f \n %.3f, %.3f, %.3f \n1st_halflife = %.3f, 2nd_halflife = %.3f \nmin = %.3g, time = %.1f seconds' % (x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], x_opt[6], first_thalf, second_thalf, minf, run_time)

    t_save = time()

    # M2 extension data histology only
    hist_days = np.array([203., 223., 265., 285., 510.]) # ************* Last is 510 ***********
    hist_extension = np.array([24.59, 26.23, 30.60, 32.50, 41.79])

    # M2 synchrotron outliers
    outlier_days = np.array([222., 251.])
    outlier_extension = np.array([20.6, 20.1])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(blood_days, blood_measures, 'r*', linewidth=1.0)
    ax.plot(days, delta, 'r-.', linewidth=1.0)
    ax.plot(days, water_step, 'b-', linewidth=.5)
    ax.text(50, -10, textstr, fontsize=8)
    fig.savefig('blood_delta_pools_opt_{0}.svg'.format(t_save))
    plt.show()

def main():

    # Blood and water isotope measurements from sheep 962
    blood_day_measures = np.array([(57., -5.71), (199., -4.96), (203., -10.34), (207., -12.21), (211., -13.14), (215., -13.49), (219., -13.16), (239., -13.46), (261., -13.29), (281., -4.87), (289., -4.97), (297., -4.60), (309., -4.94)])
    blood_days = np.array([i[0] for i in blood_day_measures])
    blood_measures = np.array([i[1] for i in blood_day_measures])

    optimize_blood_params(blood_days, blood_measures)

    #water_step, step_delta = calc_blood_step()

if __name__ == '__main__':
    main()