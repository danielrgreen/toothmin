#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  small_blood_to_toothmap.py
#  
#  Copyright 2014 Daniel Green
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

import csv
import numpy as np
import matplotlib.pyplot as plt

start = 0.

days_initial_period = 20.
initial_water_d18O = -5.5
initial_feed_d18O = -18.

air_d18O = 23.5
blood_halflife = 3. 

length_of_first_switch = 30.
first_H2O_switch_d18O = -8.
first_feed_switch_d18O = -18.

length_of_second_switch = 15.
second_H20_switch_d18O = -7
second_feed_switch_d18O = -18.

length_of_third_switch = 150.
third_H2O_switch_d18O = -5.5
third_feed_switch_d18O = -18.

length_of_fourth_switch = 60.
fourth_H2O_switch_d18O = -11.
fourth_feed_switch_d18O = -18.

length_of_fifth_switch = 20.
fifth_H2O_switch_d18O = -5.
fifth_feed_switch_d18O = -18.

likely_variance_blood = .25
measurement_error = .1
samples_per_point = 10
sample_frequency = 2.
    

def calc_d1(start, finish, air, h, water1, feed1, water2, feed2):
    '''
    calculates evolving isotope ratios (d18O per mil) in blood given
    information about air, blood, water and feed isotope ratios, and
    blood turnover time, from initial conditions.

    inputs:     start (day of experiment start, float)
                finish (day of experiment end, float)
                air (d18O of air, per mil)
                h (blood half life, days)
                water1 (d18O of initial water, per mil)
                feed1 (d18O of initial feed, per mil)
                water2 (d18O of switch water, per mil)
                feed2 (d18O of switch feed, per mil)

    outputs:    dvalue (vector of d18O over time, per mil in days)
    '''

    # Initial and equilibrium blood d18O
    bloodA = (.62*water1+.24*feed1+.14*air)/(.62+(.24*.985)+(.14*1.038)) # starting blood d18O
    bloodB = (.62*water2+.24*feed2+.14*air)/(.62+(.24*.985)+(.14*1.038)) # blood equilibrium d18O

    # create time series (days) and vector for d18O
    t = np.linspace(start, finish, num=(finish-start+1)) # time t in days
    dvalue = np.empty(finish-start+1) # empty vector for isotope ratio

    # calculate changing d18O over time
    for d in dvalue:
        dvalue = ((bloodA - bloodB)*(np.exp( (-(np.log(2))/h)*t))) + bloodB

    finish = t[-1]
    bloodA = dvalue[-1]

    return (dvalue, finish, bloodA, water2)


def calc_d2(start, finish, air, h, bloodA, water2, feed2):
    '''
    calculates evolving isotope ratios (d18O per mil) in blood given
    information about air, blood, water and feed isotope ratios, and
    blood turnover time, after initial conditions
        
    inputs:     start (day of experiment start, float)
                finish (day of experiment end, float)
                air (d18O of air, per mil)
                h (blood half life, days)
                water2 (d18O of switch water, per mil)
                feed2 (d18O of switch feed, per mil)
        
    outputs:    dvalue (vector of d18O over time, per mil in days)
    '''
    
    # Initial and equilibrium blood d18O
    bloodB = (.62*water2+.24*feed2+.14*air)/(.62+(.24*.985)+(.14*1.038)) # blood equilibrium d18O
    
    # create time series (days) and vector for d18O
    t = np.linspace(start, finish, num=(finish-start+1)) # time t in days
    dvalue = np.empty(finish-start+1) # empty vector for isotope ratio
    
    # calculate changing d18O over time
    for d in dvalue:
        dvalue = ((bloodA - bloodB)*(np.exp( (-(np.log(2))/h)*t))) + bloodB
    
    finish = t[-1]
    bloodA = dvalue[-1]
    
    return (dvalue, finish, bloodA, water2)

def calc_blood_hist():
    '''
    Takes initial inputs, calc_d1 and calc_d2 functions, and calculates
    overall blood, water, air and feed isotope histories.
    '''


    # phase 0: start, finish, air, h, water1, feed1, water1, feed1
    dvalue, finish, bloodA, water1 = calc_d1(0., days_initial_period, air_d18O,
                                             blood_halflife, initial_water_d18O,
                                             initial_feed_d18O, initial_water_d18O,
                                             first_feed_switch_d18O)

    dvalue0 = dvalue
    finish0 = finish
    blood0 = bloodA
    water0 = np.empty(finish0+1); water0.fill(water1)



    # phase 1: start, finish, air, h, water1, feed1, water2, feed2
    dvalue, finish, bloodA, water2 = calc_d1(0., length_of_first_switch, air_d18O,
                                             blood_halflife, initial_water_d18O,
                                             initial_feed_d18O, first_H2O_switch_d18O,
                                             first_feed_switch_d18O)

    dvalue1 = dvalue
    finish1 = finish
    blood1 = bloodA
    waterA = np.empty(finish1+1); waterA.fill(water2)
    
    # phase 2: start, finish, air, h, bloodA, water3, feed3
    dvalue, finish, bloodA, water2 = calc_d2(0., length_of_second_switch, air_d18O,
                                             blood_halflife, blood1, second_H20_switch_d18O,
                                             second_feed_switch_d18O)

    dvalue2 = dvalue
    finish2 = finish
    blood2 = bloodA
    waterB = np.empty(finish2+1); waterB.fill(water2)

    # phase 3: start, finish, air, h, bloodA, water4, feed4
    dvalue, finish, bloodA, water2 = calc_d2(0., length_of_third_switch, air_d18O,
                                             blood_halflife, blood2, third_H2O_switch_d18O,
                                             third_feed_switch_d18O)

    dvalue3 = dvalue
    finish3 = finish
    blood3 = bloodA
    waterC = np.empty(finish3+1); waterC.fill(water2)

    # phase 4: start, finish, air, h, bloodA, water5, feed5
    dvalue, finish, bloodA, water2 = calc_d2(0., length_of_fourth_switch, air_d18O,
                                             blood_halflife, blood3, fourth_H2O_switch_d18O,
                                             fourth_feed_switch_d18O)

    dvalue4 = dvalue
    finish4 = finish
    blood4 = bloodA
    waterD = np.empty(finish4+1); waterD.fill(water2)

    # phase 5: start, finish, air, h, bloodA, water6, feed6
    dvalue, finish, bloodA, water2 = calc_d2(0., length_of_fifth_switch, air_d18O,
                                             blood_halflife, blood4, fifth_H2O_switch_d18O,
                                             fifth_feed_switch_d18O)

    dvalue5 = dvalue
    finish5 = finish
    blood5 = bloodA
    waterE = np.empty(finish5+1); waterE.fill(water2)

    # append blood d18O from all phases together into one history

    a0 = np.append(dvalue0, dvalue1)
    ab = np.append(a0, dvalue2)
    bc = np.append(ab, dvalue3)
    cd = np.append(bc, dvalue4)
    d18O_history = np.append(cd, dvalue5)
   
    # append drinking water d18O from all phases into one history
   
    a0 = np.append(water0, waterA)
    ab = np.append(a0, waterB)
    bc = np.append(ab, waterC)
    cd = np.append(bc, waterD)
    water_history = np.append(cd, waterE)

    # create history for feed, air
    feed_history = np.empty(finish0+finish1+finish2+finish3+finish4+finish5+5); feed_history.fill(initial_feed_d18O)
    air_history = np.empty(finish0+finish1+finish2+finish3+finish4+finish5+5); air_history.fill(air_d18O)

    return (d18O_history, water_history, feed_history, air_history)

def noise(d18O_history, likely_variance_blood, measurement_error, samples_per_point, sample_frequency):
    '''
    Generates likely sample points at a given frequency around an estimated
    blood profile, using likely variance and measurement error
    '''

    sigma = ((likely_variance_blood**2)+(measurement_error**2))**.5
    samples = samples_per_point
    profile = d18O_history
    sigma_profile = []

    for i in profile:                    
        d = np.random.normal(i, sigma, samples)
        sigma_profile = np.append(sigma_profile, d)

    sigma_profile = np.reshape(sigma_profile, (profile.size, samples))
    
    return (sigma_profile, sigma)

def sheeps():

    data_947 = 0.


def main():

    d18O_history, water_history, feed_history, air_history = calc_blood_hist()

    # plot blood d18O over time


    sigma_profile, sigma = noise(d18O_history, likely_variance_blood, measurement_error, samples_per_point, sample_frequency)
    apts = sigma_profile[:,0]
    apts[1::sample_frequency] = np.nan
    bpts = sigma_profile[:,1]
    bpts[1::sample_frequency] = np.nan
    cpts = sigma_profile[:,2]
    cpts[1::sample_frequency] = np.nan
    dpts = sigma_profile[:,3]
    dpts[1::sample_frequency] = np.nan
    epts = sigma_profile[:,4]
    epts[1::sample_frequency] = np.nan
    fpts = sigma_profile[:,5]
    fpts[1::sample_frequency] = np.nan
    gpts = sigma_profile[:,6]
    gpts[1::sample_frequency] = np.nan
    hpts = sigma_profile[:,7]
    hpts[1::sample_frequency] = np.nan
    ipts = sigma_profile[:,8]
    ipts[1::sample_frequency] = np.nan
    jpts = sigma_profile[:,9]
    jpts[1::sample_frequency] = np.nan
    
    print 'sigma =', sigma
    
    max1, max2, max3, max4 = np.amax(d18O_history), np.amax(water_history), np.amax(feed_history), np.amax(air_history)
    maximum = np.amax(np.array([max1, max2, max3, max4]))

    min1, min2, min3, min4 = np.amin(d18O_history), np.amin(water_history), np.amin(feed_history), np.amin(air_history)
    minimum = np.amin(np.array([min1, min2, min3, min4]))
    
    fig = plt.figure(figsize=(9,8), edgecolor='none')
    ax = fig.add_subplot(1,1,1)
    ax.plot(d18O_history, c='r', alpha=1, label='blood')
    ax.plot(water_history, c='b', alpha=1, label='water')
    ax.plot(feed_history, c='g', alpha=1, label='feed')
    ax.plot(air_history, c='y', alpha=1, label='air')
    ax.plot(apts, c='r', marker='.', linestyle='none', label='sampled')
    ax.plot(bpts, c='r', marker='.', linestyle='none')
    #ax.plot(cpts, c='r', marker='.', linestyle='none')
    #ax.plot(dpts, c='r', marker='.', linestyle='none')
    #ax.plot(epts, c='r', marker='.', linestyle='none')
    #ax.plot(fpts, c='r', marker='.', linestyle='none')
    #ax.plot(gpts, c='r', marker='.', linestyle='none')
    #ax.plot(hpts, c='r', marker='.', linestyle='none')
    #ax.plot(ipts, c='r', marker='.', linestyle='none')
    #ax.plot(jpts, c='r', marker='.', linestyle='none')
    ax.set_title(b'Sheep blood $\delta^{18}$O varies with inputs & predicts tooth $\delta^{18}$O profiles')
    ax.set_ylim(-9, -3)              #(minimum-2, maximum*1.1)
    ax.set_xlim(3, 220)                       #(1., d18O_history.size)
    ax.legend(loc='best')
    ax.set_ylabel(r'$d^{18} \mathrm{O} \ \mathrm{in} \ \mathrm{VSMOW}$')
    ax.set_xlabel('time in days')
    fig.savefig('figtitle3.png', dpi=500)
    plt.show()


    return 0

if __name__ == '__main__':
    main()







