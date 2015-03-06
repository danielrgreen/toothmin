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
    w0 = kwargs.get('w0', -6.5)
    w1 = kwargs.get('w1', -6.5)
    w2 = kwargs.get('w2', -6.5)
    w3 = kwargs.get('w3', -13.5)
    w4 = kwargs.get('w4', -6.5)
    w5 = kwargs.get('w5', -6.5)
    w6 = kwargs.get('w6', -6.5)
    w7 = kwargs.get('w7', -6.5)
    w8 = kwargs.get('w8', -6.5)
    w9 = kwargs.get('w9', -6.5)
    w10 = kwargs.get('w10', -6.5)
    w11 = kwargs.get('w11', -6.5)
    w12 = kwargs.get('w12', -6.5)
    w13 = kwargs.get('w13', -6.5)
    w14 = kwargs.get('w14', -6.5)
    w15 = kwargs.get('w15', -6.5)
    w16 = kwargs.get('w16', -6.5)
    w17 = kwargs.get('w17', -6.5)
    w18 = kwargs.get('w18', -6.5)
    w19 = kwargs.get('w19', -6.5)
    w20 = kwargs.get('w20', -6.5)
    water_step = np.ones(n_days)
    water_step[0:20] = w0
    water_step[20:40] = w1
    water_step[40:60] = w2
    water_step[60:80] = w3
    water_step[80:100] = w4
    water_step[100:120] = w5
    water_step[120:140] = w6
    water_step[140:160] = w7
    water_step[160:180] = w8
    water_step[180:200] = w9
    water_step[200:220] = w10
    water_step[220:240] = w11
    water_step[240:260] = w12
    water_step[260:280] = w13
    water_step[280:300] = w14
    water_step[300:320] = w15
    water_step[320:340] = w16
    water_step[340:360] = w17
    water_step[360:380] = w18
    water_step[380:400] = w19
    water_step[400:n_days] = w20

    return water_step

def gaussian_sum(n_days, mu, sigma, A):
    x = np.arange(n_days)
    y = np.zeros(n_days, dtype='f8')

    for m,s,a in zip(mu, sigma, A):
        y += a * np.exp(-0.5*((x-m)/s)**2.)

    return y


def calc_water_gaussian(n_days, **kwargs):
    '''
    Creates an isotope history defined by a mean value and three
    gaussian functions distributed throughout the history.
    :param kwargs:  enter mean water isotope values, amplitude (A),
                    sigma (s), and offset (x) for each gaussian
                    rainfall event.
    :return:        String with isotope ratios in the shape of
                    gaussian functions.
    '''
    mean = kwargs.get('mean', 1.)
    A0 = kwargs.get('A0', -6.)
    s0 = kwargs.get('s0', 25.)
    x0 = kwargs.get('x0', 50.)
    A1 = kwargs.get('A1', -3.)
    s1 = kwargs.get('s1', 16.)
    x1 = kwargs.get('x1', 90.)
    A2 = kwargs.get('A2', -4.)
    s2 = kwargs.get('s2', 18.)
    x2 = kwargs.get('x2', 240.)

    days = np.arange(n_days)
    water_gaussian = (
                        A0*np.exp(-.5*((days-x0)/(s0))**2)
                        + A1*np.exp(-.5*((days-x1)/(s1))**2)
                        + A2*np.exp(-.5*((days-x2)/(s2))**2)
                        )

    water_gaussian += mean

    return water_gaussian

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

def integrate_delta(delta_0, alpha, beta):
    '''
    Calculate delta on every day, given an initial value, a decay rate, and
    a variable amount added on each day.

    :param delta_0: The initial delta (a constant)
    :param alpha:   The fraction that leaves the system each day (a constant)
    :param beta:    The amount added to the system on each day (an array)
    :return:        delta on each day. Has the same length as beta.
    '''

    n_days = beta.size

    decay_factor = np.exp(-alpha * np.linspace(0.5, n_days-0.5, n_days))
    delta = np.zeros(n_days, dtype='f8')

    for k,b in enumerate(beta):
        delta[k:] += decay_factor[:n_days-k] * b

    d_0 = (delta_0 - delta[0]) / decay_factor[0]
    delta += decay_factor * d_0

    return delta

def blood_d_equilibrium(d_O2, d_water, d_feed, **kwargs):
    '''

    :param d_O2:
    :param d_water:
    :param d_feed:
    :param kwargs:
    :return:
    '''
    # Get defaults
    f_H2O = kwargs.get('f_H20', 0.62)
    f_O2 = kwargs.get('f_O2', 0.24)
    alpha_O2 = kwargs.get('alpha_O2', 0.992)
    f_feed = kwargs.get('f_feed', 0.14)

    f_H2O_en = kwargs.get('f_H2O_en', 0.62)  # Fraction effluent H2O unfractionated
    alpha_H2O_ef = kwargs.get('alpha_H2O_ef', 0.992)
    f_H2O_ef = kwargs.get('f_H2O_ef', 0.14)
    alpha_CO2_H2O = kwargs.get('alpha_CO2_H2O', 1.038)
    f_CO2 = kwargs.get('f_CO2', 0.24)

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

    return R2d(R_eq)

def blood_delta(d_O2, d_water, d_feed, **kwargs):
    # Calculate equilibrium on each day
    d_eq = blood_d_equilibrium(d_O2, d_water, d_feed, **kwargs)

    # Integrate differential equation to get
    t_half = kwargs.get('t_half', 3.)
    alpha = np.log(2.) / t_half
    beta = alpha * d_eq

    return integrate_delta(d_eq[0], alpha, beta)


def test_integration():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    n_days = 1000
    alpha = 0.05
    beta = np.ones(n_days, dtype='f8')
    beta[:n_days/5] = -1.
    beta[n_days/5:2*n_days/5] = 3.
    beta[2*n_days/5:4*n_days/5] = 0.
    beta[4*n_days/5:] = -5.
    delta_0 = -2.
    delta = integrate_delta(delta_0, alpha, beta)

    days = np.arange(n_days)
    ax.plot(days, delta)
    ax.plot(days, beta/alpha)

    ax.set_ylim(1.2*np.min(beta/alpha), 1.2*np.max(beta/alpha))
    plt.show()

def calc_blood_step(**kwargs):
    '''

    :param water:       Takes an isotope water history 1D vector
                        from calc_water_step
    :param kwargs:      Key word arguments include feed and air isotope
                        delta values
    :return:            Stepped blood delta values in SMOW
    '''

    feed = kwargs.get('feed', 24.)
    air = kwargs.get('air', 23.5)

    water_step = calc_water_step(400)

    n_days = water_step.size
    days = np.arange(n_days)
    d_eq = blood_d_equilibrium(air, water_step, feed)
    delta = blood_delta(air, water_step, feed, t_half=3.)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(days, delta, 'r')
    ax.plot(days, water_step, 'b')
    ax.plot(days, d_eq, 'k')

    ax.set_ylim(-16., 6.)
    #vmin = max(np.max(delta), np.max(water))
    #ax.set_ylim(1.2*vmin, 1.2*vmax)
    #plt.show()

    return water_step, delta

def calc_blood_gaussian(**kwargs):
    '''

    :param water:       Episodic isotope delta history taken
                        from calc_water_gaussian
    :param kwargs:      Key word arguments include feed and air
                        isotope delta values
    :return:            Gaussian blood history in delta SMOW
    '''
    feed = kwargs.get('feed', 24.)
    air = kwargs.get('air', 23.5)

    water_gaussian = calc_water_gaussian(400)

    n_days = water_gaussian.size
    days = np.arange(n_days)
    d_eq = blood_d_equilibrium(air, water_gaussian, feed)
    delta = blood_delta(air, water_gaussian, feed, t_half=3.)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(days, delta, 'r')
    ax.plot(days, water_gaussian, 'b')
    ax.plot(days, d_eq, 'k')

    ax.set_ylim(-15., 5.)
    #vmin = max(np.max(delta), np.max(water))
    #ax.set_ylim(1.2*vmin, 1.2*vmax)
    #plt.show()

    return water_gaussian, delta

def test_blood_delta():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    n_days = 339
    feed = 24.
    air = 23.5
    water = np.ones(n_days, dtype='f8')
    water[:8] = -7.
    water[8:28] = -9.5
    water[28:53] = -8.
    water[53:183] = -7.
    water[183:189] = -14.5
    water[189:339] = -8.

    d_eq = blood_d_equilibrium(air, water, feed)
    delta = blood_delta(air, water, feed, t_half=3.)

    days = np.arange(n_days)
    ax.plot(days, delta, 'r')
    ax.plot(days, water, 'b')
    ax.plot(days, d_eq, 'k')

    ax.set_xlim(0., 300.)
    ax.set_ylim(-15., -4.)
    #vmin = max(np.max(delta), np.max(water))
    #ax.set_ylim(1.2*vmin, 1.2*vmax)
    #plt.show()

def main():
    water_gaussian, gaussian_delta = calc_blood_gaussian()
    water_step, step_delta = calc_blood_step()

if __name__ == '__main__':
    main()