__author__ = 'Daniel'


import numpy as np
import matplotlib.pyplot as plt


def blood_history(length, first_d18O, second_d18O, switch_day):

    blood_hist = np.ones(length)
    blood_hist[:switch_day] = first_d18O
    blood_hist[switch_day:] = second_d18O

    return blood_hist

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

def PO4_dissoln_reprecip(reprecip_eq_t_half, pause, pct_flux, d_blood):

    pause = int(pause)
    new_d_blood = d_blood[pause:]
    alpha = np.log(2.) / reprecip_eq_t_half
    beta = alpha * new_d_blood
    d_tooth_phosphate = np.empty(d_blood.size, dtype='f8')
    d_tooth_phosphate[:-pause] = integrate_delta(d_blood[0], alpha, beta)
    d_tooth_phosphate[-pause:] = d_blood[-pause:]

    phosphate_eq = (d_tooth_phosphate * pct_flux) + (d_blood * (1 - pct_flux))

    return phosphate_eq

def PO4_evolve(start, blood_hist, PO4lambda, tex):
    '''
    x_frac = 1_d array of fractional contribution of original d18O over time in days
    x_frac_contrib = 1_d array fractional contribution each day of resetting PO4
    initial_value = composition original d18O
    '''

    blood_to_PO4_hist = blood_hist[start:]
    initial_value = blood_to_PO4_hist[0]
    #print "initial_value =", initial_value
    x_frac = np.ones(len(blood_to_PO4_hist))

    for a,b in enumerate(x_frac[tex:]):
        x_frac[tex+a] = np.exp(-PO4lambda*a)

    x_frac_contrib = np.ones(len(blood_to_PO4_hist))
    for a,b in enumerate(x_frac[:-1]):
        x_frac_contrib[a] = x_frac[a] - x_frac[a+1]
    x_frac_contrib[-1] = 0.

    frac_HAp_added = np.sum(x_frac_contrib)
    frac_remaining = 1 - frac_HAp_added

    #print "iteration # ", start
    #print "x_frac =", x_frac
    #print "x_frac_contrib =", x_frac_contrib
    #print "frac_HAp_added =", frac_HAp_added
    #print "frac_remaining =", frac_remaining
    #print "blood_to_PO4_hist =", blood_to_PO4_hist

    added_d18O = blood_to_PO4_hist*(x_frac_contrib/frac_HAp_added)
    added_d18O[np.isnan(added_d18O)] = 0.

    #print "added_d18O =", added_d18O

    added_d18O = np.sum(added_d18O)

    #print "added_d18O =", added_d18O

    PO4_evolved = frac_HAp_added*added_d18O + (1-frac_HAp_added)*initial_value

    #print "PO4_evolved =", PO4_evolved

    return PO4_evolved

def new_PO4_equations(half_life, tex, frac, blood_hist):

    PO4lambda = np.log(2.) / half_life
    tex = int(np.rint(tex))

    new_PO4_eq = np.ones(len(blood_hist))
    for a,b in enumerate(blood_hist):
        new_PO4_eq[a] = PO4_evolve(a, blood_hist, PO4lambda, tex)
        #print new_PO4_eq[a]
        new_PO4_eq[a] = frac*new_PO4_eq[a]+(1-frac)*blood_hist[a]
        #print new_PO4_eq[a]

    #print new_PO4_eq

    return new_PO4_eq

def main():

    blood_hist = blood_history(100, -5., -10., 50)
    phosphate_eq = PO4_dissoln_reprecip(3.0, 20, .3, blood_hist)
    new_PO4_eq = new_PO4_equations(3.0, 20, .3, blood_hist)
    days = np.arange(len(blood_hist))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(days, blood_hist, 'ro')
    ax.plot(days, new_PO4_eq, 'g*')
    ax.plot(days, phosphate_eq, 'k--')
    ax.plot(days, (new_PO4_eq+phosphate_eq)/2., 'b-')

    ax.set_ylim(-11., -4.)
    ax.set_xlim(0., 100.)
    plt.show()

if __name__ == '__main__':
    main()