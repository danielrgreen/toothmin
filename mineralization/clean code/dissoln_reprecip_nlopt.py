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
import matplotlib.mlab as mlab
import h5py
import nlopt
from PIL import Image

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.misc import imresize
from blood_delta import calc_blood_step, calc_water_step2, calc_water_gaussian, calc_blood_gaussian, blood_delta, tooth_phosphate_reservoir
from blood_delta_experimentation import PO4_dissoln_reprecip
import scipy.special as spec

from blood_delta import calc_blood_gaussian
from scipy.optimize import curve_fit, minimize, leastsq

from time import time

def fit_tooth_data(data_fname, model_fname='equalsize_jul2015a.h5', **kwargs):
    '''
    '''

    textstr = 'x_opt= %.2f, %.2f, %.2f, %.2f\nPO4_t= %.2f, PO4_p= %.2f, PO4_f= %.2f\nm2_params= %.2f, %.2f, %.2f, %.2f \nmin= %.2f, time= %.1f \n trials= %.1f, trials/sec= %.2f \n%s, %s' % (x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], x_opt[6], x_opt[0], x_opt[1], converted_times[0], optimized_length, minf, run_time, trials, eval_p_sec, local_method, global_method)
    print textstr

    fig = plt.figure()
    ax1 = fig.add_subplot(7,1,1)

    ax1.plot(days, w_iso_hist, 'b-', linewidth=2.0)
    ax1.plot(days, blood_hist, 'r-', linewidth=2.0)
    ax1.plot(days, phosphate_eq, 'g-.', linewidth=1.0)
    ax1.plot(blood_days, blood_measures, 'r*', linewidth=1.0)
    ax1.plot(water_iso_days, water_iso_measures, 'b*', linewidth=1.0)
    for s in list_water_results[:-1]:
        ax1.plot(days, s, 'g-', alpha=0.05)
    #vmin = np.min(np.concatenate((real_switch_hist, w_iso_hist, blood_hist), axis=0)) - 1.
    #vmax = np.max(np.concatenate((real_switch_hist, w_iso_hist, blood_hist), axis=0)) + 1.
    ax1.text(350, -20, textstr, fontsize=8)
    ax1.set_ylim(-30, 0)
    ax1.set_xlim(-100, 550)

    temp, model_isomap = water_hist_prob_4param(x_opt, **fit_kwargs)
    opt_params = np.array([x_opt[0], x_opt[1], x_opt[2], x_opt[3], 3., 34.5, .3])
    temp_opt, model_isomap_opt = water_hist_prob_4param(opt_params, **fit_kwargs)

    ax2 = fig.add_subplot(7,1,2)
    #ax2text = 'Inverse model result'
    #ax2.text(21, 3, ax2text, fontsize=8)
    cimg2 = ax2.imshow(np.mean(model_isomap, axis=2).T, aspect='auto', interpolation='nearest', origin='lower', cmap='bwr', vmin=9., vmax=15.)
    cax2 = fig.colorbar(cimg2)

    ax3 = fig.add_subplot(7,1,3)
    #ax3text = '962 data'
    #ax3.text(21, 3, ax3text, fontsize=8)
    cimg3 = ax3.imshow(data_isomap.T, aspect='auto', interpolation='nearest', origin='lower', cmap='bwr', vmin=9., vmax=15.)
    cax3 = fig.colorbar(cimg3)

    residuals = np.mean(model_isomap, axis=2) - data_isomap
    ax4 = fig.add_subplot(7,1,4)
    #ax4text = 'model - data residuals'
    #ax4.text(21, 3, ax4text, fontsize=8)
    cimg4 = ax4.imshow(residuals.T, aspect='auto', interpolation='nearest', origin='lower', cmap='RdGy', vmin=-1.6, vmax=1.6) # Residuals
    cax4 = fig.colorbar(cimg4)

    ax5 = fig.add_subplot(7,1,5)
    #ax5text = 'forward model expectation'
    #ax5.text(21, 3, ax5text, fontsize=8)
    cimg5 = ax5.imshow(np.mean(trial_model, axis=2).T, aspect='auto', interpolation='nearest', origin='lower', cmap='bwr', vmin=9., vmax=15.)
    cax5 = fig.colorbar(cimg5)

    trial_residuals = np.mean(trial_model, axis=2) - data_isomap
    ax6 = fig.add_subplot(7,1,6)
    #ax5text = 'forward model expectation'
    #ax5.text(21, 3, ax5text, fontsize=8)
    cimg6 = ax6.imshow(trial_residuals.T, aspect='auto', interpolation='nearest', origin='lower', cmap='RdGy', vmin=-1.6, vmax=1.6) # Residuals
    cax6 = fig.colorbar(cimg6)

    opt_residuals = np.mean(model_isomap_opt, axis=2) - data_isomap
    ax7 = fig.add_subplot(7,1,7)
    #ax5text = 'forward model expectation'
    #ax5.text(21, 3, ax5text, fontsize=8)
    cimg7 = ax7.imshow(opt_residuals.T, aspect='auto', interpolation='nearest', origin='lower', cmap='RdGy', vmin=-1.6, vmax=1.6) # Residuals
    cax7 = fig.colorbar(cimg7)

    #fig.savefig('PO4eq18p6_50k_new5_{0}a.svg'.format(t_save), dpi=300, bbox_inches='tight')
    #plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    blood_hist = blood_delta(23.5, w_iso_hist, 25.3, **kwargs) # Should be w_iso_hist
    phosphate_eq = PO4_dissoln_reprecip(PO4_t, PO4_pause, PO4_flux, blood_hist, **kwargs)
    days = np.arange(blood_hist.size)
    ax1.plot(days+2, w_iso_hist, 'b-', linewidth=2.0)
    ax1.plot(days+2, blood_hist, 'r-', linewidth=2.0)
    ax1.plot(days+2, phosphate_eq, 'g-.', linewidth=1.0)
    ax1.plot(blood_days, blood_measures, 'r*', linewidth=1.0)
    ax1.plot(water_iso_days, water_iso_measures, 'b*', linewidth=1.0)
    for s in list_water_results[:-1]:
        ax1.plot(days+2, s, 'g-', alpha=0.05)
    #vmin = np.min(np.concatenate((real_switch_hist, w_iso_hist, blood_hist), axis=0)) - 1.
    #vmax = np.max(np.concatenate((real_switch_hist, w_iso_hist, blood_hist), axis=0)) + 1.
    ax1.text(350, -15, textstr, fontsize=8)
    ax1.set_ylim(-20, -4.5)
    ax1.set_xlim(100, 400)

    #fig.savefig('PO4eq18p6_50k_new5_{0}b.svg'.format(t_save), dpi=300, bbox_inches='tight')
    #plt.show()

    residuals_real = np.isfinite(residuals)
    trial_real = np.isfinite(trial_residuals)
    opt_real = np.isfinite(opt_residuals)
    data_real = np.isfinite(data_isomap)

    min_max = (
                np.min(
                [np.min(residuals[residuals_real]),
                np.min(trial_residuals[trial_real]),
                np.min(opt_residuals[opt_real]),
                np.min(data_isomap[data_real])]),
                np.max(
                [np.max(residuals[residuals_real]),
                np.max(trial_residuals[trial_real]),
                np.max(opt_residuals[opt_real]),
                np.max(data_isomap[data_real])]) )

    trial_weights = np.ones_like(trial_residuals[trial_real])/len(trial_residuals[trial_real])
    residuals_weights = np.ones_like(residuals[residuals_real])/len(residuals[residuals_real])
    opt_weights = np.ones_like(opt_residuals[opt_real])/len(opt_residuals[opt_real])

    normals = np.random.normal(0., .25, 100000)
    normal_weights = np.ones_like(normals)/len(normals)

    xg = np.linspace(-3,3,1000)
    gaus = 1/(np.sqrt(2*np.pi)) * np.exp(-(xg**2)/(2*(.25**2)))

    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax1.hist(trial_residuals[trial_real], bins=(np.linspace(-3., 3., 24)), weights=trial_weights, histtype='stepfilled', normed=False, color='#0040FF', alpha=.8, label='Low')
    ax1.plot(xg, gaus, 'k--')
    #ax1.hist(normals, bins=(np.linspace(-3,3,24)), weights=normal_weights, alpha=.3)
    ax1.set_ylim(0, .45)
    ax2 = fig.add_subplot(3,1,3)
    ax2.hist(residuals[residuals_real], bins=(np.linspace(-3., 3., 24)), weights=residuals_weights, histtype='stepfilled', normed=False, color='#0040FF', alpha=.8, label='Low')
    ax2.plot(xg, gaus, 'k--')
    #ax2.hist(normals, bins=(np.linspace(-3,3,24)), weights=normal_weights, alpha=.3)
    ax2.set_ylim(0, .45)
    ax3 = fig.add_subplot(3,1,2)
    ax3.hist(opt_residuals[opt_real], bins=(np.linspace(-3., 3., 24)), weights=opt_weights, histtype='stepfilled', normed=False, color='#0040FF', alpha=.8, label='Low')
    ax3.plot(xg, gaus, 'k--')
    #ax3.hist(normals, bins=(np.linspace(-3,3,24)), weights=normal_weights, alpha=.3)
    ax3.set_ylim(0, .45)
    fig.savefig('PO4eq18p6_50k_new5_{0}c.svg'.format(t_save), dpi=300, bbox_inches='tight')
    plt.show()


def main():

    fit_tooth_data('/Users/darouet/Documents/code/mineralization/clean code/962_tooth_iso_data.csv')

    return 0

if __name__ == '__main__':
    main()




