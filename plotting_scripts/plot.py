import time, sys, os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/sedpy')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/prospector')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/p_scripts')

sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/cam_models') # My directory of models
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/cam_sps') # My directory of sps
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/cam_obs') # My directory of obs
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/stats') # My directory of stat functions
#sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/prospector_scripts/plotting_scripts') # My directory of plotting functions

sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/python_fsps_c3k')
import fsps


import sedpy
import prospect
from prospect.utils.obsutils import fix_obs
import prospect.io.read_results as reader
from helper_functions import * 

from non_parametric_model_1 import build_model
from non_parametric_model_old import build_model_old
from non_parametric_model_spec_cal import build_model_new
from non_parametric_model_nir_fix import build_model_nir_fix
from non_parametric_model_add_bin import build_model_add_bin
from non_parametric_model_latest import build_model_latest
from parametric_model_1 import build_model_parametric 
from fast_step_basis_sps import build_sps
from csps_step_basis_sps import build_sps_csps
from obs_1_spectra import *
from calc_likelihood import *

import emcee
import dynesty


def generate_model(theta, obs, sps, model):
    
    obs['wavelength'] = None
    mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)

    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    wphot = obs["phot_wave"]
    if obs["wavelength"] is None:
        wspec = sps.wavelengths
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]
        
    return wspec, mspec, mphot, wphot, mextra


def get_from_reader(filename, g_name, hizea_file, model_type):
    result, obs, _ = reader.results_from(filename, dangerous=False)
    data = hizea_file.data[0]
    run_params = result['run_params']
    run_params['g_name'] = g_name


    if model_type == 'parametric':
        model = build_model_parametric(**run_params)
        sps = build_sps_csps(**run_params)

    elif model_type == 'non_parametric':
        #model = build_model_new(**run_params)
        model, n_params = build_model_latest(**run_params, obs = obs)
        #model, n_params = build_model_add_bin(**run_params, obs = obs)
        sps = build_sps(**run_params)

    elif model_type == 'non_parametric_old':
        model = build_model_old(**run_params)
        sps = build_sps(**run_params)


    imax = np.argmax(result['lnprobability'])
    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()
    
    return theta_max, obs, sps, model, run_params, result





def make_corner_plot(galaxy_file, g_name, hizea_file, results_dir, n_params, model_type):
    
    theta_max, obs, sps, model, run_params, result = get_from_reader(galaxy_file, g_name, hizea_file, model_type) 
    
    thin = 5
    cornerfig = reader.subcorner(result, start=0, thin=thin, truths=theta_max,
                                 fig=plt.subplots(n_params, n_params, figsize=(25,25))[0], plot_datapoints = False)
    plt.savefig('{}{}/cornerplot_new.png'.format(results_dir, g_name), dpi = 300)
    plt.close()


def make_traceplot(galaxy_file, g_name, hizea_file, results_dir, model_type):
    
    theta_max, obs, sps, model, run_params, result = get_from_reader(galaxy_file, g_name, hizea_file, model_type) 
    tracefig = reader.traceplot(result, figsize=(28,18))
    plt.savefig('{}{}/traceplot_new.png'.format(results_dir, g_name), dpi = 300, overwrite = True)
    plt.close()



def plot_best_spec(ax_in, obs, wspec, mspec, mphot, wphot, run_params):

    z = run_params['object_redshift']
    
    ax_in.set_xlabel('Wavelength [AA]')
    ax_in.set_ylabel('Flux [maggies]')
    
    ax_in.plot(wspec/(1+z), mspec, lw = .2, color = 'blue', alpha = 0.5, label = 'Model Spectrum',zorder=1)
    
    
    ax_in.scatter(obs['phot_wave']/(1+z),obs['maggies'], c = 'magenta', s = 30, label = 'Data',zorder=2)
    ax_in.errorbar(obs['phot_wave']/(1+z),obs['maggies'], obs['maggies_unc'], c = 'magenta', linestyle = 'None',zorder=2)
    ax_in.scatter(obs['phot_wave']/(1+z), mphot, c = 'red', s = 30, label = 'Model photometry',zorder=2)

    ax_in.set_xscale('log')
    ax_in.set_yscale('log') 
    

    
    return ax_in


def random_draw(ax, run_params, obs, sps, model, result, flux_shift = False):
    randint = np.random.randint
    theta_rand_list = []
    nwalkers, niter = run_params['nwalkers'], run_params['niter']
    z = run_params['object_redshift']
    for i in range(100):
        theta = result['chain'][randint(nwalkers), randint(niter)]
        wspec, mspec, mphot, wphot, mextra = generate_model(theta, obs, sps, model) 
        
        if flux_shift:
            mspec = mspec/(1+z)

        ax.plot(wspec/(1+z), mspec, lw = 0.1, alpha = 0.5, color = 'gray', zorder = 0)
    return ax



def make_sed_plot(galaxy_file, g_name, hizea_file, results_dir, model_type, set_limits = True, return_fig_ax = False):

    theta_max, obs, sps, model, run_params, result = get_from_reader(galaxy_file, g_name, hizea_file, model_type) 

    wave_spectrum = obs['wavelength']
    flux_spectrum = obs['spectrum']
    #obs['wavelength'] = None


    # SED plot
    wspec, mpsec, mphot, wphot, mextra = generate_model(theta_max, obs, sps, model)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(g_name)

    z = run_params['object_redshift']
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    #xmin, xmax = min(wave_spectrum/(1+z)), max(wave_spectrum/(1+z))
    #xmin, xmax = 2500, 7000
    ymin, ymax = obs["maggies"].min()*0.3, obs["maggies"].max()/0.8

    if set_limits:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    spectrum = fits.open('/Users/cam/Downloads/hizea_mask/{}_mask.fit'.format(g_name))[1]
    z = run_params['object_redshift']
    w = spectrum.data['wave'][0]
    f = spectrum.data['flux'][0]
    f = (f*3.34e4*w**2*1e-17)/3631

    # obs frame: ax.plot(w*(1+z), f*(1+z), lw = 0.1, c = 'green', alpha = 0.5, zorder = 0)
    range_cut = np.where((w > 3500) & (w < 4200))
    ax.plot(w, f*(1+z), lw = 0.1, c = 'green', alpha = 0.5, zorder = 0)
    ax.plot(w[range_cut], f[range_cut]*(1+z), c = 'orange', lw = .5, label = 'fit region', zorder = 0)
    ax = random_draw(ax, run_params, obs, sps, model, result)
    ax = plot_best_spec(ax, obs, wspec, mpsec, mphot, wphot, run_params)

    plt.legend()
    
    if return_fig_ax:
        return (fig, ax)
    else:
        plt.savefig('{}{}/{}_sed_plot_full.png'.format(results_dir, g_name, g_name), dpi = 300, overwrite = True)
        plt.close()
        

