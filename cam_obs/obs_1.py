import sys
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/python-fsps')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/sedpy')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/prospector')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/p_scripts')

import fsps
import sedpy
import prospect
from prospect.utils.obsutils import fix_obs
import numpy as np



def build_obs(object_data, object_redshift, **extras):
    """Build a dictionary of observational data.  
    
    :returns obs:
    """
    
    data = object_data
    
    # The obs dictionary, empty for now
    obs = {}

    galex = ['galex_FUV', 'galex_NUV']
    sdss = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']]
    wise = ['wise_W1', 'wise_W2', 'wise_W3', 'wise_W4']
    vista = ['vista_{0}'.format(b) for b in ['y', 'j', 'h', 'k']] 
    ukidss = ['ukidss_{0}'.format(b) for b in ['y', 'j', 'h', 'k']] 
    

    
    if (np.any(data['VISTA_VEGAMAG']) != 0.):
        filternames = galex + sdss + vista + wise
        obs['ukidss_or_vista'] = 'VISTA'

    elif (np.any(data['UKIDDS_VEGAMAG']) != 0.):
        filternames = galex + sdss + ukidss + wise
        obs['ukidss_or_vista'] = 'UKIDSS'

    else:
        filternames = galex + sdss + ukidss + wise
        obs['ukidss_or_vista'] = 'N/A'


    filternames = np.array(filternames)
    

    M_AB = np.array(data['AB_MAG'])
    flux_flam = np.array(data['FLUX_FLAM'])
    flux_flam_err = np.array(data['FLUX_FLAM_ERR'])

    
    indices_good = np.where((np.abs(M_AB) != 99.) & 
            (~np.isnan(M_AB)) & (M_AB > 15.00) & (flux_flam != 0.) & 
            (filternames != 'vista_y')
            )
    

    flux_flam = flux_flam[indices_good[0]]
    flux_flam_err = flux_flam_err[indices_good[0]]
    
        
    wave = data['PHOT_WAVE'][indices_good[0]]
    obs['phot_wave'] = wave
    obs["filters"] = sedpy.observate.load_filters(filternames[indices_good[0]])
    
    maggies = (flux_flam*3.34e4*wave**2*1e-17)/(3631)
    maggies_unc = (flux_flam_err*3.34e4*wave**2*1e-17)/(3631)
    obs["maggies"] = maggies
    obs["maggies_unc"] = maggies_unc
    
    obs["wavelength"] = None
    obs["spectrum"] = None
    obs['unc'] = None
    obs['mask'] = None
    obs = fix_obs(obs)

    return obs 

