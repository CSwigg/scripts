import sys
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/python-fsps')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/sedpy')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/prospector')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/p_scripts')

import fsps
import sedpy
import prospect

def build_sps_csps(zcontinuous=1, **extras):
    """
    :param zcontinuous: 
        A vlue of 1 insures that we use interpolation between SSPs to 
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps
