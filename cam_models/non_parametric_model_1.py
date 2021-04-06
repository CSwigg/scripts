import sys 
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/python-fsps')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/sedpy')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/prospector')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/p_scripts')

import fsps
import sedpy
import prospect
import numpy as np

def build_model(object_redshift=None, ldist=10.0, fixed_metallicity=None, add_duste=False, 
                **extras):

    from prospect.models.sedmodel import SedModel, SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, transforms

    model_params = TemplateLibrary["continuity_sfh"]
    nbins_sfh = 11
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh-1
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                  scale=np.full(nbins_sfh-1,0.3),
                                                                  df=np.full(nbins_sfh-1,2))
    
    
    model_params["dust1"] = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                        "init": 1., "units": "optical depth towards young stars"}

    model_params["dust_ratio"] = {"N": 1, "isfree": True, 
                             "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                             "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params["dust_index"] = {"N": 1, "isfree": True,
                             "init": 0.0, "units": "power-law multiplication of Calzetti",
                             "prior": priors.TopHat(mini=-2.0, maxi=1.)}
    


    log_agebins = np.array([[0., 7.], 
                            [7., 7.25],
                            [7.25, 7.5],
                            [7.5, 7.75],
                            [7.75, 8.],
                            [8., 8.25],
                            [8.25, 8.5],
                            [8.5, 8.75],
                            [8.75, 9.],
                            [9., 9.5],
                            [9.5, 10.]
                            ])
    #agebins_linear = np.array([[0.,.2],
    #                           [.2,1],
    #                           [1,5.6],
    #                           [5.6,6.3],
    #                           [6.3,16],
    #                           [16,28],
    #                           [28,50],
    #                           [50,100],
    #                           [100,150],
    #                           [150,250],
    #                           [250,500],
    #                           [500,1000],
    #                           [1000,2500],
    #                           [2500,5000],
    #                           [5000,9000]
    #                          ])*1e6 #Myr
    
    #log_agebins = np.log10(agebins_linear)
    model_params['agebins']['init'] = log_agebins
    
    model_params["logmass"]["init"] = 11.
    model_params["dust2"]["init"] = 0.5
    model_params["logzsol"]["init"] = 0.0

    
    model_params["logmass"]["prior"] = priors.TopHat(mini=10., maxi=12.)
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1., maxi=1.0)
    
    
    if fixed_metallicity is not None:
        model_params["logzsol"]["isfree"] = False
        model_params["logzsol"]['init'] = fixed_metallicity 

    if object_redshift is not None:
        model_params["zred"]['isfree'] = False
        model_params["zred"]['init'] = object_redshift

    if add_duste:        
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_umin']['isfree'] = True
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_gamma']['isfree'] = True
        
        model_params['duste_umin']['prior'] = priors.LogUniform(mini=0.1, maxi=15.)
        model_params['duste_qpah']['prior'] = priors.LogUniform(mini=0.1, maxi=7.)
        model_params['duste_gamma']['prior'] = priors.LogUniform(mini=0.001, maxi=0.15)
        
        model_params['duste_umin']['init'] = 7.5
        model_params['duste_qpah']['init'] = 3.5
        model_params['duste_gamma']['init'] = 0.075

        model_params["duste_qpah"]["disp_floor"] = 3.0
        model_params["duste_umin"]["disp_floor"] = 4.5
        model_params["duste_gamma"]["disp_floor"] = 0.15
        model_params["duste_qpah"]["init_disp"] = 3.0
        model_params["duste_umin"]["init_disp"] = 5.0
        model_params["duste_gamma"]["init_disp"] = 0.2
        
    #model_params['dust2']['disp_floor'] = 1e-2 
    #model_params['dust1']['disp_floor'] = 1e-2 
    #model_params['logzsol']['disp_floor'] = 1e-3
    #model_params['dust_index']['disp_floor'] = 1e-3


    model = SedModel(model_params)

    return model

