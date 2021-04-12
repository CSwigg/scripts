'''
Model that adopts a parametric delayed-tau SFH, two component dust attenuation,
and dust emission. 
'''
import sys 
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/python-fsps')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/sedpy')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/prospector')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/p_scripts')

import fsps
import sedpy
import prospect
import numpy as np

def build_model_parametric(object_redshift=None, ldist=10.0, fixed_metallicity=None, add_duste=False, 
                **extras):

    from prospect.models.sedmodel import SedModel, SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, transforms

    model_params = TemplateLibrary["parametric_sfh"]
    
    model_params["dust1"] = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                        "init": 1., "units": "optical depth towards young stars"}

    model_params["dust_ratio"] = {"N": 1, "isfree": True, 
                             "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                             "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params["dust_index"] = {"N": 1, "isfree": True,
                             "init": 0.0, "units": "power-law multiplication of Calzetti",
                             "prior": priors.TopHat(mini=-2.0, maxi=1.)}

    model_params["logmass"] = {"N": 1, "isfree": True,
                             "init": 10.25, "units": "log(total mass formed)",
                             "prior": priors.TopHat(mini=9.5, maxi=12)}
    


    #log_agebins = np.log10(agebins_linear)
    model_params["dust2"]["init"] = 0.5
    model_params["logzsol"]["init"] = 0.0
    model_params["tau"]["init"] = priors.LogUniform(mini=1e-1, maxi=1e2)

    
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1., maxi=1.0)

    model_params["tage"]["isfree"] = True
    model_params["tage"]["init"] = 5.
    model_params["tage"]["prior"] = priors.LogUniform(mini=1, maxi=7)

    model_params["tau"]["isfree"] = True
    model_params['tau']['init'] = 1.
    model_params["tau"]["prior"] = priors.LogUniform(mini=.1, maxi=15)
    
    
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
        
    model_params['dust2']['disp_floor'] = 1e-2 
    model_params['dust1']['disp_floor'] = 1e-2 
    model_params['logzsol']['disp_floor'] = 1e-3
    #model_params['dust_index']['disp_floor'] = 1e-3


    model = SedModel(model_params)

    return model

