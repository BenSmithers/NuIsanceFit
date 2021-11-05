"""
This script is used to generate fluxes that we will use to do the weighting! 

Should be able to make these fluxes
    - conventional atmo
    - prompt atmo
    - astro 
"""

from scipy import interpolate
from NuIsanceFit import steering
from NuIsanceFit.utils import NewPhysicsParams
from NuIsanceFit import Logger

import os
import pickle


# simulates the cosmic ray showers 
from MCEq.core import MCEqRun
import crflux.models as crf

import numpy as np
from math import sin, asin, acos, pi

from scipy.interpolate import interp1d

# CONSTANTS 
energy_bins = 121
angular_bins = 50
interaction_model = "SIBYLL23C"
flux_model = (crf.HillasGaisser2012, 'H3a')
r_e = 6.378e6 # meters
ic_depth = 1.5e3 # meters 

def _get_key(flavor:int, neutrino:int)->str:
    """
    Little function to get the key to access the mceq dictionary
    You just give it the flavor/neutrino index 
    """
    # flavor 0, 1, 2, 3
    # neutrino 0, 1

    key = ""
    if flavor==0:
        key+="nue_"
    elif flavor==1:
        key+="numu_"
    elif flavor==2:
        key+="nutau_"
    elif flavor==3:
        return("") # sterile 
    else:
        raise ValueError("Invalid flavor {}".format(flavor))

    if neutrino==0:
        key+="flux"
    elif neutrino==1:
        key+="bar_flux"
    else:
        raise ValueError("Invalid Neutrino Type {}".format(neutrino))

    return(key)

def _prompt_initial_state(energies, zeniths, n_nu, **kwargs):
    return _atmo_initial_state(energies, zeniths, n_nu, flux="prompt", **kwargs)

def _conv_initial_state(energies, zeniths, n_nu, **kwargs):
    return _atmo_initial_state(energies, zeniths, n_nu, flux="conventional", **kwargs)

def _atmo_initial_state(energies, zeniths, n_nu, **kwargs):
    """
        Take lists of values at which we wish to know the neutrino fluxes! 
    """
    # adapt filename incase this isn't prompt or whatever 
    if "flux" in kwargs:
        ftype = kwargs["flux"].lower()
        if ftype == "conventional":
            ftype = "conv"
    else:
        ftype = "both"
    
    # use this to stick the "_both" or "_conv" at the end of the filename but before the extention ".pkl"
    subname = ".".join( steering["mceq_filename"].split(".")[:-1] )
    subname+= "_"+ftype
    subname = ".".join([subname,steering["mceq_filename"].split(".")[-1] ])

    path = os.path.join(steering["resource_dir"], "fluxes", subname)
    if os.path.exists(path):
        Logger.Log("Loading MCEq Flux from {}".format(path))
        f = open(path, 'rb')
        mceq_data_dict = pickle.load(f)
        f.close()
    else:
        Logger.Log("Generating MCEq Flux!")


        inistate = np.zeros(shape=(angular_bins, energy_bins, 2, n_nu))
        mceq = MCEqRun(
                interaction_model = interaction_model,
                primary_model = flux_model,
                theta_deg = 0.
                )
        
        mag = 0. # power energy is raised to and then used to scale the flux
        for angle_bin in range(angular_bins):
            # get the MCEq angle from the icecube zenith angle 
            angle_deg = asin(sin(pi-acos(zeniths[angle_bin]))*(r_e-ic_depth)/r_e)
            angle_deg = angle_deg*180./pi
            if angle_deg > 180.:
                angle_deg = 180.

            print("Evaluating {} deg Flux".format(angle_deg))
            # for what it's worth, if you try just making a new MCEqRun for each angle, you get a memory leak. 
            # so you need to manually set the angle 
            mceq.set_theta_deg(angle_deg)
            mceq.solve()

            mceq_data_dict = {}
            mceq_data_dict['e_grid'] = mceq.e_grid

            
            if ftype == "both":
                mceq_data_dict['nue_flux'] = mceq.get_solution('total_nue',mag)
                mceq_data_dict['nue_bar_flux'] = mceq.get_solution('total_antinue',mag)
                mceq_data_dict['numu_flux'] = mceq.get_solution('total_numu',mag)
                mceq_data_dict['numu_bar_flux'] = mceq.get_solution('total_antinumu',mag)
                mceq_data_dict['nutau_flux'] = mceq.get_solution('total_nutau',mag)
                mceq_data_dict['nutau_bar_flux'] = mceq.get_solution('total_antinutau',mag)
            elif ftype == "conv" or ftype=="conventional":
                mceq_data_dict['nue_flux'] = mceq.get_solution('conv_nue',mag)
                mceq_data_dict['nue_bar_flux'] = mceq.get_solution('conv_antinue',mag)
                mceq_data_dict['numu_flux'] = mceq.get_solution('conv_numu',mag)
                mceq_data_dict['numu_bar_flux'] = mceq.get_solution('conv_antinumu',mag)
                mceq_data_dict['nutau_flux'] = mceq.get_solution('conv_nutau',mag)
                mceq_data_dict['nutau_bar_flux'] = mceq.get_solution('conv_antinutau',mag)
            elif ftype=="prompt":
                mceq_data_dict['nue_flux'] = mceq.get_solution('pr_nue',mag)
                mceq_data_dict['nue_bar_flux'] = mceq.get_solution('pr_antinue',mag)
                mceq_data_dict['numu_flux'] = mceq.get_solution('pr_numu',mag)
                mceq_data_dict['numu_bar_flux'] = mceq.get_solution('pr_antinumu',mag)
                mceq_data_dict['nutau_flux'] = mceq.get_solution('pr_nutau',mag)
                mceq_data_dict['nutau_bar_flux'] = mceq.get_solution('pr_antinutau',mag)
            else:
                raise ValueError("Unrecognized atmospheric flux arg: '{}'".format(kwargs["flux"]))
                
            

            for neut_type in range(2):
                for flavor in range(n_nu):
                    flav_key = _get_key(flavor, neut_type)
                    if flav_key=="":
                        continue

                    int_func = interp1d(mceq_data_dict["e_grid"], mceq_data_dict[flav_key], assume_sorted=True)
                    for energy_bin in range(energy_bins):
                        # (account for the difference in units between mceq and nusquids! )
                        inistate[angle_bin][energy_bin][neut_type][flavor] = int_func(energies[energy_bin]/(1e9))
        if np.min(inistate)<0:
            raise ValueError("Found negaitve flux in initial state, something's wrong!")
        
        f = open(path, 'wb')
        pickle.dump(inistate, f, -1)
        f.close()
    
    return inistate


def astr_initial_state():
    pass