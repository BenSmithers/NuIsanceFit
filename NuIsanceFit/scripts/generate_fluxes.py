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
from NuIsanceFit import weighter
from weighter import FluxComponent

try:
    import nuSQuIDS as nsq
except ImportError:
    import nuSQUIDSpy as nsq

import os
import pickle

# simulates the cosmic ray showers 
from MCEq.core import MCEqRun
import crflux.models as crf

import numpy as np
from math import sin, asin, acos, pi, log10

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

# this will need to be modifed for distributed computing jobs 
xs_obj = nsq.loadDefaultCrossSections()

def evolve_flux(which:FluxComponent, params:NewPhysicsParams, **kwargs) -> str:
    """
        Generates a nuSQUIDSAtm object using a given flux component and a set of new physics parameters.
        Writes this object to an hdf5 file, and returns the filepath to the generated hdf5 file. 
    """
    Logger.Log("Generating Flux")

    if which==FluxComponent.atmConv:
        state_setter = _conv_initial_state
        flux_name = "conv"
    elif which==FluxComponent.atmPrompt:
        state_setter = _prompt_initial_state
        flux_name = "prompt"
    elif which==FluxComponent.diffuseAstro:
        state_setter = _astr_initial_state
        flux_name = "astr"
    else:
        raise NotImplementedError("Unimplemented flux component: {}".format(which))

    expected_kwargs = ["force_path", "force_filename"]
    for kwarg in kwargs:
        if kwarg not in expected_kwargs:
            raise ValueError("Unexpected keyword argument: '{}'".format(kwarg))

    if "force_path" in kwargs:
        root_dir = kwargs["force_path"]
    else:
        root_dir = os.path.join(steering["resource_dir"], "fluxes")

    if "force_filename" in kwargs:
        full_name = kwargs["force_filename"]
        full_name = "nus_atm_" + flux_name + ".h5"

    n_nu = 4
    Emin = 1.*(1e9)
    Emax = 10.*(1e15)
    cos_zenith_min = -0.999
    cos_zenith_max = 0.2

    use_earth_interactions = True

    zeniths = nsq.linspace(cos_zenith_min, cos_zenith_max, angular_bins)
    energies = nsq.logspace(Emin, Emax, energy_bins) # DIFFERENT FROM NUMPY LOGSPACE

    nus_atm = nsq.nuSQUIDSAtm(zeniths, energies, n_nu, nsq.NeutrinoType.both, use_earth_interactions)

    nus_atm.Set_MixingAngle(0,1,0.563942)
    nus_atm.Set_MixingAngle(0,2,0.154085)
    nus_atm.Set_MixingAngle(1,2,0.785398)
    nus_atm.Set_SquareMassDifference(1,7.65e-05)
    nus_atm.Set_SquareMassDifference(2,0.00247)

    #sterile parameters 
    nus_atm.Set_MixingAngle(0,3,params.theta03)
    nus_atm.Set_MixingAngle(1,3,params.theta13)
    nus_atm.Set_MixingAngle(2,3,params.theta23)
    nus_atm.Set_SquareMassDifference(3,params.msq2)

    nus_atm.SetNeutrinoCrossSections(xs_obj)

    nus_atm.Set_TauRegeneration(True)

    #settting some zenith angle stuff 
    nus_atm.Set_rel_error(1.0e-6)
    nus_atm.Set_abs_error(1.0e-6)
    #nus_atm.Set_GSL_step(gsl_odeiv2_step_rk4)
    nus_atm.Set_GSL_step(nsq.GSL_STEP_FUNCTIONS.GSL_STEP_RK4)

    # we load in the initial state. Generating or Loading from a file 
    inistate = state_setter(energies, zeniths, n_nu, **kwargs)
    if np.min(inistate)<0:
        raise ValueError("Found negative value in inistate: {}".format(np.min(inistate)))
    nus_atm.Set_initial_state(inistate, nsq.Basis.flavor)

    Logger.Log("Initial State Set, evolving")
    # we turn off the progress bar for jobs run on the cobalts 
    nus_atm.Set_ProgressBar(False)
    nus_atm.Set_IncludeOscillations(True)

    nus_atm.EvolveState()

    
    fname = os.path.join(root_dir, full_name)

    nus_atm.WriteStateHDF5( fname )


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


def _astr_initial_state(energies, zeniths, n_nu, **kwargs):
    """
    Sets up the initial state (pre-osc) for an astrophysical neutrino flux

    Basically a sad looking power-law
    """

    pivot = 100*(1e12) # eV, needs to be in nuSQuIDS units 
    norm = 1e-18 # [GeV s cm^2 sr]^-1  <- verify
    gamma = -2.5

    inistate = np.zeros(shape=(angular_bins, energy_bins, 2, n_nu))
    
    def get_flux(energy):
        return norm*(energy/pivot)**(gamma)

    for i_e in range(energy_bins):
        flux = get_flux(energies[i_e])
        if flux<0:
            raise ValueError("Somehow got negative flux... {}".format(flux))
        for flavor in range(n_nu):
            for i_a in range(angular_bins):
                for neut_type in range(2):
                    inistate[i_a][i_e][neut_type][flavor] += flux*(kwargs["flavor_ratio"][flavor])
    return inistate


if __name__=="__main__":
    evolve_flux(FluxComponent.atmConv, NewPhysicsParams())
    evolve_flux(FluxComponent.atmPrompt, NewPhysicsParams())
    evolve_flux(FluxComponent.diffuseAstro, NewPhysicsParams())
