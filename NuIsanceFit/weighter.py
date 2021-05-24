from NuIsanceFit.event import Event, EventCache
from NuIsanceFit.param import ParamPoint, params as global_params
from NuIsanceFit.logger import Logger 
from NuIsanceFit.histogram import get_loc

import photospline as ps
from enum import Enum
from math import isnan
import os 
import numpy as np
from glob import glob # finding the attenuation splines
"""
Here we design and implement classes to Weight events

Later on, we might need to implement distinct treatments for derivavites.Consider...
    from scipy.misc import derivative
This is only 1D differentiation, but you can make it work with some lambda functions to evaluate a 1D derivative of an arbitrary function (at some point)
"""


def fill_fluxcomp_dict(folder):
    files = glob(folder+"/*.fits")
    if len(files)==0:
        Logger.Fatal("Didn't find any splines in {}".format(folder), IOError)

    ret_dict = {}
    for item in files:
        Logger.Trace("    {}".format(item))
        dirname, filename = os.path.split(item)

        # cut off the extension
        filename = ".".join(filename.split(".")[:-1])

        # access the different parts (nuMu, numuBar, etc...)
        broken_up = filename.split("_")

        subComponent = broken_up[-1] # TODO: Update this when we have dedicated particle types from nuSQuIDS or LeptonWeighter <--- this one  
        if "mu" in subComponent.lower():
            pid = 13
        elif "e" in subComponent.lower():
            pid = 15
        elif "tau" in subComponent.lower():
            pid = 17
        else:
            pid = subComponent
        if "nu" in subComponent.lower():
            pid+=1
        if "minus" in subComponent.lower() or "bar" in subComponent.lower():
            pid*=-1
        


        component1 = broken_up[-3]
        component2 = broken_up[-2]

        # we do this because of unpleasant naming conventions for the splines 
        if component2=="mu" and component1=="diffuseAstro":
            fc = "_".join([component1,component2]) #eg, duffuseAstro_mu
        else:
            fc = component2#eg, atmConv

        try:
            fluxComponent = getattr(FluxComponent, fc)
        except AttributeError:
            Logger.Warn("Working in {}".format(folder))
            Logger.Warn("File {}".format(item))
            Logger.Fatal("Failed", ValueError)
        
        # make sure we have the right structure to do this
        if fluxComponent not in ret_dict:
            ret_dict[fluxComponent] = {}

        # load in the spline 
        ret_dict[fluxComponent][pid] = ps.SplineTable(item)

    return ret_dict


# ========================= Weighter Maker Code =====================================
class SimWeighter:
    """
    This object can take a set of parameters and create a Meta-MetaWeighter that weights events according to that set of parameters 

    Note that the weights are only meaningful in a relative sense. We don't care about constants 
    """
    def __init__(self, steering, dtype=float):
        """
        This function will load in and cache anything needed to make the Weighters.
        Splines and stuff... 
        """
        resources = steering["resources"]

        self.medianConvEnergy = steering["fixed_params"]["medianConvEnergy"]
        self.medianPromptEnergy = steering["fixed_params"]["medianPromptEnergy"]
        self.astroPivotEnergy = steering["fixed_params"]["astroPivotEnergy"]

        Logger.Log("Loading in splines for Weighting")
        Logger.Trace("    {}".format(resources["atmospheric_density_spline"]))
        self._atmosphericDensityUncertaintySpline = ps.SplineTable(resources["atmospheric_density_spline"])
        Logger.Trace("    {}".format(resources["atmospheric_kaonlosses_spline"]))
        self._kaonLossesUncertaintySpline = ps.SplineTable(resources["atmospheric_kaonlosses_spline"])

        self._attenuationSplineDict = fill_fluxcomp_dict(resources["attenuation_splines"])
        self._domSplines = fill_fluxcomp_dict(resources["dom_splines"])
        self._hqdomSplines = fill_fluxcomp_dict(resources["hq_dom_splines"])

        self._holeiceSplines = fill_fluxcomp_dict(resources["hole_ice_splines"])

        # Now the ice gradients?
        eff_grad_files = glob(resources["ice_gradients"]+"/Energy_Eff_Grad_*.txt")
        if len(eff_grad_files)==0:
            Logger.Fatal("Found no files in {}".format(resources["ice_gradients"]))
        # in these 
        #   0 is left side of bin
        #   1 is right side of bin
        #   2 is value in the bin
        self._ice_grad_data = [np.loadtxt(fname, dtype=float, delimiter=" ").transpose() for fname in eff_grad_files]

        params = ParamPoint()
        self.params = params

        Logger.Trace("Creating new metaWeighter")

        #Logger.Trace("Conv")
        #conventionalComponent   = PowerLawTiltWeighter(params["convNorm"]*1e5, -2.5 + params["CRDeltaGamma"])
        #Logger.Trace("Prompt")
        #promptComponent         = PowerLawTiltWeighter(params["promptNorm"]*1e5, -2.5 )
        #Logger.Trace("Astro")
        #astroComponent          = PowerLawTiltWeighter(params["astroNorm"]*1e5, -2.0 + params["astroDeltaGamma"])

        # The caches! 
        self.astroMuFlux = CachedValueWeighter("astroMuWeight")
        self.convPionFlux = CachedValueWeighter("convPionWeight")
        self.convKaonFlux = CachedValueWeighter("convKaonWeight")
        self.promptFlux = CachedValueWeighter("promptWeight")
        self.barrWPComp = CachedValueWeighter("barrModWP")
        self.barrWMComp = CachedValueWeighter("barrModWM")
        self.barrYPComp = CachedValueWeighter("barrModYP")
        self.barrYMComp = CachedValueWeighter("barrModYM")
        self.barrZPComp = CachedValueWeighter("barrModZP")
        self.barrZMComp = CachedValueWeighter("barrModZM")

        self.neuaneu_w = AntiparticleWeighter(params["NeutrinoAntineutrinoRatio"])

        self.aduw = AtmosphericUncertaintyWeighter(self._atmosphericDensityUncertaintySpline, params["zenithCorrection"])
        self.kluw = AtmosphericUncertaintyWeighter(self._kaonLossesUncertaintySpline, params["kaonLosses"])

        self.conv_nu_att_weighter = AttenuationWeighter(self._attenuationSplineDict, FluxComponent.atmConv, params["nuxs"], params["nubarxs"], dtype)
        self.prompt_nu_att_weighter = AttenuationWeighter(self._attenuationSplineDict, FluxComponent.atmPrompt, params["nuxs"], params["nubarxs"], dtype)
        self.astro_nu_att_weighter = AttenuationWeighter(self._attenuationSplineDict, FluxComponent.diffuseAstro_mu, params["nuxs"], params["nubarxs"], dtype)

        self.convDOMEff = TopoWeighter(self._domSplines, FluxComponent.atmConv, params["domEfficiency"], dtype)
        self.promptDOMEff = TopoWeighter(self._domSplines, FluxComponent.atmPrompt, params["domEfficiency"], dtype)
        self.astroNuMuDOMEff = TopoWeighter(self._domSplines, FluxComponent.diffuseAstro_mu, params["domEfficiency"], dtype)

        # these aren't actually used. 
        #hqconvDOMEff = TopoWeighter(self._hqdomSplines, FluxComponent.atmConv, params["hqdomEffficiency"],dtype)
        #hqpromptDOMEff = TopoWeighter(self._hqdomSplines, FluxComponent.atmPrompt, params["hqdomEffficiency"],dtype)

        self.convHoleIceWeighter = TopoWeighter(self._holeiceSplines, FluxComponent.atmConv, params["holeiceForward"], dtype)
        self.promptHoleIceWeighter = TopoWeighter(self._holeiceSplines, FluxComponent.atmPrompt, params["holeiceForward"], dtype)
        self.astroNuMuHoleIceWeighter = TopoWeighter(self._holeiceSplines, FluxComponent.diffuseAstro_mu, params["holeiceForward"], dtype)

        edges, values = self._extract_edges_values(self._ice_grad_data[0])
        self.ice_grad_0 = IceGradientWeighter(edges, values, params["icegrad0"], dtype)

        edges, values = self._extract_edges_values(self._ice_grad_data[1])
        self.ice_grad_1 = IceGradientWeighter(edges, values, params["icegrad1"], dtype)

        self.conv_flux_weighter = PowerLawTiltWeighter(self.medianConvEnergy, params["CRDeltaGamma"])
        self.prompt_flux_weighter = PowerLawTiltWeighter(self.medianPromptEnergy, self.params["CRDeltaGamma"])
        self.astro_flux_weigter = PowerLawTiltWeighter(self.astroPivotEnergy, self.params["astroDeltaGamma"])


    def __call__(self, event): 
        ice_grads = self.ice_grad_0(event)*self.ice_grad_1(event)

        conventionalComponent = self.params["convNorm"]*self.aduw(event)*self.kluw(event)*(self.convPionFlux(event)+ self.params["piKRatio"]*self.convKaonFlux(event) 
                                +self.params["barrWP"]*self.barrWPComp(event) +self.params["barrWM"]*self.barrWMComp(event) +self.params["barrZP"]*self.barrZPComp(event)
                                +self.params["barrZM"]*self.barrZMComp(event) +self.params["barrYP"]*self.barrYPComp(event) +self.params["barrYM"]*self.barrYMComp(event)) \
                                *self.conv_flux_weighter(event) \
                                *self.convHoleIceWeighter(event)*self.convDOMEff(event)*ice_grads\
                                *self.conv_nu_att_weighter(event)
        promptComponent = self.params["promptNorm"]*self.promptFlux(event)*self.promptHoleIceWeighter(event)*self.promptDOMEff(event)\
                            *self.prompt_flux_weighter(event)*ice_grads\
                            *self.prompt_nu_att_weighter(event)
        astroComponent = self.params["astroNorm"]*self.astroMuFlux(event) \
                            *self.astroNuMuHoleIceWeighter(event)*self.astroNuMuDOMEff(event) \
                            *self.astro_flux_weigter(event)*ice_grads\
                            *self.astro_nu_att_weighter(event) \
                            *self.neuaneu_w(event)
        return conventionalComponent+promptComponent+astroComponent

    def _extract_edges_values(self, entry):
        edges = np.append(entry[0], entry[1][-1])
        values= entry[1]
        return edges, values

    def configure(self, params, dtype=float):
        self.params = params

        # some of these need to be reconfigured 

        self.neuaneu_w.configure(balance = params["NeutrinoAntineutrinoRatio"])

        self.aduw.configure(scale = params["zenithCorrection"])
        self.kluw.configure(scale = params["kaonLosses"])

        self.conv_nu_att_weighter.configure( scale_nu= params["nuxs"], scale_nubar=params["nubarxs"] )
        self.prompt_nu_att_weighter.configure( scale_nu= params["nuxs"], scale_nubar=params["nubarxs"] )
        self.astro_nu_att_weighter.configure( scale_nu= params["nuxs"], scale_nubar=params["nubarxs"] )

        self.convDOMEff.configure(scale_factor= params["domEfficiency"])
        self.promptDOMEff.configure(scale_factor= params["domEfficiency"])
        self.astroNuMuDOMEff.configure(scale_factor= params["domEfficiency"])

        self.convHoleIceWeighter.configure(scale_factor= params["holeiceForward"])
        self.promptHoleIceWeighter.configure(scale_factor= params["holeiceForward"])
        self.astroNuMuHoleIceWeighter.configure(scale_factor= params["holeiceForward"])

        self.ice_grad_0.configure( scale= params["icegrad0"])
        self.ice_grad_1.configure( scale= params["icegrad1"])

        self.conv_flux_weighter.configure(deltaIndex = params["CRDeltaGamma"])
        self.prompt_flux_weighter.configure(deltaIndex = params["CRDeltaGamma"])
        self.astro_flux_weigter.configure(deltaIndex = params["astroDeltaGamma"])
