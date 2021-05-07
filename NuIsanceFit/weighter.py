from NuIsanceFit.event import Event, EventCache
from NuIsanceFit.param import params as global_params
from NuIsanceFit.logger import Logger 
from NuIsanceFit.histogram import get_loc

from math import log10
import photospline as ps
from enum import Enum
from numbers import Number
import os 
import numpy as np
from glob import glob # finding the attenuation splines
"""
Here we design and implement classes to Weight events

Later on, we might need to implement distinct treatments for derivavites.Consider...
    from scipy.misc import derivative
This is only 1D differentiation, but you can make it work with some lambda functions to evaluate a 1D derivative of an arbitrary function (at some point)
"""

# when you cast these as strings, they look like
#    FluxComponent.atmConv 
class FluxComponent(Enum):
    """
    A simple enum to keep track of the different flux components 
    """
    atmConv = 0
    atmPrompt = 1
    atmMuon = 2
    diffuseAstro = 3
    diffuseAstro_e = 4
    diffuseAstro_mu = 5
    diffuseAstro_tau = 6
    diffuseAstroSec = 7
    GZK = 8


class Weighter:
    """
    A Default type from which all the weighters will be made. 

    Structured such that Weighters can weight things of any type 
    """
    def __init__(self, dtype=float):
        #if not isinstance(dtype, type):
        #    Logger.Fatal("Arg 'dtype' must be {}, got {}".format(type, type(dtype)), TypeError) # weird...

        self._dtype = dtype

    @property
    def dtype(self):
        return(self._dtype)

    def __call__(self, event):
        """
        Calculate the weight of the event
        """
        if not isinstance(event, Event):
            Logger.Fatal("Expected {}, got {}".format(Event, type(event)), TypeError)

        return( self.dtype() )

    def __add__(self, other):
        """
        Here, we combine two weighters into one meta-weighter
        This returns another Wighter object that evaluates the sum of the parent weighters' calculated weights 
        """
        if not isinstance(other, Weighter):
            Logger.Fatal("Expected {}, got {}".format(Weighter, type(other)), TypeError)

        Logger.Trace("Combining two weighters")

        return MetaWeighter(self, other, "+")

    def __sub__(self, other):
        """
        Same as addition, but now subtraction
        """
        if not isinstance(other, Weighter):
            raise TypeError("Expected {}, got {}".format(Weighter, type(other)))
            
        return MetaWeighter(self, other, "-")

    def __mul__(self, other):
        """
        Define what happens when we multiply weighters together 

        This produces a meta-Weighter using two other weighters. This weighter weights a given event with the parent weighters, then multiplies the weights together and returns the product 
        """
        return MetaWeighter(self, other,"*")
    
    def __div__(self, other):
        """
        Exactly the same as the multiplication, but now it's dividing 
        """
        return MetaWeighter(self, other, "-")

class MetaWeighter(Weighter):
    """
    This is a class used to combine two weighter objects together according to some operation. 
    """
    def __init__(self, parent1, parent2, op):
        """
        Make the metaweighter using the two parent weighters, and some operation we use to combine them 

        The supported operations are: +, -, /, and *
        The operation should be passed as a string (LENGTH 1) 
        """
        if not isinstance(parent1, Weighter):
            Logger.Fatal("I need a {}, not a {}".format(Weighter, type(parent1)), TypeError)
        if op!="+" and op!="-" and op!="*" and op!="/":
            Logger.Fatal("Not sure how to combine weighters with {}".format(op))

        # parent2 could be a Weighter, a number, or a derivative. We need to know whether or not to call it! 
        if isinstance(parent2, Weighter):
            self._call_parent_2 = True
        else:
            self._call_parent_2 = False

        self._parent1 = parent1
        self._parent2 = parent2
        self._op = op
        
        # this way we can ascertain the data we get by combining these things in this way
        self._dtype = type(self._combine(parent1.dtype() , parent2.dtype() if self._call_parent_2 else self._parent2 ))

    def __call__(self, event):
        return self._combine(self._parent1(event), self._parent2(event) if self._call_parent_2 else self._parent2)
    
    def _combine(self, obj1, obj2):
        """
        Use the configured operation to combined the two objects we are passed 
        """
        if self._op=="+":
            return obj1 + obj2
        elif self._op=="-":
            return obj1- obj2
        elif self._op=="*":
            return obj1 * obj2
        elif self._op=="/":
            return obj1 / obj2
        else:
            Logger.Fatal("Reached the Unreachable")


# ======================= Implemented Weighters ==================================

class PowerLawTiltWeighter(Weighter):
    def __init__(self, medianEnergy, deltaIndex):
        Weighter.__init__(self, type(deltaIndex))

        self.medianEnergy = medianEnergy
        self.deltaIndex = deltaIndex

    def __call__(self, event):
        Weighter.__call__(self, event)
        if event.primaryEnergy<=0:
            Logger.Warn("Primary en {}; zenith {}; azimuth {}. {}".format(event.primaryEnergy, event.primaryZenith, event.primaryAzimuth, event.is_mc))
            Logger.Fatal("Cannot weight zero-energy event...")
        return( pow( event.primaryEnergy/self.medianEnergy, self.deltaIndex) )

class BrokenPowerlawTiltWeighter(PowerLawTiltWeighter):
    def __init__(self, medianEnergy, deltaIndex1, deltaIndex2):
        PowerLawTiltWeighter.__init__(self, medianEnergy, deltaIndex1)

        self.deltaIndex2 = deltaIndex2 

    def __call__(self, event):
        Weighter.__call__(self, event)
        return 5 


class SimpleDataWeighter(Weighter):
    def __init__(self, dtype=float):
        Weighter.__init__(self, dtype)
    def __call__(self, event):
        return 1. #event.cachedWeight.weight

class AntiparticleWeighter(Weighter):
    def __init__(self, balance):
        Weighter.__init__(self, float)
        self.balance = balance
    def __call__(self, event):
        return( self.balance if event.primaryType<0 else 2-self.balance)

class CachedValueWeighter(Weighter):
    """
    I'm pretty sure this will just save a number and return it 
    """
    def __init__(self, cache, key):
        Weighter.__init__(self, float)
        if not isinstance(cache, EventCache):
            Logger.Fatal("Cannot create cache of type {}".format(type(cache)))
        if not isinstance(key, str):
            Logger.Fatal("Access key must be {}, not {}".format(str, type(key)))
        self.cache = cache 
        self.key = key

    def __call__(self, event):
        return getattr(self.cache, self.key)

class SplineWeighter(Weighter):
    """
    Generic spliney Weighter. I'm using this as an intermediate so I don't have to keep writing out the spline check
    """
    def __init__(self, spline, dtype):
        Weighter.__init__(dtype)
        if not isinstance(spline, ps.SplineTable):
            Logger.Fatal("Need Spline... not a {}".format(type(spline)), TypeError)
        self._spline = spline
        self._zero = dtype()

    @property
    def spline(self):
        return self._spline

    def __call__(self,event):
        coordinates = (log10(event.primaryEnergy), event.primaryZenith) #note: we already keep the event zenith in cosTh space 
        correction = self.spline(coordinates)
        if correction<self._zero:
            Logger.Fatal("Made a negative weight: {}".format(correction), ValueError)
        return self.dtype(correction)


class AtmosphericUncertaintyWeighter(SplineWeighter):
    """
    This works for both the atmospheric density one _and_ the kaon one. 

    Seriously, look in GF. Those both are exactly identical functions. I don't get it 
    """
    def __init__(self, spline, scale, dtype=float):
        SplineWeighter.__init__(self, spline, dtype)
        if not isinstance(scale, dtype):
            Logger.Fatal("Expected {}, not {}".format(dtype, type(scale)), TypeError)
        self._scale = scale 
    def __call__(self, event):
        value = SplineWeighter.__call__(self, event)
        value = 1.0 + value*self._scale
        return self.dtype(value)

class FluxCompWeighter(Weighter):
    """
    This is the attenuation weighter. It has a (energy, zenith) spline of attenuation for each possible combination of "FluxComponent" and "primaryType"
    """
    def __init__(self, spline_map, fluxComp, dtype):
        Weighter.__init__(self, dtype)
        if not isinstance(fluxComp, FluxComponent):
            Logger.Fatal("Expected {}, got {}".format(FluxComponent, type(fluxComp)), TypeError)
        if not isinstance(spline_map, dict):
            Logger.Fatal("Expected {} for spline_map, got {}".format(dict, type(spline_map)), TypeError)

        self.fluxComp = fluxComp
        self.spline_map = spline_map

        self._present_component = self.fluxComp in spline_map

        if not self._present_component:
            Logger.Fatal("Found no flux component {} in spline map".format(self.fluxComp), KeyError)
        for key in self.spline_map[self.fluxComp]:
            if not isinstance(self.spline_map[self.fluxComp][key], ps.SplineTable):
                Logger.Fatal("Found entry in the spline map, {}:{}, which is not a spline. It's a {}".format(self.fluxComp, key, type(self.spline_map[self.fluxComp][key])), TypeError)

        


class DOMEffWeighter(FluxCompWeighter):
    def __init__(self, spline_map, fluxComp, domEfficiency, dtype):
        FluxCompWeighter.__init__(self, spline_map, fluxComp, dtype)
        if not isinstance(domEfficiency, dtype):
            Logger.Fatal("Expected {}, got {}, for dom efficiency".format(dtype, type(domEfficiency)))
        
        self.domEfficiency = domEfficiency

        if not (self.fluxComp == FluxComponent.atmConv or self.fluxComp == FluxComponent.atmPrompt or self.fluxComp == FluxComponent.diffuseAstro_mu):
            Logger.Fatal("Weighter configured with disallowed flux component: {}".format(self.fluxComp), ValueError)

    def __call__(self, event):
        if self.fluxComp == FluxComponent.atmConv:
            cache = event.EventCache.domEffConv
        elif self.fluxComp == FluxComponent.atmPrompt:
            cache = event.EventCache.domEffPrompt
        elif self.fluxComp == FluxComponent.diffuseAstro_mu:
            cache = event.EventCache.domEffAstro
        else:
            Logger.Fatal("Should be unreachable")

        access_topo = ""
        if event._topology == 0:
            access_topo = "track"
        elif event._topology == 1:
            access_topo = "cascade"
        else:
            Logger.Fatal("Unsupported track topology for the DOM Efficiency weighter! {}".format(event._topology), ValueError)

        coordinates = (log10(event.primaryEnergy), event.zenith  , self.domEfficiency)
        correction = self.spline_map[self.fluxComp][access_topo](coordinates)
        if correction < 0:
            Logger.Fatal("Weighter returned negative weight {}!".format(correction),ValueError)

        if correction==0.0 or np.isnan(correction):
            # This is a cow? We just ignore cows? 
            return 0.0
        else: 
            return pow(10., correction - cache) 

class AttenuationWeighter(FluxCompWeighter):
    def __init__(self, spline_map, fluxComp, scale_nu, scale_nubar, dtype):
        FluxCompWeighter.__init__(self, spline_map, fluxComp, dtype)
        if not isinstance(scale_nu, dtype):
            Logger.Fatal("Expected {} for scale_nu, got {}".format(dtype, type(scale_nu)), TypeError)
        if not isinstance(scale_nubar, dtype):
            Logger.Fatal("Expected {} for scale_nubar, got {}".format(dtype, type(scale_nubar)), TypeError)

        self.scale_nu = scale_nu
        self.scale_nubar = scale_nubar

    def __call__(self, event):
        Weighter.__call__(self, event)
        if event.primaryType not in self.spline_map[self.fluxComp]:
            return self.dtype(1.0)
        if event.primaryType>0:
            scale = self.scale_nu
        else:
            scale = self.scale_nubar
        
        coordinates = (log10(event.primaryEnergy), event.primaryZenith, scale)
        correction = self.spline_map[self.fluxComp][event.primaryType](coordinates)
        if correction < 0:
            Logger.Fatal("Weighter returned negative weight {}!".format(correction),ValueError)
        return correction
        

class IceGradientWeighter(Weighter):
    def __init__(self, bin_edges, gradient, scale, dtype):
        Weighter.__init__(self, dtype)
        if not isinstance(scale, dtype):
            Logger.Fatal("Expected scale of type {}, got {}".format(dtype, type(scale)), ValueError)

        self._bin_edges = bin_edges
        self._gradient = gradient

        self._scale = scale

    def __call__(self, event):
        Weighter.__call__(self, event)

        index = get_loc(log10(event.energy), self._bin_edges)
        

"""
TODO: 
    ??? cachedValueWeighter
        DONE antiparticleWeighter
        DONE DOMEffWeighter
    holeIceWeighter
        DONE atmosphericDensityUncertainty weighter
        DONE kaonLossesUncertainty weighter 
    icegradient weighter 
        DONE attenuationWeighter

This will require some spline stuff 

Should probably also restructure these weighters to be built from a params object that just reads in what it needs - rather than a list of parameters! 
"""

def fill_fluxcomp_dict(folder):
    files = glob(folder+"/*fits.")
    ret_dict = {}
    for item in files:
        Logger.Trace("    {}".format(item))
        dirname, filename = os.path.split(item)

        # cut off the extension
        filename = ".".join(filename.split(".")[:-1])

        # access the different parts (nuMu, numuBar, etc...)
        broken_up = filename.split("_")
        # we do this because of unpleasant naming conventions for the splines 
        if broken_up[3]=="diffuseAstro":
            fc = "_".join(broken_up[3:5])  #eg, duffuseAstro_mu
        else:
            fc = broken_up[3]  #eg, atmConv

        fluxComponent = getattr(FluxComponent, fc)
        subComponent = broken_up[-1] # TODO: Update this when we have dedicated particle types from nuSQuIDS or LeptonWeighter <--- this one  
        
        # make sure we have the right structure to do this
        if fluxComponent not in ret_dict:
            ret_dict[fluxComponent] = {}

        # load in the spline 
        ret_dict[fluxComponent][subComponent] = ps.SplineTable(item)

    return ret_dict


# ========================= Weighter Maker Code =====================================
class WeighterMaker:
    """
    This object can take a set of parameters and create a Meta-MetaWeighter that weights events according to that set of parameters 

    Note that the weights are only meaningful in a relative sense. We don't care about constants 
    """
    def __init__(self, steering):
        """
        This function will load in and cache anything needed to make the Weighters.
        Splines and stuff... 
        """
        resources = steering["resources"]
        Logger.Log("Loading in splines for Weighting")
        Logger.Trace("    {}".format(resources["atmospheric_density_spline"]))
        self._atmosphericDensityUncertaintySpline = ps.SplineTable(resources["atmospheric_density_spline"])
        Logger.Trace("    {}".format(resources["atmospheric_kaonlosses_spline"]))
        self._kaonLossesUncertaintySpline = ps.SplineTable(resources["atmospheric_kaonlosses_spline"])

        self._attenuationSplineDict = fill_fluxcomp_dict(resources["attenuation_splines"])
        self._domSplines = fill_fluxcomp_dict(resources["dom_splines"])
        self._hqdomSplines = fill_fluxcomp_dict(resources["hq_dom_splines"])

        # Now the ice gradients?
        eff_grad_files = glob(resources["ice_gradients"]+"/Energy_Eff_Grad_*.txt")
        # in these 
        #   0 is left side of bin
        #   1 is right side of bin
        #   2 is value in the bin
        self._ice_grad_data = [np.loadtxt(fname, dtype=float, delimiter=" ").transpose() for fname in eff_grad_files]

    def _extract_edges_values(self, entry):
        edges = entry[0].append(entry[1][-1])
        values= entry[1]
        return edges, values


    def __call__(self, params, dtype=float): 

        Logger.Trace("Creating new metaWeighter")

        Logger.Trace("Conv")
        conventionalComponent   = PowerLawTiltWeighter(params["convNorm"]*1e5, -2.5 + params["CRDeltaGamma"])
        Logger.Trace("Prompt")
        promptComponent         = PowerLawTiltWeighter(params["promptNorm"]*1e5, -2.5 )
        Logger.Trace("Astro")
        astroComponent          = PowerLawTiltWeighter(params["astroNorm"]*1e5, -2.0 + params["astroDeltaGamma"])

        neuaneu_t = AntiparticleWeighter(params["NeutrinoAntineutrinoRatio"])

        aduw = AtmosphericUncertaintyWeighter(self._atmosphericDensityUncertaintySpline, params["zenithCorrection"])
        kluw = AtmosphericUncertaintyWeighter(self._kaonLossesUncertaintySpline, params["kaonLosses"])

        conv_nu_att_weighter = AttenuationWeighter(self._attenuationSplineDict, FluxComponent.atmConv, params["nuxs"], params["nubarxs"], dtype)
        prompt_nu_att_weighter = AttenuationWeighter(self._attenuationSplineDict, FluxComponent.atmPrompt, params["nuxs"], params["nubarxs"], dtype)
        astro_nu_att_weighter = AttenuationWeighter(self._attenuationSplineDict, FluxComponent.diffuseAstro_mu, params["nuxs"], params["nubarxs"], dtype)

        convDOMEff = DOMEffWeighter(self._domSplines, FluxComponent.atmConv, params["domEfficiency"], dtype)
        promptDOMEff = DOMEffWeighter(self._domSplines, FluxComponent.atmPrompt, params["domEfficiency"], dtype)
        astroNuMuDOMEff = DOMEffWeighter(self._domSplines, FluxComponent.diffuseAstro_mu, params["domEfficiency"], dtype)

        hqconvDOMEff = DOMEffWeighter(self._hqdomSplines, FluxComponent.atmConv, params["hqdomEffficiency"],dtype)
        hqpromptDOMEff = DOMEffWeighter(self._hqdomSplines, FluxComponent.atmPrompt, params["hqdomEffficiency"],dtype)


        return( conventionalComponent + promptComponent + astroComponent )

