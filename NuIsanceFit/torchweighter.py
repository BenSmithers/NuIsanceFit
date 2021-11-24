"""
"""

from NuIsanceFit.param import ParamPoint
from NuIsanceFit.data import Data
from NuIsanceFit.event import Event
from NuIsanceFit.utils import get_lower
from NuIsanceFit.logger import Logger
from NuIsanceFit.weighter import FluxComponent

import torch
from torch import tensor
import numpy as np
from glob import glob
import os

from photospline import SplineTable

_ex_pp = ParamPoint()

def fill_fluxcomp_dict(folder:str):
    files = glob(folder+"/*.fits")
    if len(files)==0:
        raise IOError("Didn't find any splines in {}".format(folder))

    ret_dict = {}
    broken_up = []
    use_subc=False

    for item in files:
        filename_list = os.path.split(item)[1]

        # cut off the extension
        filename = ".".join(filename_list.split(".")[:-1])

        # access the different parts (nuMu, numuBar, etc...)
        broken_up = filename.split("_")

        subComponent = broken_up[-1] # TODO: Update this when we have dedicated particle types from nuSQuIDS or LeptonWeighter <--- this one  
        pid = 0
        if "mu" in subComponent.lower():
            pid = 13
        elif "e" in subComponent.lower():
            pid = 11
        elif "tau" in subComponent.lower():
            pid = 15
        else:
            use_subc = True
        if "nu" in subComponent.lower():
            pid+=1
        if "minus" in subComponent.lower() or "bar" in subComponent.lower():
            pid*=-1

        component1 = broken_up[-3]
        component2 = broken_up[-2]

        # we do this because of unpleasant naming conventions for the splines 
        fc = ""
        if component2=="mu" and component1=="diffuseAstro":
            fc = "_".join([component1,component2]) #eg, duffuseAstro_mu
        else:
            fc = component2#eg, atmConv

        try:
            fluxComponent = getattr(FluxComponent, fc)
        except AttributeError:
            raise ValueError("Failed?")
        
        # make sure we have the right structure to do this
        if fluxComponent not in ret_dict:
            ret_dict[fluxComponent] = {}

        # load in the spline 
        if use_subc:
            ret_dict[fluxComponent][subComponent] = SplineTable(item)
        else:
            ret_dict[fluxComponent][pid] = SplineTable(item)

    return ret_dict


class SimReWeighter:
    """
    This object can take a set of parameters and create a Meta-MetaWeighter that weights events according to that set of parameters 
    Note that the weights are only meaningful in a relative sense. We don't care about constants 

    For each Weighter, we keep a tensor of a proto-weight that we rescale. Sometimes the rescaling is linear, sometimes not. It speeds things up to keep it that way
        so we don't have to re-calculate splines and such. Tensors are the way to go! 

    Unfortunately, that means we need to keep a lot of stuff around in memory
        ~ 30 x 30 x (time bins) x (azimuth bins) x 2 x (max events) * 8 bytes per = could be... a lot 
        for each weighter! 

    We keep these in memory as sparse tensors, which should dramatically reduce the memory usage, but we'll see 
    """
    def __init__(self, data:Data)->None:
        """
        This function will load in and cache anything needed to make the Weighters.
        Splines and stuff... 
        """
        self._data = data
        self._steering = data.steering

        self.maxsize = 1
        # find the maxdim for when we make the sparse tensors
        shape = np.shape(self.simulation.fill)
        for i_e in range(shape[0]): #energy
            for i_c in range(shape[1]): # costheta
                for i_a in range(shape[2]): # azimuth
                    for i_t in range(shape[3]): # topology 
                        for i_y in range(shape[4]): # year/time
                            if len(self.simulation[i_e][i_c][i_a][i_t][i_y])>self.maxsize:
                                self.maxsize=len(self.simulation[i_e][i_c][i_a][i_t][i_y])

        # kinda silly, but now we need to find the "ones" sparse tensor that can be used by the weighters. 
        # we can't just add "1" to sparse tensors, we need to specifically tell it which entries have 1.... 
        ones = tensor(np.zeros(shape=list(np.shape(self.simulation.fill)) + [self.maxsize]))
        for i_e in range(shape[0]): #energy
            for i_c in range(shape[1]): # costheta
                for i_a in range(shape[2]): # azimuth
                    for i_t in range(shape[3]): # topology 
                        for i_y in range(shape[4]): # year/time
                            e_bin = self.simulation[i_e][i_c][i_a][i_t][i_y]
                            for i_bin in range(len(e_bin)):
                                ones[i_e,i_c,i_a,i_t,i_y,i_bin] = 1.0
        self.ones = ones.to_sparse()
    

        if self.maxsize>10000:
            Logger.Warn("Might run in to memory issues!")


        resources = self._steering["resources"]

        self.medianConvEnergy = self._steering["fixed_params"]["medianConvEnergy"]
        self.medianPromptEnergy = self._steering["fixed_params"]["medianPromptEnergy"]
        self.astroPivotEnergy = self._steering["fixed_params"]["astroPivotEnergy"]

        self._atmosphericDensityUncertaintySpline = SplineTable(resources["atmospheric_density_spline"])
        self._kaonLossesUncertaintySpline = SplineTable(resources["atmospheric_kaonlosses_spline"])

        self._attenuationSplineDict = fill_fluxcomp_dict(resources["attenuation_splines"])
        self._domSplines = fill_fluxcomp_dict(resources["dom_splines"])
        self._hqdomSplines = fill_fluxcomp_dict(resources["hq_dom_splines"])
        self._holeiceSplines = fill_fluxcomp_dict(resources["hole_ice_splines"])

        eff_grad_files = glob(resources["ice_gradients"]+"/Energy_Eff_Grad_*.txt")
        if len(eff_grad_files)==0:
            raise IOError("Found no files in {}".format(resources["ice_gradients"]))

        self._ice_grad_data = [np.loadtxt(fname, dtype=float, delimiter=" ").transpose() for fname in eff_grad_files]

        self.astroMuFlux = CachedValueWeighter(self,key="astroMuWeight")
        self.convPionFlux = CachedValueWeighter(self,key="convPionWeight")
        self.convKaonFlux = CachedValueWeighter(self,key="convKaonWeight")
        self.promptFlux = CachedValueWeighter(self,key="promptWeight")
        self.barrWPComp = CachedValueWeighter(self,key="barrModWP")
        self.barrWMComp = CachedValueWeighter(self,key="barrModWM")
        self.barrYPComp = CachedValueWeighter(self,key="barrModYP")
        self.barrYMComp = CachedValueWeighter(self,key="barrModYM")
        self.barrZPComp = CachedValueWeighter(self,key="barrModZP")
        self.barrZMComp = CachedValueWeighter(self,key="barrModZM")

        self.neuaneu_w = AntiparticleWeighter(self)
        self.aduw = AtmosphericUncertaintyWeighter(self, self._atmosphericDensityUncertaintySpline, "zenithCorrection")
        self.kluw = AtmosphericUncertaintyWeighter(self,  self._kaonLossesUncertaintySpline,"kaonLosses")


        edges, values = self._extract_edges_values(self._ice_grad_data[0])
        self.ice_grad_0 = IceGradientWeighter(self, bin_edges=edges,gradient= values,scale_key="icegrad0")

        edges, values = self._extract_edges_values(self._ice_grad_data[1])
        self.ice_grad_1 = IceGradientWeighter(self, bin_edges=edges,gradient= values,scale_key= "icegrad1")

        self.conv_flux_weighter = PowerLawTiltWeighter(self, self.medianConvEnergy, delta_key="CRDeltaGamma")
        self.prompt_flux_weighter = PowerLawTiltWeighter(self, medianEnergy=self.medianPromptEnergy, delta_key="CRDeltaGamma")
        self.astro_flux_weigter = PowerLawTiltWeighter(self, medianEnergy=self.astroPivotEnergy, delta_key="astroDeltaGamma")

        self.convNorm_index = _ex_pp.valid_keys.index("convNorm")
        self.promptNorm_index = _ex_pp.valid_keys.index("promptNorm")
        self.astroNorm_index = _ex_pp.valid_keys.index("astroNorm")
        self.pik_ratio_index = _ex_pp.valid_keys.index("piKRatio")

        self.barrWP_index = _ex_pp.valid_keys.index("barrWP")   
        self.barrZP_index = _ex_pp.valid_keys.index("barrZP")   
        self.barrYP_index = _ex_pp.valid_keys.index("barrYP")   
        self.barrWM_index = _ex_pp.valid_keys.index("barrWM")   
        self.barrZM_index = _ex_pp.valid_keys.index("barrZM")   
        self.barrYM_index = _ex_pp.valid_keys.index("barrYM")   


    @property 
    def simulation(self):
        return self._data.simulation


    def __call__(self, params:tensor)->tensor:
        ice_grads = self.ice_grad_0(params)*self.ice_grad_1(params)

        convComp = params[self.convNorm_index]*self.aduw(params)*self.kluw(params)*(self.convPionFlux(params) + params[self.pik_ratio_index]*self.convKaonFlux(params) 
                    + params[self.barrWP_index]*self.barrWPComp(params) + params[self.barrYP_index]*self.barrYPComp(params) + params[self.barrZP_index]*self.barrZPComp(params)
                    + params[self.barrWM_index]*self.barrWMComp(params) + params[self.barrYM_index]*self.barrYMComp(params) + params[self.barrZM_index]*self.barrZMComp(params)) \
                    *self.conv_flux_weighter(params) \
                    *ice_grads

        promptComp = params[self.promptNorm_index]*self.promptFlux(params)*self.prompt_flux_weighter(params)
                    
        astroComp  = params[self.astroNorm_index]*self.astroMuFlux(params)*self.astro_flux_weigter(params)*ice_grads \
                    *self.neuaneu_w(params)

        return convComp+promptComp+astroComp

    def _extract_edges_values(self, entry:np.ndarray )->tuple:
        edges = np.append(entry[0], entry[1][-1])
        values = entry[-1]
        return( list(edges), list(values) )

class Weighter:
    """
    The weighters first weight all the events and store the event weights in tensors
    Then, when we normalize to a new parameter we scale our cached tensor by the related scaling factor 

    I noticed that just about all the weighters in golemfit are 
    """
    def __init__(self, parent: SimReWeighter, **kwargs):
        self._parent = parent

        shape = list(np.shape(  self._parent.simulation.fill )) + [self._parent.maxsize]
        Logger.Log("Making Weighter with shape {}".format(shape))

        self._weights = tensor(np.zeros(shape=shape))
        self._reweight(**kwargs)
        self._weights = self._weights.to_sparse()

        if not hasattr(self, "index"):
            raise NotImplementedError("You forgot to implement an \"index\" attribute!")

    def _reweight(self, **kwargs):
        shape = np.shape(self._parent.simulation.fill)
        for i_e in range(shape[0]): #energy
            for i_c in range(shape[1]): # costheta
                for i_a in range(shape[2]): # azimuth
                    for i_t in range(shape[3]): # topology 
                        for i_y in range(shape[4]): # year/time
                            ebin = self._parent.simulation[i_e][i_c][i_a][i_t][i_y]
                            for i_bin in range(len( ebin )):
                                self._weights[i_e,i_c,i_a,i_t,i_y,i_bin] = self._weight(ebin[i_bin], **kwargs)

    def _weight(self, event: Event):
        raise NotImplementedError()

    def __call__(self, params:tensor) -> tensor:


        raise NotImplementedError()



class SplineWeighter(Weighter):
    """
    Generic spliney Weighter. I'm using this as an intermediate so I don't have to keep writing out the spline check
    """
    def __init__(self, parent:SimReWeighter, spline:SplineTable, **kwargs):
        self._spline = spline

        Weighter.__init__(self, parent)

    def _weight(self, event: Event):
        return self._spline((event.logPrimaryEnergy, event.primaryZenith)) #note: we already keep the event zenith in cosTh space 


class AtmosphericUncertaintyWeighter(SplineWeighter):
    def __init__(self, parent:SimReWeighter, spline:SplineTable, param_name:str):
        self.index = _ex_pp.valid_keys.index(param_name)
        SplineWeighter.__init__(self, parent, spline)

    def __call__(self, params:tensor)->tensor:
        return self._parent.ones + params[self.index]*self._weights

class AntiparticleWeighter(Weighter):
    """
    This is used to change the particle/antiparticle balance

    the balance is centered around 0.0, a positive balance shifts it towards the antimatter 
    """
    def __init__(self, parent:SimReWeighter):
        self.index = _ex_pp.valid_keys.index("NeutrinoAntineutrinoRatio")
        Weighter.__init__(self, parent)

    def _weight(self, event: Event):
        return  1.0 if event.primaryType<0 else -1.0

    def __call__(self, params:tensor)->tensor:
        balance = params[self.index]*self._weights
        return balance + self._parent.ones

class CachedValueWeighter(Weighter):
    def __init__(self, parent:SimReWeighter, key:str):

        self.index = -1 # param-independent, actually
        self._key = key
        Weighter.__init__(self, parent)
        
    def _weight(self, event: Event):
        return event.cachedWeight[self._key]

    def __call__(self, params:tensor)->tensor:
        return self._weights
        
class IceGradientWeighter(Weighter):
    def __init__(self, parent: SimReWeighter, bin_edges: list, gradient:list, scale_key:str):
        self._bin_edges = bin_edges
        self._bin_centers = [(self._bin_edges[i+1]+self._bin_edges[i])/2. for i in range(len(self._bin_edges)-1)]
        self._gradient = gradient

        self.index = _ex_pp.valid_keys.index(scale_key)

        Weighter.__init__(self, parent)
    
    def _weight(self, event:Event):
        gradbin = get_lower(event.logEnergy, self._bin_edges)
        if gradbin == -1:
            return 1.0
        rel = (1.0+self._gradient[gradbin])
        return rel

    def __call__(self,params:tensor)->tensor:
        return self._weights*params[self.index]
        
class PowerLawTiltWeighter(Weighter):
    def __init__(self, parent:SimReWeighter, medianEnergy:float, delta_key:str):

        self._medianEnergy = medianEnergy
        self.index = _ex_pp.valid_keys.index(delta_key)
        Weighter.__init__(self, parent)

    def _weight(self, event: Event):
        return (event.primaryEnergy/self._medianEnergy)

    def __call__(self, params:tensor)->tensor:
        if params[self.index]==0.:
            return self._parent.ones #this would make it dense! Not good 
        else:
            return self._weights**float(params[self.index]) # I really hope this doesn't kill the weighting 

class FluxCompWeighter(Weighter):
    """
    This is the attenuation weighter. It has a (energy, zenith) spline of attenuation for each possible combination of "FluxComponent" and "primaryType"
    """
    def __init__(self, parent:SimReWeighter, spline_map: dict, fluxComp:FluxComponent):
        self.fluxComp = fluxComp
        self.spline_map = spline_map

        if not self.fluxComp in spline_map:
            raise KeyError("Found no flux component {} in spline map, which has {}".format(self.fluxComp, spline_map.keys()))
        for key in self.spline_map[self.fluxComp]:
            if not isinstance(self.spline_map[self.fluxComp][key], SplineTable):
                raise TypeError("Found entry in the spline map, {}:{}, which is not a spline. It's a {}".format(self.fluxComp, key, type(self.spline_map[self.fluxComp][key])))

        Weighter.__init__(self, parent)


class TopoWeighter(FluxCompWeighter):
    """
    This kind of weighter reweights the events according to their topologies. 
    This is used by the 
        DOMEfficiency Weighters
        HQ DOM Efficiency Weighters
        the Hole Ice Weighters
    """
    def __init__(self, parent:SimReWeighter, spline_map: dict, fluxComp:FluxComponent, key_scale:str):
        self.index = _ex_pp.valid_keys.index(key_scale)
        if not (self.fluxComp == FluxComponent.atmConv or self.fluxComp == FluxComponent.atmPrompt or self.fluxComp == FluxComponent.diffuseAstro_mu):
            raise ValueError("Weighter configured with disallowed flux component: {}".format(self.fluxComp))
        FluxCompWeighter.__init__( self, parent, spline_map, fluxComp)

    def _weight(self, event: Event):
        if self.fluxComp == FluxComponent.atmConv:
            cache = event.cachedWeight["domEffConv"]
        elif self.fluxComp == FluxComponent.atmPrompt:
            cache = event.cachedWeight["domEffPrompt"]
        elif self.fluxComp == FluxComponent.diffuseAstro_mu:
            cache = event.cachedWeight["domEffAstro"]
        else:
            raise NotImplementedError("Should be unreachable")

        if event.getTopology() == 0:
            access_topo = "track"
        elif event.getTopology() == 1:
            access_topo = "cascade"
        else:
            raise ValueError("Unsupported track topology for the topology weighter! {}".format(event.getTopology()))

        if access_topo not in self.spline_map[self.fluxComp]:
            return 1.0
        
        raise NotImplementedError("Well well well")
        correction = self.spline_map[self.fluxComp][access_topo]((event.getLogPrimaryEnergy(), event.getPrimaryZenith()  , 1.0 ))
        
        if np.isnan(correction):
            # This is a cow? We just ignore cows? 
            return 0.0
        else: 
            return 10**(correction - cache)