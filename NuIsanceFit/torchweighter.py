"""
"""

from NuIsanceFit.param import ParamPoint
from NuIsanceFit.data import Data
from NuIsanceFit.event import Event
from NuIsanceFit.utils import get_lower
from NuIsanceFit.weighter import FluxComponent

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
    """
    def __init__(self, data:Data)->None:
        """
        This function will load in and cache anything needed to make the Weighters.
        Splines and stuff... 
        """
        self._data = data
        self._steering = data.steering

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

    @property 
    def simulation(self):
        return self._data.simulation


    def __call__(self, params:ParamPoint)->tensor:
        pass


class Weighter:
    """
    The weighters first weight all the events and store the event weights in tensors
    Then, when we normalize to a new parameter we scale our cached tensor by the related scaling factor 

    I noticed that just about all the weighters in golemfit are 
    """
    def __init__(self, parent: SimReWeighter, **kwargs):
        self._parent = parent

        shape = np.shape(self._parent.simulation.fill)
        self._weights = tensor(np.zeros(shape=shape))
        self._reweight()

        if not hasattr(self, "index"):
            raise NotImplementedError("You forgot to implement an \"index\" attribute!")

    def _reweight(self, **kwargs):
        shape = np.shape(self._parent.simulation.fill)
        for i_e in range(shape[0]): #energy
            for i_c in range(shape[1]): # costheta
                for i_a in range(shape[2]): # azimuth
                    for i_t in range(shape[3]): # topology 
                        for i_y in range(shape[4]): # year/time
                            for event in self._parent.simulation[i_e,i_c,i_a,i_t,i_y]:
                                self._weights[i_e,i_c,i_a,i_t,i_y] += self._weight(event)

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
        return self._spline((event.getLogPrimaryEnergy(), event.getPrimaryZenith())) #note: we already keep the event zenith in cosTh space 


class AtmosphericUncertaintyWeighter(SplineWeighter):
    def __init__(self, parent:SimReWeighter, spline:SplineTable, param_name:str):
        self.index = _ex_pp.valid_keys.index(param_name)
        SplineWeighter.__init__(self, parent, spline)

    def __call__(self, params:tensor)->tensor:
        return 1.0 + params[self.index]*self._weights

class AntiparticleWeighter(Weighter):
    """
    This is used to change the particle/antiparticle balance
    """
    def __init__(self, parent:SimReWeighter):
        self.index = _ex_pp.valid_keys.index("NeutrinoAntineutrinoRatio")
        Weighter.__init__(self, parent)

    def _weight(self, event: Event):
        return  1.0 if event.getPrimaryType()<0 else -1.0

    def __call__(self, params:tensor)->tensor:
        balance = params[self.index]*self._weights
        balance[balance<0] = 2.0 + balance
        return balance

class CachedValueWeighter(Weighter):
    def __init__(self, parent:SimReWeighter, key:str):

        self.index = -1 # param-independent, actually
        self._key = key
        Weighter.__init__(self, parent)
        
    def _weight(self, event: Event):
        return event.getCache()[self._key]

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
        gradbin = get_lower(event.getLogEnergy(), self._bin_edges)
        if gradbin == -1:
            return 1.0
        rel = (1.0+self._gradient[gradbin])
        return rel

    def __call__(self,params:tensor)->tensor:
        return self._weighs*params[self.index]
        
class PowerLawTiltWeighter(Weighter):
    def __init__(self, parent:SimReWeighter, medianEnergy:float, delta_key:str):

        self._medianEnergy = medianEnergy
        self.index = _ex_pp.valid_keys.index(delta_key)
        Weighter.__init__(self, parent)

        raise NotImplementedError("This one's actually tricky")
        
    def _weight(self, event: Event):
        return (event.getPrimaryEnergy()/self.medianEnergy)

    def __call__(self, params:tensor)->tensor:
        return self._weights