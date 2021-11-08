import cython

from event cimport Event
from event import EventCache
#from photospline import SplineTable
from libc.math cimport isnan
from libc.math cimport pow as cpow

#import photospline as ps
#
cimport numpy as np
import numpy as np

import os
from glob import glob

from NuIsanceFit.param import ParamPoint

from photospline import SplineTable


cdef int get_lower(float x, list domain):
    """
    Cython implementation of the get_lower function. Ues a binary search to find the bin to the left of "x" in "domain"
    """
    if (len(domain)<=1):
        raise ValueError("Get_lower function only works on domains of length>1. This is length {}".format(len(domain)))

    if (x<domain[0] or x>domain[-1]):
        return -1
    
    cdef int min_abs=0
    cdef int max_abs=len(domain)-1
    cdef int lower_bin = int((max_abs-min_abs)/2)
    cdef int upper_bin = lower_bin+1

    while (not (domain[lower_bin]<=x and domain[upper_bin]>=x)):
        if (x<domain[lower_bin]):
            max_abs=lower_bin
        
        if (x>domain[upper_bin]):
            min_abs=upper_bin
        

        lower_bin = min_abs + int((max_abs-min_abs)/2)
        upper_bin = lower_bin + 1
    
    return lower_bin



"""
A simple enum to keep track of the different flux components 
"""


cdef class Weighter:
    def __cinit__(self, **kwargs):
        pass

    cpdef float evalEvt(self, Event event):
        raise NotImplementedError()

    def configure(self, list params):
        pass
    
cdef class PowerLawTiltWeighter(Weighter):
    def __cinit__(self, float medianEnergy, float deltaIndex):
        self.medianEnergy = medianEnergy
        self.deltaIndex = deltaIndex

    def configure(self, list params):
        self.deltaIndex = params[0]
    
    cpdef float evalEvt(self, Event event):
        if event.getPrimaryEnergy()<=0.0:
            raise ValueError("Cannot weight event with negative energy: {}".format(event.getPrimaryEnergy()))
        return (event.getPrimaryEnergy()/self.medianEnergy)**self.deltaIndex

cdef class SimpleDataWeighter(Weighter):
    def __cinit__(self, **kwargs):
        pass
    def __call__(self, Event event):
        return 1. #event.cachedWeight.weight

cdef class AntiparticleWeighter(Weighter):
    def __cinit__(self, float balance):
        self.balance = balance
    def configure(self, list params):
        self.balance = params[0]
    cpdef float evalEvt(self, Event event):
        return( self.balance if event.getPrimaryType()<0 else 2-self.balance)

cdef class CachedValueWeighter(Weighter):
    """
    This is used to return a weight from an Event's cache! 
    """
    def __cinit__(self,str key):
        if not (key in EventCache(1.0,1.0)):
            raise KeyError("Invalid Event Cahe key {}".format(key))

        self.key = key

    cpdef float evalEvt(self,Event event):
        return event.getCache()[self.key]

cdef class SplineWeighter(Weighter):
    """
    Generic spliney Weighter. I'm using this as an intermediate so I don't have to keep writing out the spline check
    """
    def __cinit__(self, object spline, **kwargs):
        self._spline = spline

    cpdef float evalEvt(self,Event event):
        return self._spline((event.getLogPrimaryEnergy(), event.getPrimaryZenith())) #note: we already keep the event zenith in cosTh space 

cdef class AtmosphericUncertaintyWeighter(SplineWeighter):
    """
    This works for both the atmospheric density one _and_ the kaon one. 

    Seriously, look in GF. Those both are exactly identical functions. I don't get it 
    """
    def __cinit__(self, object spline, float scale, **kwargs):
        self._scale = scale 
    def configure(self,list params):
        self._scale = params[0]
    cpdef float evalEvt(self,Event event):
        return 1.0 + self._scale*SplineWeighter.evalEvt(self, event)

cdef class FluxCompWeighter(Weighter):
    """
    This is the attenuation weighter. It has a (energy, zenith) spline of attenuation for each possible combination of "FluxComponent" and "primaryType"
    """
    def __cinit__(self, dict spline_map,FluxComponent fluxComp, **kwargs):
        self.fluxComp = fluxComp
        self.spline_map = spline_map

        if not self.fluxComp in spline_map:
            raise KeyError("Found no flux component {} in spline map, which has {}".format(self.fluxComp, spline_map.keys()))
        for key in self.spline_map[self.fluxComp]:
            if not isinstance(self.spline_map[self.fluxComp][key], SplineTable):
                raise TypeError("Found entry in the spline map, {}:{}, which is not a spline. It's a {}".format(self.fluxComp, key, type(self.spline_map[self.fluxComp][key])))

    cpdef float evalEvt(self, Event event):
        raise NotImplementedError("Use derived class")


cdef class TopoWeighter(FluxCompWeighter):
    """
    This kind of weighter reweights the events according to their topologies. 
    This is used by the 
        DOMEfficiency Weighters
        HQ DOM Efficiency Weighters
        the Hole Ice Weighters
    """
    def __cinit__(self, dict spline_map, FluxComponent fluxComp, float scale_factor):        
        self.scale_factor = scale_factor
        if not (self.fluxComp == FluxComponent.atmConv or self.fluxComp == FluxComponent.atmPrompt or self.fluxComp == FluxComponent.diffuseAstro_mu):
            raise ValueError("Weighter configured with disallowed flux component: {}".format(self.fluxComp))
    def configure(self, list params):
        self.scale_factor = params[0]

    cpdef float evalEvt(self,Event event):
        if self.fluxComp == FluxComponent.atmConv:
            cache = event.getCache()["domEffConv"]
        elif self.fluxComp == FluxComponent.atmPrompt:
            cache = event.getCache()["domEffPrompt"]
        elif self.fluxComp == FluxComponent.diffuseAstro_mu:
            cache = event.getCache()["domEffAstro"]
        else:
            raise NotImplementedError("Should be unreachable")

        cdef str access_topo = ""
        if event.getTopology() == 0:
            access_topo = "track"
        elif event.getTopology() == 1:
            access_topo = "cascade"
        else:
            raise ValueError("Unsupported track topology for the topology weighter! {}".format(event.getTopology()))

        cdef float correction = self.spline_map[self.fluxComp][access_topo]((event.getLogPrimaryEnergy(), event.getPrimaryZenith()  , self.scale_factor))
        
        if isnan(correction):
            # This is a cow? We just ignore cows? 
            return 0.0
        else: 
            return 10**(correction - cache)

cdef class AttenuationWeighter(FluxCompWeighter):
    def __cinit__(self,dict spline_map,FluxComponent fluxComp,float scale_nu,float scale_nubar):
        self.scale_nu = scale_nu
        self.scale_nubar = scale_nubar

    def configure(self,list params):
        self.scale_nu = params[0]
        self.scale_nubar = params[1]

    cpdef float evalEvt(self,Event event):
        cdef float correction
        if event._primaryType>0:
            correction = self.spline_map[self.fluxComp][event.getPrimaryType()]((event.getLogPrimaryEnergy(), event.getPrimaryZenith(), self.scale_nu))
        else:
            correction = self.spline_map[self.fluxComp][event.getPrimaryType()]((event.getLogPrimaryEnergy(), event.getPrimaryZenith(), self.scale_nubar))

        if correction < 0:
            raise ValueError("Weighter returned negative weight {}!".format(correction))
        return correction
        

cdef class IceGradientWeighter(Weighter):
    def __cinit__(self, list bin_edges,list gradient,float scale):
        self._bin_edges = bin_edges
        self._bin_centers = [(self._bin_edges[i+1]+self._bin_edges[i])/2. for i in range(len(self._bin_edges)-1)]
        self._gradient = gradient

        self._scale = scale

    def configure(self,list params):
        self._scale = params[0]

    cpdef float evalEvt(self, Event event):
        cdef int gradbin = get_lower(event.getLogEnergy(), self._bin_edges)

        if gradbin == -1:
            return 1.0

        cdef float rel = (1.0+self._gradient[gradbin])*self._scale
        if rel<0:
            raise ValueError("Somehow got negative weight {}".format(rel))
        return rel
        
def fill_fluxcomp_dict(str folder):
    cdef list files = glob(folder+"/*.fits")
    if len(files)==0:
        raise IOError("Didn't find any splines in {}".format(folder))

    cdef dict ret_dict = {}
    cdef list broken_up
    cdef str filename, subComponent, component1, component2, fc, filename_list
    cdef int pid
    cdef FluxComponent fluxComponent
    cdef bint use_subc=False
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
            pid = 15
        elif "tau" in subComponent.lower():
            pid = 17
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

cdef class SimReWeighter:
    """
    This object can take a set of parameters and create a Meta-MetaWeighter that weights events according to that set of parameters 

    Note that the weights are only meaningful in a relative sense. We don't care about constants 
    """
    def __cinit__(self, dict steering):
        """
        This function will load in and cache anything needed to make the Weighters.
        Splines and stuff... 
        """
        cdef dict resources = steering["resources"]

        self.medianConvEnergy = steering["fixed_params"]["medianConvEnergy"]
        self.medianPromptEnergy = steering["fixed_params"]["medianPromptEnergy"]
        self.astroPivotEnergy = steering["fixed_params"]["astroPivotEnergy"]

        _atmosphericDensityUncertaintySpline = SplineTable(resources["atmospheric_density_spline"])
        _kaonLossesUncertaintySpline = SplineTable(resources["atmospheric_kaonlosses_spline"])

        self._attenuationSplineDict = fill_fluxcomp_dict(resources["attenuation_splines"])
        self._domSplines = fill_fluxcomp_dict(resources["dom_splines"])
        self._hqdomSplines = fill_fluxcomp_dict(resources["hq_dom_splines"])

        self._holeiceSplines = fill_fluxcomp_dict(resources["hole_ice_splines"])

        # Now the ice gradients?
        cdef list eff_grad_files = glob(resources["ice_gradients"]+"/Energy_Eff_Grad_*.txt")
        if len(eff_grad_files)==0:
            raise IOError("Found no files in {}".format(resources["ice_gradients"]))
        # in these 
        #   0 is left side of bin
        #   1 is right side of bin
        #   2 is value in the bin
        self._ice_grad_data = [np.loadtxt(fname, dtype=float, delimiter=" ").transpose() for fname in eff_grad_files]

        cdef dict params = ParamPoint().as_dict()
        self.params = params

        # The caches! 
        self.astroMuFlux = CachedValueWeighter(key="astroMuWeight")
        self.convPionFlux = CachedValueWeighter(key="convPionWeight")
        self.convKaonFlux = CachedValueWeighter(key="convKaonWeight")
        self.promptFlux = CachedValueWeighter(key="promptWeight")
        self.barrWPComp = CachedValueWeighter(key="barrModWP")
        self.barrWMComp = CachedValueWeighter(key="barrModWM")
        self.barrYPComp = CachedValueWeighter(key="barrModYP")
        self.barrYMComp = CachedValueWeighter(key="barrModYM")
        self.barrZPComp = CachedValueWeighter(key="barrModZP")
        self.barrZMComp = CachedValueWeighter(key="barrModZM")

        self.neuaneu_w = AntiparticleWeighter(balance=params["NeutrinoAntineutrinoRatio"])

        self.aduw = AtmosphericUncertaintyWeighter(spline = _atmosphericDensityUncertaintySpline, scale= params["zenithCorrection"])
        self.kluw = AtmosphericUncertaintyWeighter(spline = _kaonLossesUncertaintySpline,scale= params["kaonLosses"])

        self.conv_nu_att_weighter = AttenuationWeighter(spline_map = self._attenuationSplineDict, fluxComp=FluxComponent.atmConv, scale_nu=params["nuxs"], scale_nubar=params["nubarxs"])
        self.prompt_nu_att_weighter = AttenuationWeighter(spline_map=self._attenuationSplineDict, fluxComp=FluxComponent.atmPrompt, scale_nu= params["nuxs"],scale_nubar= params["nubarxs"])
        self.astro_nu_att_weighter = AttenuationWeighter(spline_map=self._attenuationSplineDict, fluxComp=FluxComponent.diffuseAstro_mu, scale_nu=params["nuxs"],scale_nubar= params["nubarxs"])

        self.convDOMEff = TopoWeighter(spline_map=self._domSplines,fluxComp= FluxComponent.atmConv,scale_factor= params["domEfficiency"])
        self.promptDOMEff = TopoWeighter(spline_map=self._domSplines,fluxComp= FluxComponent.atmPrompt,scale_factor= params["domEfficiency"])
        self.astroNuMuDOMEff = TopoWeighter(spline_map=self._domSplines,fluxComp= FluxComponent.diffuseAstro_mu,scale_factor= params["domEfficiency"])

        # these aren't actually used. 
        #hqconvDOMEff = TopoWeighter(self._hqdomSplines, FluxComponent.atmConv, params["hqdomEffficiency"],dtype)
        #hqpromptDOMEff = TopoWeighter(self._hqdomSplines, FluxComponent.atmPrompt, params["hqdomEffficiency"],dtype)

        self.convHoleIceWeighter = TopoWeighter(spline_map=self._holeiceSplines,fluxComp= FluxComponent.atmConv,scale_factor= params["holeiceForward"])
        self.promptHoleIceWeighter = TopoWeighter(spline_map=self._holeiceSplines, fluxComp=FluxComponent.atmPrompt,scale_factor= params["holeiceForward"])
        self.astroNuMuHoleIceWeighter = TopoWeighter(spline_map=self._holeiceSplines,fluxComp= FluxComponent.diffuseAstro_mu,scale_factor= params["holeiceForward"])

        edges, values = self._extract_edges_values(self._ice_grad_data[0])
        self.ice_grad_0 = IceGradientWeighter(bin_edges=edges,gradient= values,scale= params["icegrad0"])

        edges, values = self._extract_edges_values(self._ice_grad_data[1])
        self.ice_grad_1 = IceGradientWeighter(bin_edges=edges,gradient= values,scale= params["icegrad1"])

        self.conv_flux_weighter = PowerLawTiltWeighter(medianEnergy=self.medianConvEnergy, deltaIndex=params["CRDeltaGamma"])
        self.prompt_flux_weighter = PowerLawTiltWeighter(medianEnergy=self.medianPromptEnergy, deltaIndex=params["CRDeltaGamma"])
        self.astro_flux_weigter = PowerLawTiltWeighter(medianEnergy=self.astroPivotEnergy, deltaIndex=params["astroDeltaGamma"])

    def _extract_edges_values(self, np.ndarray entry):
        cdef np.ndarray edges = np.append(entry[0], entry[1][-1])
        cdef values = entry[-1]
        return( list(edges), list(values) )

    def configure(self, dict params):
        self.params = params

        # some of these need to be reconfigured 

        self.neuaneu_w.configure([params["NeutrinoAntineutrinoRatio"]])

        self.aduw.configure([params["zenithCorrection"]])
        self.kluw.configure([params["kaonLosses"]])

        self.conv_nu_att_weighter.configure([ params["nuxs"], params["nubarxs"]])
        self.prompt_nu_att_weighter.configure([ params["nuxs"], params["nubarxs"]])
        self.astro_nu_att_weighter.configure([ params["nuxs"], params["nubarxs"]])

        self.convDOMEff.configure([params["domEfficiency"]])
        self.promptDOMEff.configure([ params["domEfficiency"]])
        self.astroNuMuDOMEff.configure([params["domEfficiency"]])

        self.convHoleIceWeighter.configure([params["holeiceForward"]])
        self.promptHoleIceWeighter.configure([params["holeiceForward"]])
        self.astroNuMuHoleIceWeighter.configure([ params["holeiceForward"]])

        self.ice_grad_0.configure([params["icegrad0"]])
        self.ice_grad_1.configure([params["icegrad1"]])

        self.conv_flux_weighter.configure([params["CRDeltaGamma"]])
        self.prompt_flux_weighter.configure([params["CRDeltaGamma"]])
        self.astro_flux_weigter.configure([params["astroDeltaGamma"]])

    def __call__(self, Event event):
        cdef float ice_grads = self.ice_grad_0.evalEvt(event)*self.ice_grad_1.evalEvt(event)

        cdef float conventionalComponent = self.params["convNorm"]*self.aduw.evalEvt(event)*self.kluw.evalEvt(event)*(self.convPionFlux.evalEvt(event)+ self.params["piKRatio"]*self.convKaonFlux.evalEvt(event) 
                                +self.params["barrWP"]*self.barrWPComp.evalEvt(event) +self.params["barrWM"]*self.barrWMComp.evalEvt(event) +self.params["barrZP"]*self.barrZPComp.evalEvt(event)
                                +self.params["barrZM"]*self.barrZMComp.evalEvt(event) +self.params["barrYP"]*self.barrYPComp.evalEvt(event) +self.params["barrYM"]*self.barrYMComp.evalEvt(event)) \
                                *self.conv_flux_weighter.evalEvt(event) \
                                *self.convHoleIceWeighter.evalEvt(event)*self.convDOMEff.evalEvt(event)*ice_grads\
                                *self.conv_nu_att_weighter.evalEvt(event)
        cdef float promptComponent = self.params["promptNorm"]*self.promptFlux.evalEvt(event)*self.promptHoleIceWeighter.evalEvt(event)*self.promptDOMEff.evalEvt(event)\
                            *self.prompt_flux_weighter.evalEvt(event)*ice_grads\
                            *self.prompt_nu_att_weighter.evalEvt(event)
        cdef float astroComponent = self.params["astroNorm"]*self.astroMuFlux.evalEvt(event) \
                            *self.astroNuMuHoleIceWeighter.evalEvt(event)*self.astroNuMuDOMEff.evalEvt(event) \
                            *self.astro_flux_weigter.evalEvt(event)*ice_grads\
                            *self.astro_nu_att_weighter.evalEvt(event) \
                            *self.neuaneu_w.evalEvt(event)
        return conventionalComponent+promptComponent+astroComponent