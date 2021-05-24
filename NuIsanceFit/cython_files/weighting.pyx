import cython

from event cimport Event
from event import EventCache
#from photospline import SplineTable
from libc.math cimport isnan

import photospline as ps

cdef class SplineTable:
    cdef object table
    def __cinit__(self, str filepath):
        self.table = ps.SplineTable(filepath)
    def __call__(self, tuple coords):
        return self.table(coords)

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
cpdef enum FluxComponent:
    atmConv = 0
    atmPrompt = 1
    atmMuon = 2
    diffuseAstro = 3
    diffuseAstro_e = 4
    diffuseAstro_mu = 5
    diffuseAstro_tau = 6
    diffuseAstroSec = 7
    gzk = 8

cdef class Weighter:
    def __cinit__(self, **kwargs):
        pass
    def __call__(self, Event event):
        pass

    def configure(self, list params):
        pass
    
cdef class PowerLawTiltWeighter(Weighter):
    def __cinit__(self, float medianEnergy, float deltaIndex):
        self.medianEnergy = medianEnergy
        self.deltaIndex = deltaIndex

    def configure(self, list params):
        self.deltaIndex = params[0]
    
    def __call__(self, Event event):
        pass

cdef class SimpleDataWeighter(Weighter):
    def __cinit__(self):
        pass
    def __call__(self, Event event):
        return 1. #event.cachedWeight.weight

cdef class AntiparticleWeighter(Weighter):
    cdef float balance
    def __cinit__(self, float balance):
        self.balance = balance
    def configure(self, list params):
        self.balance = params[0]
    def __call__(self, event):
        return( self.balance if event.primaryType<0 else 2-self.balance)

cdef class CachedValueWeighter(Weighter):
    """
    This is used to return a weight from an Event's cache! 
    """
    cdef str key
    def __cinit__(self,str key):
        if not (key in EventCache(1.0,1.0)):
            raise KeyError("Invalid Event Cahe key {}".format(key))

        self.key = key

    def __call__(self,Event event):
        return event.cachedWeight[self.key]

cdef class SplineWeighter(Weighter):
    cdef SplineTable _spline
    """
    Generic spliney Weighter. I'm using this as an intermediate so I don't have to keep writing out the spline check
    """
    def __cinit__(self, SplineTable spline, **kwargs):
        self._spline = spline

    def spline(self):
        return self._spline

    def __call__(self,Event event):
        return self.spline((event.logPrimaryEnergy, event.primaryZenith)) #note: we already keep the event zenith in cosTh space 

cdef class AtmosphericUncertaintyWeighter(SplineWeighter):
    cdef float _scale
    """
    This works for both the atmospheric density one _and_ the kaon one. 

    Seriously, look in GF. Those both are exactly identical functions. I don't get it 
    """
    def __cinit__(self, SplineTable spline, float scale):
        self._scale = scale 
    def configure(self,list params):
        self.scale = params[0]
    def __call__(self,Event event):
        cdef float value = SplineWeighter.__call__(self, event)
        value = 1.0 + value*self._scale
        return value

cdef class FluxCompWeighter(Weighter):
    cdef dict spline_map
    cdef FluxComponent fluxComp
    """
    This is the attenuation weighter. It has a (energy, zenith) spline of attenuation for each possible combination of "FluxComponent" and "primaryType"
    """
    def __cinit__(self, dict spline_map,FluxComponent fluxComp):
        self.fluxComp = fluxComp
        self.spline_map = spline_map

        if not self.fluxComp in spline_map:
            raise KeyError("Found no flux component {} in spline map, which has {}".format(self.fluxComp, spline_map.keys()))
        for key in self.spline_map[self.fluxComp]:
            if not isinstance(self.spline_map[self.fluxComp][key], SplineTable):
                raise TypeError("Found entry in the spline map, {}:{}, which is not a spline. It's a {}".format(self.fluxComp, key, type(self.spline_map[self.fluxComp][key])))

    def __call__(self, event):
        raise NotImplementedError("Use derived class")


cdef class TopoWeighter(FluxCompWeighter):
    cdef float scale_factor
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

    def __call__(self,Event event):
        if self.fluxComp == FluxComponent.atmConv:
            cache = event.cachedWeight["domEffConv"]
        elif self.fluxComp == FluxComponent.atmPrompt:
            cache = event.cachedWeight["domEffPrompt"]
        elif self.fluxComp == FluxComponent.diffuseAstro_mu:
            cache = event.cachedWeight["domEffAstro"]
        else:
            raise NotImplementedError("Should be unreachable")

        cdef str access_topo = ""
        if event._topology == 0:
            access_topo = "track"
        elif event._topology == 1:
            access_topo = "cascade"
        else:
            raise ValueError("Unsupported track topology for the topology weighter! {}".format(event._topology))

        cdef float correction = self.spline_map[self.fluxComp][access_topo]((event.logPrimaryEnergy, event.zenith  , self.scale_factor))
        
        if isnan(correction):
            # This is a cow? We just ignore cows? 
            return 0.0
        else: 
            return 10**(correction - cache) 

cdef class AttenuationWeighter(FluxCompWeighter):
    cdef float scale_nu
    cdef float scale_nubar
    def __cinit__(self,dict spline_map,FluxComponent fluxComp,float scale_nu,float scale_nubar):
        self.scale_nu = scale_nu
        self.scale_nubar = scale_nubar

    def configure(self,list params):
        self.scale_nu = params[0]
        self.scale_nubar = params[1]

    def __call__(self,Event event):
        cdef float scale

        if event.primaryType>0:
            scale = self.scale_nu
        else:
            scale = self.scale_nubar
        
        cdef float correction = self.spline_map[self.fluxComp][event.primaryType]((event.logPrimaryEnergy, event.primaryZenith, scale))

        if correction < 0:
            raise ValueError("Weighter returned negative weight {}!".format(correction))
        return correction
        

cdef class IceGradientWeighter(Weighter):
    cdef list _bin_edges
    cdef list _bin_centers
    cdef list _gradient  
    cdef float _scale 
    def __cinit__(self, list bin_edges,list gradient,float scale):
        self._bin_edges = bin_edges
        self._bin_centers = [(self._bin_edges[i+1]+self._bin_edges[i])/2. for i in range(len(self._bin_edges)-1)]
        self._gradient = gradient

        self._scale = scale

    def configure(self,list params):
        self.scale = params[0]

    def __call__(self, Event event):
        cdef int gradbin = get_lower(event.logEnergy, self._bin_centers)

        if gradbin == -1:
            return 1.0

        cdef float rel = (1.0+self._gradient[gradbin])*self._scale
        if rel<0:
            raise ValueError("Somehow got negative weight {}".format(rel))
        return rel
        