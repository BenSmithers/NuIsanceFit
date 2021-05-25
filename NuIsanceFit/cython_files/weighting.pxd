cdef class Weighter:
    pass

cdef class SimpleDataWeighter(Weighter):
    pass

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

cdef class SplineTable:
    cdef readonly object table

cdef class PowerLawTiltWeighter(Weighter):
    cdef readonly float medianEnergy
    cdef readonly float deltaIndex

cdef class AntiparticleWeighter(Weighter):
    cdef readonly float balance

cdef class CachedValueWeighter(Weighter):
    """
    This is used to return a weight from an Event's cache! 
    """
    cdef readonly str key

cdef class SplineWeighter(Weighter):
    cdef readonly SplineTable _spline

cdef class AtmosphericUncertaintyWeighter(SplineWeighter):
    cdef readonly float _scale

cdef class FluxCompWeighter(Weighter):
    cdef readonly dict spline_map
    cdef readonly FluxComponent fluxComp


cdef class TopoWeighter(FluxCompWeighter):
    cdef readonly float scale_factor

cdef class AttenuationWeighter(FluxCompWeighter):
    cdef readonly float scale_nu
    cdef readonly float scale_nubar

cdef class IceGradientWeighter(Weighter):
    cdef readonly list _bin_edges
    cdef readonly list _bin_centers
    cdef readonly list _gradient  
    cdef readonly float _scale 