from event cimport Event 

cpdef enum FluxComponent:
    atmConv = 0,
    atmPrompt = 1,
    atmMuon = 2,
    diffuseAstro = 3,
    diffuseAstro_e = 4,
    diffuseAstro_mu = 5,
    diffuseAstro_tau = 6,
    diffuseAstroSec = 7,
    gzk = 8

"""
cdef extern from "<photospline>" namespace "photospline":
    cdef cppclass splinetable[Alloc=*]:
        ctypedef Alloc allocator_type
        splinetable() except + 
        splinetable(str) except +
        float operator()(tuple)
"""

# from photospline import splinetable

cdef class Weighter:
    cpdef float evalEvt(self, Event event)

cdef class SimpleDataWeighter(Weighter):
    pass

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
    cdef readonly object _spline

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

cdef class SimWeighter:
    cdef readonly float medianConvEnergy
    cdef readonly float medianPromptEnergy 
    cdef readonly float astroPivotEnergy

    #cdef readonly SplineTable _atmosphericDensityUncertaintySpline 
    #cdef readonly SplineTable _kaonLossesUncertaintySpline 

    cdef readonly dict _attenuationSplineDict 
    cdef readonly dict _domSplines 
    cdef readonly dict _hqdomSplines
    cdef readonly dict _holeiceSplines 

    cdef readonly list _ice_grad_data

    cdef readonly dict params

    cdef readonly CachedValueWeighter astroMuFlux
    cdef readonly CachedValueWeighter convPionFlux
    cdef readonly CachedValueWeighter convKaonFlux
    cdef readonly CachedValueWeighter promptFlux
    cdef readonly CachedValueWeighter barrWPComp
    cdef readonly CachedValueWeighter barrWMComp
    cdef readonly CachedValueWeighter barrYPComp
    cdef readonly CachedValueWeighter barrYMComp
    cdef readonly CachedValueWeighter barrZPComp 
    cdef readonly CachedValueWeighter barrZMComp 

    cdef readonly AntiparticleWeighter neuaneu_w 

    cdef readonly AtmosphericUncertaintyWeighter aduw
    cdef readonly AtmosphericUncertaintyWeighter kluw

    cdef readonly AttenuationWeighter conv_nu_att_weighter
    cdef readonly AttenuationWeighter prompt_nu_att_weighter
    cdef readonly AttenuationWeighter astro_nu_att_weighter 

    cdef readonly TopoWeighter convDOMEff 
    cdef readonly TopoWeighter promptDOMEff 
    cdef readonly TopoWeighter astroNuMuDOMEff

    cdef readonly TopoWeighter convHoleIceWeighter
    cdef readonly TopoWeighter promptHoleIceWeighter 
    cdef readonly TopoWeighter astroNuMuHoleIceWeighter 

    cdef readonly IceGradientWeighter ice_grad_0 
    cdef readonly IceGradientWeighter ice_grad_1

    cdef readonly PowerLawTiltWeighter conv_flux_weighter
    cdef readonly PowerLawTiltWeighter prompt_flux_weighter
    cdef readonly PowerLawTiltWeighter astro_flux_weigter

