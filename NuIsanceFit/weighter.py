from NuIsanceFit.event import Event, EventCache
from NuIsanceFit.param import params as global_params

from NuIsanceFit.logger import Logger 

from math import log10
import photospline as ps
from enum import Enum
from numbers import Number
"""
Here we design and implement classes to Weight events

The 
"""
class FluxComponent(Enum):
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
    
        dtype = type(other.dtype ()+ self.dtype())
        
        # make a little meta weighter object. It does weighting! 
        class metaWeighter(Weighter):
            def __init__(self_meta, dtype):
                Weighter.__init__(self_meta,dtype)
            def __call__(self_meta,event):
                return(self(event)+other(event))

        return(metaWeighter(dtype))

    def __mul__(self, other):
        """
        Define what happens when we multiply weighters together 

        This produces a meta-Weighter using two other weighters. This weighter weights a given event with the parent weighters, then multiplies the weights together and returns the product 
        """
        if not isinstance(other, Weighter):
            raise TypeError("Expected {}, got {}".format(Weighter, type(other)))

        dtype = type(other.dtype()* self.dtype())

        class metaWeighter(Weighter):
            def __init__(self_meta, dtype):
                Weighter.__init__(self_meta, dtype)
            def __call__(self_meta, event):
                return(self(event)*other(event))

        return(metaWeighter(dtype))
    
    def __div__(self, other):
        """
        Exactly the same as the multiplication, but now it's dividing 
        """
        if not isinstance(other, Weighter):
            raise TypeError("Expected {}, got {}".format(Weighter, type(other)))

        dtype = type(other.dtype() / self.dtype())

        class metaWeighter(Weighter):
            def __init__(self_meta, dtype):
                Weighter.__init__(self_meta, dtype)
            def __call__(self_meta, event):
                return(self(event)/other(event))

        return(metaWeighter(dtype))

# ======================= Implemented Weighters ==================================

class powerLawTiltWeighter(Weighter):
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

class brokenPowerlawTiltWeighter(powerLawTiltWeighter):
    def __init__(self, medianEnergy, deltaIndex1, deltaIndex2):
        powerLawTiltWeighter.__init__(self, medianEnergy, deltaIndex1)

        self.deltaIndex2 = deltaIndex2 

    def __call__(self, event):
        Weighter.__call__(self, event)
        return 5 


class simpleDataWeighter(Weighter):
    def __init__(self, dtype=float):
        Weighter.__init__(self, dtype)
    def __call__(self, event):
        return 1. #event.cachedWeight.weight

class antiparticleWeighter(Weighter):
    def __init__(self, balance):
        Weighter.__init__(self, float)
        self.balance = balance
    def __call__(self, event):
        return( self.balance if event.primaryType<0 else 2-self.balance)

class cachedValueWeighter(Weighter):
    """
    Wait the whole concept of this weighter seems stupid. It's just a number. That's... like... all it is. 

    So we just give it a cache object and it returns that?
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
            Logger.Fatal("Need Spline... not a {}".format(type(spline)))
        self._spline = spline
        self._zero = dtype()

    @property
    def spline(self):
        return self._spline

    def __call__(self,event):
        coordinates = (log10(event.primaryEnergy), event.primaryZenith)
        correction = self.spline(coordinates)
        if correction<self._zero:
            Logger.Fatal("Made a negative weight: {}".format(correction))
        return self.dtype(correction)

class DOMEffWeighter(SplineWeighter):
    def __init__(self, domSpline, deltaDomEff, flux, dtype):
        SplineWeighter.__init__(self, domSpline, dtype)
        if not isinstance(deltaDomEff, Number):
            Logger.Fatal("deltaDomEff should be number, not {}".format(type(deltaDomEff)))
        if not isinstance(flux, FluxComponent):
            Logger.Fatal("Need {}, not {}".format(FluxComponent, type(flux)))

        self._deltaDomEff = deltaDomEff

class atmosphericUncertaintyWeighter(SplineWeighter):
    """
    This works for both the atmospheric density one _and_ the kaon one. 

    Seriously, look in GF. Those both are exactly identical functions. I don't get it 
    """
    def __init__(self, spline, scale, dtype=float):
        SplineWeighter.__init__(self, spline, dtype)
        if not isinstance(scale, dtype):
            Logger.Fatal("Expected {}, not {}".format(dtype, type(scale)))
        self._scale = scale 
    def __call__(self, event):
        value = SplineWeighter.__call__(self, event)
        value = 1.0 + value*self._scale
        return self.dtype(value)

class AttenuationWeighter(Weighter):
    def __init__(self, flux, xs_scaling, cross_sections, secondaries_included, dtype):
        Weighter.__init__(self, dtype)
        if not isinstance(xs_scaling, dtype):
            Logger.Fatal("xs_scaling must be {}, not {}".format(dtype, type(xs_scaling)))
        if not isinstance(flux, FluxComponent):
            Logger.Fatal("Give me a {} flux, not a {}".format(FluxComponent, type(flux)))
            


"""
TODO: 
    ??? cachedValueWeighter
        DONE antiparticleWeighter
    DOMEffWeighter
    holeIceWeighter
        DONE atmosphericDensityUncertainty weighter
        DONE kaonLossesUncertainty weighter 
    icegradient weighter 
    attenuationWeighter

This will require some spline stuff 

Should probably also restructure these weighters to be built from a params object that just reads in what it needs - rather than a list of parameters! 
"""

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
        self._atmosphericDensityUncertaintySpline = ps.SplineTable(resources["atmospheric_density_spline"])
        self._kaonLossesUncertaintySpline = ps.SplineTable(resources["atmospheric_kaonlosses_spline"])

    def __call__(self, params): 

        Logger.Trace("Creating new metaWeighter")

        Logger.Trace("Conv")
        conventionalComponent   = powerLawTiltWeighter(params["convNorm"]*1e5, -2.5 + params["CRDeltaGamma"])
        Logger.Trace("Prompt")
        promptComponent         = powerLawTiltWeighter(params["promptNorm"]*1e5, -2.5 )
        Logger.Trace("Astro")
        astroComponent          = powerLawTiltWeighter(params["astroNorm"]*1e5, -2.0 + params["astroDeltaGamma"])

        return( conventionalComponent + promptComponent + astroComponent )

