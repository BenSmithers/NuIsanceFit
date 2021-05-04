from NuIsanceFit.event import Event 
from NuIsanceFit.param import params as global_params

from NuIsanceFit.logger import Logger 

"""
Here we design and implement classes to Weight events

The 
"""

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
    
        # create default event. 
        ev = Event()
        dtype = type(other(ev) + self(ev))
        
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

        ev = Event()
        dtype = type(other(ev)*self(ev))

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

        ev = Event()
        dtype = type(other(ev)/self(ev))

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

        return( pow( event.primaryEnergy/self.medianEnergy, -1*self.deltaIndex) )

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

"""
TODO: 
    cachedValueWeighter
    antiparticleWeighter
    DOMEffWeighter
    holeIceWeighter
    atmosphericDensityUncertainty weighter
    kaonLossesUncertainty weighter 
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
    def __init__(self):
        """
        This function will load in and cache anything needed to make the Weighters.
        Splines and stuff... 

        nothing right now, but soon! 
        """
        pass

    def __call__(self, params): 

        Logger.Trace("Creating new metaWeighter")

        Logger.Trace("Conv")
        conventionalComponent   = powerLawTiltWeighter(params["convNorm"]*1e5, -2.5 + params["CRDeltaGamma"])
        Logger.Trace("Prompt")
        promptComponent         = powerLawTiltWeighter(params["promptNorm"]*1e5, -2.5 )
        Logger.Trace("Astro")
        astroComponent          = powerLawTiltWeighter(params["astroNorm"]*1e5, -2.0 + params["astroDeltaGamma"])

        return( conventionalComponent + promptComponent + astroComponent )

