from Event import Event 
from param import params as global_params

"""
Here we design and implement classes to Weight events

The 
"""


class Weighter:
    """
    A Default type from which all the weighters will be made. 
    """
    def __init__(self, dtype=float):
        if not isinstance(dtype, type):
            raise TypeError("Arg 'dtype' must be {}, got {}".format(type, type(dtype))) # weird...

        self._dtype = dtype

    @property
    def dtype(self):
        return(self._dtype)

    def __call__(self, event):
        """
        Calculate the weight of the event
        """
        if not isinstance(event, Event):
            raise TypeError("Expected {}, got {}".format(Event, type(event)))

        return( self.dtype() )

    def __add__(self, other):
        """
        Here, we combine two weighters into one super-weighter
        This returns another Wighter object that evaluates the sum of the parent weighters' calculated weights 
        """
        if not isinstance(other, Weighter):
            raise TypeError("Expected {}, got {}".format(Weighter, type(other)))

        # create default event 
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


class WeighterMaker:
    """
    This object can take a set of parameters and create a Meta-MetaWeighter that weights events according to that set of parameters 

    Note that the weights are only meaningful in a relative sense. We don't care about constants 
    """
    def __init__(self):
        pass

    def __call__(self, params): 

        conventionalComponent   = powerLawTiltWeighter(params["convNorm"]*1e5, -2.5 + params["CRDeltaGamma"])
        promptComponent         = powerLawTiltWeighter(params["promptNorm"]*1e5, -2.5 )
        astroComponent          = powerLawTiltWeighter(params["astroNorm"]*1e5, -2.0 + params["astroDeltaGamma"])

        return( concentionalComponent + promptComponent + astroComponent )

