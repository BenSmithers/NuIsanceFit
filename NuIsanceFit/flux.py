"""
Here, we will put some wrappers for fluxes
"""

from logger import Logger
from NuIsanceFit import steering
from NuIsanceFit.event import Event

class Flux:
    """
    Base class for Fluxes.
    We can have different versions of these for fluxes of different sources
    """
    def __init__(self, **kwargs):
        Logger.Fatal("Need to use derived class, not abstract base class", NotImplementedError)
    def __call__(self, event:Event)->float:
        """
        Returns differential flux in /[GeV s sr cm^2]^-1 
        """
        Logger.Fatal("NEed to use derived class, not abstract base class", NotImplementedError)
        

class Atmo_Flux(Flux):
    def __init__(self, **kwargs):
        pass

    def __call__(self, event:Event)->float:
        pass