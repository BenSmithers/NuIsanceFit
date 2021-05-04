"""
Here, we will put some wrappers for fluxes
"""

from logger import Logger
from event import Event 

class Flux:
    """
    Base class for Fluxes.
    We can have different versions of these for fluxes of different sources
    """
    def __init__(self, **kwargs):
        Logger.Fatal("Need to use derived class, not abstract base class", NotImplementedError)
    def __call__(self, event):
        if not isinstance(event, Event):
            Logger.Fatal("Can only weight objects of type {}, not {}".format(Event, type(event)), TypeError)
        
