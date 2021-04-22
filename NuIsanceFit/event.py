
from logger import Logger

"""
Here, we describe the Event classes and methods for interacting with the events 
"""

class EventCache:
    """
    This object is used to cache various properties for the Event class 
    """
    def __init__(self):

        self.livetime = 0.0
        self.weight = 0.0
        self.convPionWeight = 0.0
        self.convKaonWeight = 0.0
        self.convWeight = 0.0
        self.promptWeight = 0.0
        self.astroMuWeight = 0.0
        



class Event:

    def __init__(self):
        self._primaryEnergy = 0.0
        self._primaryAzimuth = 0.0
        self._primaryZenith = 0.0
        self._totalColumnDepth = 0.0
        self._intX = 0
        self._intY = 0

        self._oneWeight = 0.0
        self._number_of_generated_mc_events = 0.0

        self._num_events = 0

        self._energy = 0.0
        self._zenith = 0.0
        self._azimuth = 0.0

        self._sample = None
        
        self._topology = 0

        self._year = 0

        self._cachedweight = EventCache()

    @property
    def primaryEnergy(self):
        return self._primaryEnergy

    @property
    def primaryAzimuth(self):
        return self._primaryAzimuth

    @property
    def primaryZenith(self):
        return self._primaryZenith

    @property
    def totalColumnDepth(self):
        return self._totalColumnDepth

    def _get_meta_weight(self):
        return self._cachedweight.convWeight + self._cachedweight.promptWeight + self._cachedweight.astroMuWeight

    def __add__(self, other):
        meta_weight = self._get_meta_weight
        if np.isnan(meta_weight) or meta_weight<0:
            Logger.Fatal("Found invalid metaweight! {}".format(meta_weight))
    def __eq__(self, other):
        pass
    def __mul__(self, other):
        pass

def buildEventFromData(self, dataEvent):
    raise NotImplementedError("Still need to implement this")
