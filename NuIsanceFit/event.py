import enum
from logger import Logger

"""
Here, we describe the Event classes and methods for interacting with the events 
"""


class EventCache:
    """
    This object is used to cache various properties for the Event class 
    """
    def __init__(self):
        # quantities
        self.livetime = 0.0
        self.weight = 0.0

        # flux cache
        self.convPionWeight = 0.0
        self.convKaonWeight = 0.0
        self.convWeight = 0.0
        self.promptWeight = 0.0
        self.astroMuWeight = 0.0
        
        # barr cache
        self.barrModWP = 0.0
        self.barrModWM = 0.0
        self.barrModYP = 0.0
        self.barrModYM = 0.0
        self.barrModZP = 0.0
        self.barrModZM = 0.0

        # holeice cache
        self.holeIceConv = 0.0
        self.holeIcePrompt = 0.0
        self.holeIceAstro = 0.0

        #dom efficiency cache
        self.domEffConv = 0.0
        self.domEffPrompt = 0.0
        self.domEffAstro = 0.0


class Event:

    def __init__(self):
        self._primaryEnergy = 0.0
        self._primaryAzimuth = 0.0
        self._primaryZenith = 0.0
        self._totalColumnDepth = 0.0
        self._intX = 0
        self._intY = 0

        self._oneWeight = 0.0

        self._num_events = 0

        self._energy = 0.0
        self._zenith = 0.0
        self._azimuth = 0.0

        self._sample = 0 # SampleTag.CASCADE
        
        self._topology = 0

        self._year = 0

        self._cachedweight = EventCache()

    # =================== Getters and Setters =======================
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
    @property
    def intX(self):
        return self._intX
    @property
    def intY(self):
        return self._intY
    @property
    def oneWeight(self):
        return self._oneWeight
    @property
    def num_events(self):
        return self._num_events
    @property
    def energy(self):
        return self._energy
    @property 
    def zenith(self):
        return self._zenith
    @property
    def azimuth(self):
        return self._azimuth
    @property 
    def topology(self):
        return self._topology
    @property
    def year(self):
        return self._year 
    @property 
    def cachedWeight(self):
        return self._cachedweight

    # ========================= Utilities ============================
    def _get_meta_weight(self):
        return self._cachedweight.convWeight + self._cachedweight.promptWeight + self._cachedweight.astroMuWeight

    def __add__(self, other):
        meta_weight = self._get_meta_weight
        if np.isnan(meta_weight) or meta_weight<0:
            Logger.Fatal("Found invalid metaweight! {}".format(meta_weight))
    def __eq__(self, other):
        raise NotImplementedError()
    def __mul__(self, other):
        raise NotImplementedError()

def buildEventFromData(self, dataEvent):
    raise NotImplementedError("Still need to implement this")
