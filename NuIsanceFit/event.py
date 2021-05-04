import enum
from .logger import Logger
from numbers import Number 

import numpy as np
from math import pi

"""
Here, we describe the Event classes and methods for interacting with the events 
"""

class EventCache:
    """
    This object is used to cache various properties used during the Weighting 
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
    """
    This defines 'events' in the detector and all the possible parameters it can have. These are really the fundamental 
    object type that is used in the weighting and likelihood calculation. 

    We bin these in the histograms (well, in the histograms' eventBins)  - see histogram.py
    """
    def __init__(self):
        """
        Initializes a null event. Decided against putting all these parameters in the constructor, seems like bad form to have a dozen arguments here .

        Considering later adding a **kwargs and parsing those individually. (check for name, pass to setBLAH function)
        """
        self._primaryEnergy = 0.0
        self._primaryAzimuth = 0.0
        self._primaryZenith = 0.0
        self._totalColumnDepth = 0.0
        self._intX = 0
        self._intY = 0

        self._oneWeight = 0.0

        self._num_events = 1

        self._energy = 0.0
        self._zenith = 0.0
        self._azimuth = 0.0

        self._sample = 0 # SampleTag.CASCADE
       
        # 0 is track, 1 is cascade
        self._topology = 0

        self._year = 0

        self._cachedweight = EventCache()

    # =================== Getters and Setters =======================
    """
    Check that the object's type is right, and then check that it's a physical value. 

    All of these work the same exact way, and just set some parameter. 
    """
    def setPrimaryEnergy(self, energy):
        if not isinstance(energy, Number):
            Logger.Fatal("Cannot set energy to {}".format(type(energy)))
        if energy<0:
            Logger.Fatal("Cannot set negative energy {}".format(energy))
        self._primaryEnergy = energy
    def setPrimaryAzimuth(self, azimuth):
        if not isinstance(azimuth, Number):
            Logger.Fatal("Cannot set azimuth to {}".format(type(azimuth)))
        if (azimuth<0) or (azimuth>2*pi):
            Logger.Fatal("Invalid azimuth: {}. Is this degrees? Should be radians!".format(azimuth))
        self._primaryAzimuth = azimuth
    def setPrimaryZenith(self, zenith):
        if not isinstance(zenith, Number):
            Logger.Fatal("Cannot set zenith to {}".format(type(zenith)))
        if (zenith<-1) or (zenith>1):
            Logger.Fatal("Invalid zenith {}. Is this in degrees? It should be radians!".format(zenith))
        self._primaryZenith =zenith
    def setTotalColumnDepth(self, totalColumnDepth):
        if not isinstance(totalColumnDepth, Number):
            Logger.Fatal("TCD needs to be a number, not {}".format(type(totalColumnDepth)))
        if totalColumnDepth<0:
            Logger.Fatal("TCD cannot be negative".format(totalColumnDepth))
        self._totalColumnDepth = totalColumnDepth
    def setIntX(self, intX):
        if not isinstance(intX, Number):
            Logger.Fatal("Bjorken X should be a number, not {}".format(type(intX)))
        if intX<0:
            Logger.Fatal("Cannot have negative Bjorken X {}".format(intX))
        self._intX = intX
    def setIntY(self, intY):
        if not isinstance(intY, Number):
            Logger.Fatal("Bjorken Y should be a number, not {}".format(type(intY)))
        if intY<0:
            Logger.Fatal("Cannot have negative Bjorken Y {}".format(intY))
        self._intY = intY
    def setOneWeight(self, oneWeight):
        if not isinstance(oneWeight, Number):
            Logger.Fatal("OneWeight should be a number, not {}".format(type(oneWeight)))
        if oneWeight<0:
            Logger.Fatal("Cannot have negative oneWeight {}".format(oneWeight))
        self._oneWeight = oneWeight
    def setNumEvents(self, num_events):
        if not isinstance(num_events, int):
            Logger.Fatal("NumEvents should be an int, not {}".format(type(num_events)))
        if num_events<0:
            Logger.Fatal("Cannot have negative number of events {}".format(num_events))
        self._num_events = num_events
    def setEnergy(self, energy):
        if not isinstance(energy, Number):
            Logger.Fatal("Cannot set energy to {}".format(type(energy)))
        if energy<0:
            Logger.Fatal("Cannot set negative energy {}".format(energy))
        self._energy = energy
    def setAzimuth(self, azimuth):
        if not isinstance(azimuth, Number):
            Logger.Fatal("Cannot set azimuth to {}".format(type(azimuth)))
        if (azimuth<0) or (azimuth>2*pi):
            Logger.Fatal("Invalid azimuth: {}. Is this degrees? Should be radians!".format(azimuth))
        self._azimuth = azimuth
    def setZenith(self, zenith):
        if not isinstance(zenith, Number):
            Logger.Fatal("Cannot set zenith to {}".format(type(zenith)))
        if (zenith<-1) or (zenith>1):
            Logger.Fatal("Invalid zenith {}. Is this in degrees? It should be radians!".format(zenith))
        self._zenith =zenith
    def setTopology(self, topology):
        if not isinstance(topology, int):
            Logger.Fatal("Topology should be an int, not {}".format(type(topology)))
        if topology not in [0,1]:
            Logger.Fatal("Invalid topology {}, only 0 and 1 allowed".format(topology))
        self._topology = topology
    def setYear(self, year):
        if not isinstance(year, int):
            Logger.Fatal("Year should be an int, not {}".format(type(year)))
        if year<0:
            Logger.Fatal("Cannot have negative year {}".format(year))
        self._year = year
    # =================== The Getters
    # they get parameters. 
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
        raise NotImplementedError()
    def __eq__(self, other):
        raise NotImplementedError()
    def __mul__(self, other):
        raise NotImplementedError()

def buildEventFromData(self, dataEvent):
    raise NotImplementedError("Still need to implement this")
