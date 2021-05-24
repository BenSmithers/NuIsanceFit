from libc.math cimport pi, cos, log10
from libcpp cimport bool
"""
Here, we describe the Event classes and methods for interacting with the events 
"""

def EventCache(float cache_weight, float livetime):
    """
    This object is used to cache various properties used during the Weighting 

    We use a dicionary here instead of a dedicated class to synergize better with the Weighters.
    There, we make classes that access one part of the cache. So, we construct those cache weighters with a key. 
    The alternative would be to use a "getattr" function with the class.

    The getattr function is kinda slow though. So, instead we do a key access. 
    """
    this_dict = {}
    # quantities
    this_dict["livetime"] = livetime
    this_dict["weight"] = cache_weight

    # flux cache
    this_dict["convPionWeight"] = cache_weight
    this_dict["convKaonWeight"] = cache_weight
    this_dict["convWeight"] = this_dict["convPionWeight"]
    this_dict["promptWeight"] = cache_weight
    this_dict["astroMuWeight"] = cache_weight
    
    # barr cache
    this_dict["barrModWP"] = cache_weight
    this_dict["barrModWM"] = cache_weight
    this_dict["barrModYP"] = cache_weight
    this_dict["barrModYM"] = cache_weight
    this_dict["barrModZP"] = cache_weight
    this_dict["barrModZM"] = cache_weight

    # holeice cache
    this_dict["holeIceConv"] = cache_weight
    this_dict["holeIcePrompt"] = cache_weight
    this_dict["holeIceAstro"] = cache_weight

    #dom efficiency cache
    this_dict["domEffConv"] = cache_weight
    this_dict["domEffPrompt"] = cache_weight
    this_dict["domEffAstro"] = cache_weight
    return this_dict


cdef class Event:
    """
    This defines 'events' in the detector and all the possible parameters it can have. These are really the fundamental 
    object type that is used in the weighting and likelihood calculation. 

    We bin these in the histograms (well, in the histograms' eventBins)  - see histogram.py
    """
    def __cinit__(self):
        """
        Initializes a null event. Decided against putting all these parameters in the constructor, seems like bad form to have a dozen arguments here .

        Considering later adding a **kwargs and parsing those individually. (check for name, pass to setBLAH function)
        """
        self._is_mc = False

        self._primaryEnergy = 0.0
        self._logPrimaryEnergy = 0.0

        self._primaryAzimuth = 0.0
        self._primaryZenith = 0.0
        self._rawPrimaryZenith = 0.0
        self._primaryType = 0
        self._finalType0 = 0
        self._finalType1 = 0
        self._totalColumnDepth = 0.0
        self._intX = 0
        self._intY = 0

        self._oneWeight = 0.0
        self._num_events = 1

        self._energy = 0.0
        self._logEnergy = 0.0
        self._zenith = 0.0
        self._azimuth = 0.0

        self._sample = 0 # SampleTag.CASCADE
       
        # 0 is track, 1 is cascade
        self._topology = 0
        self._year = 0

        self._cachedweight = EventCache(0.0, 0.0)

    # =================== Getters and Setters =======================
    """
    Check that the object's type is right, and then check that it's a physical value. 

    All of these work the same exact way, and just set some parameter. 
    """
    def setCache(self, dict which):
        self._cachedweight = which
    def setPrimaryType(self, int which):
        self._primaryType = which 
    def setPrimaryEnergy(self, float energy):
        if energy<=0:
            raise ValueError("Cannot set negative energy {}".format(energy))
        self._primaryEnergy = energy
        self._logPrimaryEnergy = log10(energy)
    def setPrimaryAzimuth(self, float azimuth):
        if (azimuth<0) or (azimuth>2*pi):
            raise ValueError("Invalid azimuth: {}. Is this degrees? Should be radians!".format(azimuth))
        self._primaryAzimuth = azimuth
    def setPrimaryZenith(self, float zenith):
        if (zenith<-1) or (zenith>1):
            raise ValueError("Invalid zenith {}. Is this in degrees? It should be radians!".format(zenith))
        self._primaryZenith =zenith
    def setTotalColumnDepth(self, float totalColumnDepth):
        if totalColumnDepth<0:
            raise ValueError("TCD cannot be negative: {}".format(totalColumnDepth))
        self._totalColumnDepth = totalColumnDepth
    def setRawZenith(self, float rawzenith):
        """
        This sets both the raw zenith and the zenith
        """
        if not ((rawzenith>0) and (rawzenith<pi)):
            raise ValueError("Zenith angle should be between 0 and pi")
        self._rawPrimaryZenith = rawzenith
        self.setPrimaryZenith(cos(self._rawPrimaryZenith))

    def setIntX(self, float intX):
        if intX<0:
            raise ValueError("Cannot have negative Bjorken X {}".format(intX))
        self._intX = intX
    def setIntY(self, float intY):
        if intY<0:
            raise ValueError("Cannot have negative Bjorken Y {}".format(intY))
        self._intY = intY
    def setOneWeight(self, float oneWeight):
        if oneWeight<0:
            raise ValueError("Cannot have negative oneWeight {}".format(oneWeight))
        self._oneWeight = oneWeight
    def setNumEvents(self, int num_events):
        if num_events<0:
            raise ValueError("Cannot have negative number of events {}".format(num_events))
        self._num_events = num_events
    def setEnergy(self, float energy):
        if energy<=0:
            raise ValueError("Cannot set negative energy {}".format(energy))
        self._energy = energy
        self._logEnergy = log10(energy)
    def setAzimuth(self, float azimuth):
        if (azimuth<0) or (azimuth>2*pi):
            raise ValueError("Invalid azimuth: {}. Is this degrees? Should be radians!".format(azimuth))
        self._azimuth = azimuth
    def setZenith(self, float zenith):
        if (zenith<-1) or (zenith>1):
            raise ValueError("Invalid zenith {}. Is this in degrees? It should be radians!".format(zenith))
        self._zenith =zenith
    def setTopology(self, int topology):
        if topology not in [0,1]:
            raise ValueError("Invalid topology {}, only 0 and 1 allowed".format(topology))
        self._topology = topology
    def setYear(self, int year):
        if year<0:
            raise ValueError("Cannot have negative year {}".format(year))
        self._year = year
    def setFinalType0(self,int value):
        self._finalType0 = value
    def setFinalType1(self,int value):
        self._finalType1 = value
    def setIsMC(self, bint value):
        self._is_mc = value
    # =================== The Getters
    # they get parameters. 
    @property
    def is_mc(self):
        return self._is_mc
    @property
    def rawPrimaryZenith(self):
        return self._rawPrimaryZenith
    @property
    def primaryType(self):
        return self._primaryType
    @property
    def finalType0(self):
        return self._finalType0
    @property
    def finalType1(self):
        return self._finalType1
    @property
    def primaryEnergy(self):
        return self._primaryEnergy
    @property
    def logPrimaryEnergy(self):
        return self._logPrimaryEnergy
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
    def logEnergy(self):
        return self._logEnergy
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
        return self._cachedweight["convWeight"] + self._cachedweight["promptWeight"] + self._cachedweight["astroMuWeight"]

    def __add__(self, other):
        raise NotImplementedError()
    def __eq__(self, other):
        raise NotImplementedError()
    def __mul__(self, other):
        raise NotImplementedError()

    def __str__(self):
        rep = "Event: "
        rep += "    energy {}\n".format(self.energy)
        rep += "    zenith {}\n".format(self.zenith)
        rep += "    azimuth {}\n".format(self.azimuth)
        rep += "    oneWeight {}\n".format(self.oneWeight)
        rep += "    primaryEnergy {}\n".format(self.primaryEnergy)
        rep += "    primaryAzimuth {}\n".format(self.primaryAzimuth)
        rep += "    primaryZenith {}\n".format(self.primaryZenith)
        rep += "    isMC {}".format(self.is_mc)
        return rep
