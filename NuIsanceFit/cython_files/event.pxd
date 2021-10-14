## cdef dict EventCache(float, float)

cdef class Event:
    cdef readonly bint _is_mc
    cdef readonly float _primaryEnergy
    cdef readonly float _logPrimaryEnergy
    cdef readonly float _primaryAzimuth
    cdef readonly float _primaryZenith
    cdef readonly int _primaryType
    cdef readonly float _rawPrimaryZenith
    cdef readonly int _finalType0
    cdef readonly int _finalType1
    cdef readonly float _totalColumnDepth
    cdef readonly float _intX
    cdef readonly float _intY
    cdef readonly float _oneWeight
    cdef readonly int _num_events
    cdef readonly float _energy
    cdef readonly float _logEnergy
    cdef readonly float _zenith
    cdef readonly float _azimuth
    cdef readonly int _sample
    cdef readonly int _topology
    cdef readonly int _year
    cdef readonly dict _cachedweight
    cdef readonly list _snowstorm_params

    # we want some c-level access functions that avoid the python stack!
    cdef float getPrimaryEnergy(self)
    cdef float getLogPrimaryEnergy(self)
    cdef float getPrimaryZenith(self)
    cdef float getEnergy(self)
    cdef float getLogEnergy(self)
    cdef int getPrimaryType(self)
    cdef dict getCache(self)
    cdef int getTopology(self)