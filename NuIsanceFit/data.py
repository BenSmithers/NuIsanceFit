from logger import Logger

"""
This is where we actually build the data and simulation histograms 

Sim keys:
 'FinalStateX',
 'FinalStateY',
 'FinalType0',
 'FinalType1',
 'ImpactParameter',
 'MuExAzimuth',
 'MuExEnergy',
 'MuExZenith',
 'NuAzimuth',
 'NuEnergy',
 'NuZenith',
 'PrimaryType',
 'TotalColumnDepth',
 '__I3Index__',
 'oneweight'

Data keys:
 'dec_reco',
 'energy_reco',
 'is_cascade',
 'is_track',
 'ra_reco',
 'time',
 'year',
 'zenith_reco'
"""

from histogram import bhist, eventBin
from event import Event, EventCache

import h5py as h5
import os
from math import log10

def make_edges(bin_params, key):
    """
    We take in the section from the json file specifying the binning parameters (the steering file)
    and the key corresponding to which one we want to work with.

    Then, we build the numpy array specifying the edges of the bins 
    and return this! 
    """
    emin = bin_params[key]["min"]
    emax = bin_params[key]["max"]
    if bin_params[key]["log"]:
        emin = log10(emin)
        emax = log10(emax)
        Eedges = np.logspace( emin, emax, int((emax-emin)/bin_params[key]["binWidth"]))
    else:
        Eedges = np.linspace( emin, emax, int((emax-emin)/bin_params[key]["binWidth"]))

    return Eedges

class Data:
    """
    This class maintains the data itself. It holds the data, and is the intermediate for data requests 
    """
    def __init__(self, steering):
        self.steering = steering

       # year, azimuth, zenith, energy  

        self._simToLoad = [steering["simToLoad"]]
        self._simToLoad = [os.path.join( steering["datadir"], entry) for entry in self._simToLoad]

        for entry in self._simToLoad:
            if not os.path.exists(entry):
                Logger.Fatal("Could not find simulation at {}".format(entry))

        bins = steering["binning"]

        self._Eedges = make_edges(bins, "energy")
        self._cosThEdges = make_edges(bins, "cosTh")
        self._azimuthEdges = make_edges(bins, "azimuth")
        self._topoEdges = [-0.5, 0.5, 1.5] # only two bins, 0 and 1

        self.simulation = bhist([ self._Eedges, self._cosThEdges, self._azimuthEdges, self._topoEdges ])
        self.data = bhist([ self._Eedges, self._cosThEdges, self._azimuthEdges, self._topoEdges ])

    def loadMC(self):
        
        for entry in self._simToLoad:
            data = h5.File(entry, 'r')
            i_event = 0
            while i_event<len(data["is_track"]):
                new_event = Event()
                new_event.setEnergy(  data["energy_reco"][i_event][-1] )
                new_event.setZenith(  data["zenith_reco"][i_event][-1] )
                new_event.setAzimuth(data["azimuth_reco"][i_event][-1] )
                

                i_event+=1 

    def loadData(self):
        pass 

"""
This is where we actually build the data and simulation histograms 

Sim keys:
 'FinalStateX',
 'FinalStateY',
 'FinalType0',
 'FinalType1',
 'ImpactParameter',
 'MuExAzimuth',
 'MuExEnergy',
 'MuExZenith',
 'NuAzimuth',
 'NuEnergy',
 'NuZenith',
 'PrimaryType',
 'TotalColumnDepth',
 '__I3Index__',
 'oneweight'

Data keys:
 'dec_reco',
 'energy_reco',
 'is_cascade',
 'is_track',
 'ra_reco',
 'time',
 'year',
 'zenith_reco'
"""

