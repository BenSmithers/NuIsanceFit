from NuIsanceFit.logger import Logger

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

from NuIsanceFit.histogram import bHist, eventBin
from NuIsanceFit.event import Event, EventCache

import time
from numbers import Number
import numpy as np
import h5py as h5
import os
from math import log10, cos

def make_edges(bin_params, key):
    """
    We take in the section from the json file specifying the binning parameters (the steering file)
    and the key corresponding to which one we want to work with.

    Then, we build the numpy array specifying the edges of the bins 
    and return this! 
    """
    emin = bin_params[key]["min"]
    emax = bin_params[key]["max"]
    if not isinstance(emin, Number):
        Logger.Fatal("min is a {}, not a number".format(type(emin)))
    if not isinstance(emax, Number):
        Logger.Fatal("max is a {}, not a number".format(type(emax)))
    if not isinstance(bin_params[key]["log"], bool):
        Logger.Fatal("bins is {}, not a bool".format(type(bin_params[key]["log"])))
    binno = bin_params[key]["bins"]
    if not isinstance(binno, int):
        Logger.Fatal("'bins' should be an {}, not {}".format(int, type(binno)))

    if bin_params[key]["log"]:
        emin = log10(emin)
        emax = log10(emax)
        Eedges = np.logspace( emin, emax, binno+1) # we add one since n_edges = n_bins + 1
    else:
        Eedges = np.linspace( emin, emax, binno+1)

    return Eedges

class Data:
    """
    This class maintains the data itself. It holds the data, and is the intermediate for data requests. When this is constructed, it loads in the data!  
    """
    def __init__(self, steering):
        """
        Arg 'steering' should be a dictionary. It'll be loaded in from the 'steering.json' file
        """
        if not isinstance(steering, dict):
            Logger.Fatal("Expected {}, got {}".format(dict, type(steering)))
        self.steering = steering

       # year, azimuth, zenith, energy  

        self._simToLoad = [steering["simToLoad"]]
        self._simToLoad = [os.path.join( steering["datadir"], entry) for entry in self._simToLoad]

        self._dataToLoad = [steering["dataToLoad"]]
        self._dataToLoad = [os.path.join( steering["datadir"], entry) for entry in self._dataToLoad]
        
        for entry in self._simToLoad:
            if not os.path.exists(entry):
                Logger.Fatal("Could not find simulation at {}".format(entry))

        for entry in self._dataToLoad:
            if not os.path.exists(entry):
                Logger.Fatal("Could not find data at {}".format(entry))

        bins = steering["binning"]

        # by default, azimuth and time each are both one big happy bin
        self._Eedges = make_edges(bins, "energy")
        self._cosThEdges = make_edges(bins, "cosTh")
        self._azimuthEdges = make_edges(bins, "azimuth")
        self._topoEdges = [-0.5, 0.5, 1.5] # only two bins to catch 0 and 1
        self._timeEdges = make_edges(bins, "year") 

        # ENERGY | COSTH | AZIMUTH | TOPOLOGY | TIME
        self.simulation = bHist([ self._Eedges, self._cosThEdges, self._azimuthEdges, self._topoEdges, self._timeEdges ], bintype=eventBin,datatype=Event)
        self.data = bHist([ self._Eedges, self._cosThEdges, self._azimuthEdges, self._topoEdges, self._timeEdges ], bintype=eventBin,datatype=Event)
        
        # TODO use different loadMC function depending on mctype (from steering)
        self.loadMC()
        self.loadData()

    def _loadFile(self, which_file, target_hist, is_mc):
        """
        Here, we load in the hdf5 files (one at at time), then create and bin the events we see

        The indices might seem kind of suspect, but you can verify they are correct by opening the hdf5 files and looking at the 'attrs' property of the different databases. That's a dictionary like object that stores the units and name of each entry in a database 
        """
        
        Logger.Log("Opening {}".format(which_file))
        data = h5.File(which_file, 'r')
        i_event = 0

        # we want to read in the whole dataset! 
        # TODO: do this in a 'chunked' way so it loads in (up to) a few GB at a time
        _e_reco = data["energy_reco"][:]
        _z_reco = data["zenith_reco"][:]
        _a_reco = data["azimuth_reco"][:]
        _is_cascade = data["is_cascade"][:]
        _primary = data["MCPrimary"][:]
        _weight = data["I3MCWeightDict"][:]
        Logger.Log("Opened!")   


        while i_event<len(data["is_track"]):
            new_event = Event()
            # note: the first four entries are for 
            #        Run, Event, SubEvent, SubEventStream, and Existance 
            new_event.setEnergy(  _e_reco[i_event][5] )
            new_event.setZenith( cos(_z_reco[i_event][5]) )
            new_event.setAzimuth( _a_reco[i_event][5] )
            new_event.setTopology(int(_is_cascade[i_event][5]) )
            new_event.setYear( 0 ) #TODO change this when you want to bin in time 

            if is_mc:
                new_event.is_mc = True
                #new_event.setPrimaryEnergy(  _primary[i_event][11] )
                #new_event.setPrimaryAzimuth( _primary[i_event][10] )
                #new_event.setPrimaryZenith( cos(_primary[i_event][9]) )
                new_event.setPrimaryEnergy(  _weight[i_event][33] )
                new_event.setPrimaryAzimuth( _weight[i_event][32] )
                new_event.setPrimaryZenith( cos(_weight[i_event][35]) )

                new_event.setOneWeight(_weight[i_event][29] )
            # I was finding negative Bjorkens while loading in data. Need to investiagate what's up with that...
            #new_event.setIntX( data["I3MCWeightDict"][i_event][5] )
            #new_event.setIntY( data["I3MCWeightDict"][i_event][6] )
        
            target_hist.add(new_event, new_event.energy, new_event.zenith, new_event.azimuth, new_event.topology, new_event.year)

            if i_event%100000==0:
                Logger.Log("Loaded {} Events so far".format(i_event))
            i_event+=1 

        data.close()

    def loadMC(self):
        """
        Uses the generic data loader to load simulation
        """
        for entry in self._simToLoad:
            self._loadFile( entry, self.simulation, True )
    def loadData(self):
        """
        Uses the generic data loader to load... the data
        """
        for entry in self._dataToLoad:
            self._loadFile( entry, self.data, False)

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

