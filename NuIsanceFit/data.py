from NuIsanceFit import Logger
from NuIsanceFit.histogram import bHist, eventBin
from NuIsanceFit.event import Event, EventCache
from NuIsanceFit import simdata

from numbers import Number
import numpy as np
import h5py as h5
import os
from math import log10, cos
from glob import glob

try:
    import nuSQuIDS as nsq
except ImportError:
    # older nusquids installations have a differently named nusquids module. 
    # I don't think we'll rely on any new-new features, so let's allow this 
    import nuSQuIDSpy as nsq

import LeptonWeighter as LW

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
        self._simToLoad = self._get_filtered_sims()

        self._dataToLoad = [steering["dataToLoad"]]
        self._dataToLoad = [os.path.join( steering["datadir"], entry) for entry in self._dataToLoad]
        
        for entry in self._simToLoad:
            fn = os.path.join(steering["datadir"], self._simToLoad[entry]["filename"])
            if not os.path.exists(fn):
                Logger.Fatal("Could not find simulation at {}".format(fn))

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
        self._livetime = 1.0 # this should come from the steering 

        # ENERGY | COSTH | AZIMUTH | TOPOLOGY | TIME
        self.simulation = bHist([ self._Eedges, self._cosThEdges, self._azimuthEdges, self._topoEdges, self._timeEdges ], bintype=eventBin,datatype=Event)
        self.data = bHist([ self._Eedges, self._cosThEdges, self._azimuthEdges, self._topoEdges, self._timeEdges ], bintype=eventBin,datatype=Event)

        # ================================ Prepare some weighting stuff for building the event caches 
        self._lic_files = []
        for entry in self._simToLoad:
            name = os.path.join(steering["datadir"],self._simToLoad[entry]["lic_file"])
            self._lic_files += LW.MakeGeneratorsFromLICFile(name)
        self._xs_obj = LW.CrossSectionFromSpline(self.steering["resources"]["diff_nu_cc_xs"], self.steering["resources"]["diff_nubar_cc_xs"],
                                                    self.steering["resources"]["diff_nu_nc_xs"], self.steering["resources"]["diff_nubar_nc_xs"])
        # TODO: toggleable single/double atmo file 
        self._convFluxWeighter =   LW.Weighter(LW.nuSQUIDSAtmFlux(self.steering["resources"]["conv_atmo_flux"]),self._xs_obj, self._lic_files)
        self._promptFluxWeighter = LW.Weighter(LW.nuSQUIDSAtmFlux(self.steering["resources"]["prompt_atmo_flux"]),self._xs_obj, self._lic_files)
        self._astroFluxWeighter =  LW.Weighter(LW.nuSQUIDSAtmFlux(self.steering["resources"]["astro_file"]),self._xs_obj, self._lic_files)
        self._barr_resources = self._load_barr_resources()
        Logger.Log("Loaded in Weighter data")

        # these will be necessary if we use two atmospheric files 
        #self._pionFluxWeighter = LW.Weighter(self.steering["resources"]["pion_atmo_flux"],self._xs_obj, self._lic_files)
        #self._kaonFluxWeighter = LW.Weighter(self.steering["resources"]["kaon_atmo_file"],self._xs_obj, self._lic_files)

        # TODO use different loadMC function depending on mctype (from steering)
        self.loadMC()
        self.loadData()

    def _load_barr_resources(self):
        """
        Open up the location where the barr gradients are stored, make the nusquids objects, put them in a dictionary, and return them 
        """
        barr_files = glob(os.path.join(self.steering["resources"]["barr_location"], "*.hdf5"))
        if not len(barr_files)==8:
            Logger.Fatal("Found inappropriate number of barr files, {}, at {}".format(len(barr_files), self.steering["resources"]["barr_location"]))

        barrs = {}
        for entry in barr_files:
            filename = os.path.split(entry)[1]
            barr_letters = filename.split("_")[2]
            barrs["barrMod"+barr_letters] = nsq.nuSQUIDSAtm(entry)

        return barrs

    def _get_filtered_sims(self):
        """
        This filters out the full dictionary of simulation data to _just_ what the user wants 

        First, we apply the name/tag filter. That's the easy one
        Then, we allow the user to specify other parameters. Like holeIceForward=2.0, or whatever it is they want! 
        Extra filtering 
        
        This returns a filtered dictionary 
        """
        # first we grab the relevant simsets, filtering by name
        init_f = filter(lambda fullname:self.steering["simToLoad"]["name"] in fullname, simdata)

        # users may have specified advaned sets 
        for check_key in self.steering["simToLoad"]:
            if str(check_key)=="name":
                continue
            else:
                init_f = filter(lambda entry: self.steering["simToLoad"][check_key]==entry[check_key], init_f)
        
        init_f = list(init_f)

        init_d = {}
        for key in init_f:
            init_d[key] = simdata[key]
        return init_d

    def _fillCache(self, event):
        if not isinstance(event, Event):
            Logger.Fatal("Cannot add cache to {}",format(type(event)), TypeError)
        nucache = EventCache(0.,0.)

        # make a Lw event
        nuEvent = LW.Event()
        nucache["livetime"] = self._livetime
        nuEvent.energy = event.primaryEnergy
        nuEvent.azimuth = event.primaryAzimuth
        nuEvent.zenith = event.rawPrimaryZenith
        nuEvent.total_column_depth = event.totalColumnDepth
        nuEvent.interaction_x = event.intX
        nuEvent.interaction_y = event.intY
        nuEvent.final_state_particle_0 = LW.ParticleType(event.finalType0)
        nuEvent.final_state_particle_1 = LW.ParticleType(event.finalType1)
        nuEvent.primary_type = LW.ParticleType(event.primaryType)
        nucache["convWeight"] = self._convFluxWeighter(nuEvent)*self._livetime/event.num_events
        nucache["convPionWeight"] = nucache["convWeight"]
        
        nucache["promptWeight"] = self._promptFluxWeighter(nuEvent)*self._livetime/event.num_events
        if abs(event.primaryType)==14: 
            nucache["astroMuWeight"] = self._astroFluxWeighter(nuEvent)*self._livetime/event.num_events
        # TODO Add contribution from tau neutrinos 
        if abs(event.primaryType)==12: #electron
            falvor = 0
        elif abs(event.primaryType)==14: #mu
            flavor = 1
        elif abs(event.primaryType)==16: #tau
            flavor = 2
        else:
            Logger.Fatal("non-neutrino event! PDG: {}".format(event.primaryType))
        nutype = 0 if event.primaryType>0 else 1

        for key in self._barr_resources.keys():
            nucache[key] = self._barr_resources[key].EvalFlavor(flavor, event.primaryZenith, event.primaryEnergy*(1e9), nutype )*event.oneWeight

        event.setCache(nucache)

    def _loadFile(self, which_file, target_hist, is_mc):
        """
        We use this as a switch between different methods of loading the data in. Different analysis files from different sim sets use different hdf5 file structures
        So in accessing them we need different scripts to parse the data

        TODO: put the mctype in the simdata json file, that way we can have this load in data from different analyses 
        """
        if self.steering["mctype"].lower() == "hese":
            self._loadFile_hese(which_file, target_hist, is_mc)
        elif self.steering["mctype"].lower() == "sterile":
            self._loadFile_sterile(which_file, target_hist, is_mc)
        else:
            Logger.Fatal("MC Type {} is unimplemented".format(self.steering["mctype"]), NotImplementedError)

    def _loadFile_sterile(self, which_file, target_hist, is_mc):
        Logger.Log("Opening {}".format(which_file))
        data = h5.File(which_file, 'r')
        i_event = 0

        _e_reco = data["MuExEnergy"][:]
        _z_reco = data["MuExZenith"][:]
        _a_reco = data["MuExAzimuth"][:]

        _e_true = data["NuEnergy"][:]
        _z_true = data["NuZenith"][:]
        _a_true = data["NuAzimuth"][:]
        _weight = data["oneweight"][:]
        _tcd = data["TotalColumnDepth"][:]
        _prim = data["PrimaryType"][:]
        _bjx = data["FinalStateX"][:]
        _bjy = data["FinalStateY"][:]
        _fs0 = data["FinalType0"][:]
        _fs1 = data["FinalType1"][:]
        
        Logger.Log("Opened!")  
        i_max = len(data["NuEnergy"]) 
        while i_event<i_max:
            new_event = Event()
            # note: the first four entries are for 
            #        Run, Event, SubEvent, SubEventStream, and Existance 
            new_event.setEnergy(  _e_reco[i_event][5] )
            new_event.setZenith( cos(_z_reco[i_event][5]) )
            new_event.setAzimuth( _a_reco[i_event][5] )
            new_event.setTopology( 0 ) # all of these are tracks right now
            new_event.setYear( 0 ) #TODO change this when you want to bin in time 

            if is_mc:
                new_event.setIsMC(True)
                new_event.setPrimaryEnergy(  _e_true[i_event][5] )
                new_event.setPrimaryAzimuth( _a_true[i_event][5] )
                new_event.setRawZenith( _z_true[i_event][5] ) 
                new_event.setPrimaryType(int(_prim[i_event][5]))                
                new_event.setFinalType0(int(_fs0[i_event][5]))
                new_event.setFinalType1(int(_fs1[i_event][5]))
                new_event.setNumEvents( i_max )
                new_event.setOneWeight(_weight[i_event][5] )
                new_event.setTotalColumnDepth(_tcd[i_event][5])
                new_event.setIntX( _bjx[i_event][5])
                new_event.setIntY( _bjy[i_event][5])
                self._fillCache(new_event)
        
            target_hist.add(new_event, new_event.energy, new_event.zenith, new_event.azimuth, new_event.topology, new_event.year)

            if i_event%25000==0:
                Logger.Log("Loaded {} Events so far".format(i_event))
            i_event+=1 
        data.close()

    def _loadFile_hese(self, which_file, target_hist, is_mc):
        """
        Here, we load in an hdf5 file and create and bin the events we see
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

        i_max = len(data["is_track"])
        while i_event<i_max:
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
                new_event.setPrimaryEnergy(  _weight[i_event][33] )
                new_event.setPrimaryAzimuth( _weight[i_event][32] )
                new_event.setRawZenith( _weight[i_event][35] ) # sets cos(zenith) and the raw zenith. It stores both! 
                new_event.setNumEvents( _weight[i_event][27] )
                new_event.setOneWeight(_weight[i_event][29] )
                self._fillCache(new_event)


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

        We have a filtered dictionary of simulation. We go to each entry, and extract the filename
        Some of those filenames are folders. If it's a folder, we load everything in the folder
        Otherwise we just load the file! 
        """
        for entry in self._simToLoad:
            filename = os.path.join( self.steering["datadir"], self._simToLoad[entry]["filename"])
            if os.path.isdir(filename):
                # scan through the contents of the folder 
                contents = glob(os.path.join(filename, "*.h5"))
                for content in contents:
                    self._loadFile(content, self.simulation, True)
            else:
                self._loadFile(filename, self.simulation, True )

    def loadData(self):
        """
        Uses the generic data loader to load... the data
        """
        for entry in self._dataToLoad:
            self._loadFile( entry, self.data, False)
