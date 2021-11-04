from numba.cuda import target
from NuIsanceFit import Logger
from NuIsanceFit.histogram import bHist, eventBin
from NuIsanceFit.event import Event, EventCache
from NuIsanceFit import simdata

from numbers import Number
import numpy as np
import h5py as h5
import os
from math import log10, cos, sqrt, pi
from glob import glob

try:
    import nuSQuIDS as nsq
except ImportError:
    # older nusquids installations have a differently named nusquids module. 
    # I don't think we'll rely on any new-new features, so let's allow this 
    import nuSQuIDSpy as nsq

import LeptonWeighter as LW

def temporary_fluxfunc(energy):
    return (1e-18)*( energy/(100e3) )**-2.2

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

       # this is now a dictionary with an entry for each set we're loading. 
       # the sets are the same as those defined in simdata.json
        self._simToLoad = self._get_filtered_sims()

        self._dataToLoad = [steering["dataToLoad"]]
        self._dataToLoad = [os.path.join( steering["datadir"], entry) for entry in self._dataToLoad]
        
        # verify that each of those entries exists! 
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

        # year, azimuth, zenith, energy  
        # ENERGY | COSTH | AZIMUTH | TOPOLOGY | TIME
        self.simulation = bHist([ self._Eedges, self._cosThEdges, self._azimuthEdges, self._topoEdges, self._timeEdges ], bintype=eventBin,datatype=Event)
        self.data = bHist([ self._Eedges, self._cosThEdges, self._azimuthEdges, self._topoEdges, self._timeEdges ], bintype=eventBin,datatype=Event)


        # ================================ Prepare some weighting stuff for building the event caches 
        have_lic_files = True
        self._lic_files = []
        for entry in self._simToLoad:
            if "lic_file" not in self._simToLoad[entry]:
                have_lic_files = False
                break
            name = os.path.join(steering["datadir"],self._simToLoad[entry]["lic_file"])
            self._lic_files += LW.MakeGeneratorsFromLICFile(name)

        if have_lic_files:
            self._xs_obj = LW.CrossSectionFromSpline(self.steering["resources"]["diff_nu_cc_xs"], self.steering["resources"]["diff_nubar_cc_xs"],
                                                        self.steering["resources"]["diff_nu_nc_xs"], self.steering["resources"]["diff_nubar_nc_xs"])
            # TODO: toggleable single/double atmo file 
            self._convFluxWeighter =   LW.Weighter(LW.nuSQUIDSAtmFlux(self.steering["resources"]["conv_atmo_flux"]),self._xs_obj, self._lic_files)
            self._promptFluxWeighter = LW.Weighter(LW.nuSQUIDSAtmFlux(self.steering["resources"]["prompt_atmo_flux"]),self._xs_obj, self._lic_files)
            self._astroFluxWeighter =  LW.Weighter(LW.nuSQUIDSAtmFlux(self.steering["resources"]["astro_file"]),self._xs_obj, self._lic_files)
        else:
            self._convFluxWeighter =   None
            self._promptFluxWeighter = None
            self._astroFluxWeighter =  None

        self._barr_resources = self._load_barr_resources()

        Logger.Log("Loaded in Weighter data")

        # these will be necessary if we use two atmospheric files 
        #self._pionFluxWeighter = LW.Weighter(self.steering["resources"]["pion_atmo_flux"],self._xs_obj, self._lic_files)
        #self._kaonFluxWeighter = LW.Weighter(self.steering["resources"]["kaon_atmo_file"],self._xs_obj, self._lic_files)

        self._ss_mode = True
        self._ss_param_names = []
        self._ss_param_centers = []
        self._ss_param_sigmas = []

        # TODO use different loadMC function depending on mctype (from steering)
        self.loadMC()
        self.loadData()

        # load in the snowstorm centers and sigmas 
        if self._ss_mode:
            for entry in self._simToLoad:
                # load in if empty
                if self._ss_param_centers==[]:
                    self._ss_param_centers=self._simToLoad[entry]["snowstorm_centers"]
                    self._ss_param_sigmas= self._simToLoad[entry]["snowstorm_sigmas"]
                else:
                    #check they're the same if not empty 
                    if not len(self._ss_param_sigmas)==len(self._simToLoad[entry]["snowstorm_sigmas"]):
                        raise ValueError("One sigmas list had len {}, the other {}".format(len(self._ss_param_sigmas),len(self._simToLoad[entry]["snowstorm_sigmas"])))
                    if not all(self._ss_param_sigmas[i] == self._simToLoad[entry]["snowstorm_sigmas"] for i in range(len(self._simToLoad[entry]["snowstorm_sigmas"]))):
                        raise ValueError("Tried loading in samples with different sigmas")

                    if not len(self._ss_param_centers)==len(self._simToLoad[entry]["snowstorm_centers"]):
                        raise ValueError("One sigmas list had len {}, the other {}".format(len(self._ss_param_centers),len(self._simToLoad[entry]["snowstorm_centers"])))
                    if not all(self._ss_param_centers[i] == self._simToLoad[entry]["snowstorm_centers"] for i in range(len(self._simToLoad[entry]["snowstorm_centers"]))):
                        raise ValueError("Tried loading in samples with different sigmas")


    def get_splits(self):
        if not self._ss_mode: #can only do this in snowstorm mode 
            return

        e_above = np.zeros(shape=(len(self._Eedges)-1, len(self._ss_param_centers)))
        e_below = np.zeros(shape=(len(self._Eedges)-1, len(self._ss_param_centers)))

        z_above = np.zeros(shape=(len(self._cosThEdges)-1, len(self._ss_param_centers)))
        z_below = np.zeros(shape=(len(self._cosThEdges)-1, len(self._ss_param_centers)))

        # count all the events above/below in each bin
        print("There are {} events total.".format(len(np.sum(self.simulation))))
        made_it = False
        for i_e in range(len(self.simulation)):
            for i_cth in range(len(self.simulation[i_e])):
                for i_azi in range(len(self.simulation[i_e][i_cth])):
                    for i_topo in range(len(self.simulation[i_e][i_cth][i_azi])):
                        for i_time in range(len(self.simulation[i_e][i_cth][i_azi][i_topo])):
                            # this gets us a bin, iterate over the events now
                            for event in self.simulation[i_e][i_cth][i_azi][i_topo][i_time]:
                                if not made_it:
                                    print("Should be at least one")
                                    made_it = True
                                evt_params = event.snowStormParams
                                n_eff = event.oneWeight*temporary_fluxfunc(event.primaryEnergy)

                                for i_param in range(len(self._ss_param_centers)):
                                    if evt_params[i_param]>self._ss_param_centers[i_param]:
                                        e_above[i_e][i_param] +=n_eff
                                        z_above[i_cth][i_param] +=n_eff
                                    else:
                                        e_below[i_e][i_param]+=n_eff
                                        z_below[i_cth][i_param]+=n_eff


        # split along all the different snow storm parameters 
        e_gradient =  np.zeros(shape=(len(self._Eedges)-1, len(self._ss_param_centers)))
        z_gradient =  np.zeros(shape=(len(self._cosThEdges)-1, len(self._ss_param_centers)))

        for i_param in range(len(self._ss_param_centers)):
            for e_bin in range(len(e_gradient)):
                e_gradient[e_bin][i_param] = (1/self._ss_param_sigmas[i_param])*sqrt(pi/2)*(e_above[e_bin][i_param] - e_below[e_bin][i_param])
            for cth_bin in range(len(z_gradient)):
                z_gradient[cth_bin][i_param] = (1/self._ss_param_sigmas[i_param])*sqrt(pi/2)*(z_above[cth_bin][i_param] - z_below[cth_bin][i_param])

        # get the covariance matrices for the nuisance paramters
        # these come from calibration, right now I just have a diagonal covariance 
        cov_mat = np.zeros(shape=(len(self._ss_param_centers), len(self._ss_param_centers)))
        for i in range(len(self._ss_param_centers)):
            cov_mat[i][i] = 1.
        multiv_gauss = np.linalg.inv(cov_mat)

        # extract analysis level covariance 
        final_cov_e = np.zeros(shape=(len(self._Eedges)-1,len(self._Eedges)-1))
        final_cov_cth = np.zeros(shape=(len(self._cosThEdges)-1,len(self._cosThEdges)-1))
        for i_sandwich in range(len(self._ss_param_centers)):
            for j_sandwich in range(len(self._ss_param_centers)):
                for i_alpha in range(len(e_gradient)):
                    for i_beta in range(len(e_gradient)):
                        final_cov_e[i_alpha][i_beta] = e_gradient[i_alpha][i_sandwich]*multiv_gauss[i_sandwich][j_sandwich]*e_gradient[i_beta][j_sandwich]

                for i_alpha in range(len(z_gradient)):
                    for i_beta in range(len(z_gradient)):
                        final_cov_cth[i_alpha][i_beta] = z_gradient[i_alpha][i_sandwich]*multiv_gauss[i_sandwich][j_sandwich]*z_gradient[i_beta][j_sandwich]

        return(final_cov_e, final_cov_cth)


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
        nucache = EventCache(event.oneWeight ,self._livetime)

        # make a Lw event
        nuEvent = LW.Event()
        nucache["livetime"] = self._livetime
        nuEvent.energy = event.primaryEnergy
        nuEvent.azimuth = event.primaryAzimuth
        nuEvent.zenith = event.rawPrimaryZenith
        nuEvent.total_column_depth = event.totalColumnDepth
        if event.intX>0 and event.intY>0:
            nuEvent.interaction_x = event.intX
            nuEvent.interaction_y = event.intY
        nuEvent.final_state_particle_0 = LW.ParticleType(event.finalType0)
        nuEvent.final_state_particle_1 = LW.ParticleType(event.finalType1)
        nuEvent.primary_type = LW.ParticleType(event.primaryType)
        event.setOneWeight( self._convFluxWeighter(nuEvent)*self._livetime/event.num_events )
        nucache["convWeight"] = event.oneWeight
        #nucache["convWeight"] =self._convFluxWeighter(nuEvent)*self._livetime/event.num_events
        nucache["convPionWeight"] = nucache["convWeight"]
        
        nucache["promptWeight"] = self._promptFluxWeighter(nuEvent)*self._livetime/event.num_events
        if abs(event.primaryType)==14: 
            nucache["astroMuWeight"] = self._astroFluxWeighter(nuEvent)*self._livetime/event.num_events
        # TODO Add contribution from tau neutrinos 
        if abs(event.primaryType)==12: #electron
            flavor = 0
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
        access_key = "mctype" if is_mc else "datatype"

        if is_mc and self.steering["mctype"].lower()!="casc_dnn":
            self._ss_mode = False
        if self.steering[access_key].lower() == "hese":
            self._loadFile_hese(which_file, target_hist, is_mc)
        elif self.steering[access_key].lower() == "sterile":
            self._loadFile_sterile(which_file, target_hist, is_mc)
        elif self.steering[access_key].lower()=="casc_dnn":
            self._loadFile_casc_dnn(which_file, target_hist, is_mc)
        else:
            Logger.Fatal("Access Key Type {} is unimplemented".format(self.steering[access_key]), NotImplementedError)

    def _loadFile_casc_dnn(self, which_file, target_hist, is_mc=True):
        Logger.Log("Opening {}".format(which_file))
        data = h5.File(which_file, 'r')
        i_event = 0

        _e_reco = data["reco_energy"][:]
        _a_reco = data["reco_azimuth"][:]
        _z_reco = data["reco_zenith"][:]
        
        _e_true = data["true_energy"][:]
        _a_true = data["true_azimuth"][:]
        _z_true = data["true_zenith"][:]
        _weight = data["one_weight"][:]
        _tcd = data["total_column_depth"][:]
        _prim = data["true_pid"][:]
        _bjx = data["true_bjorkenx"][:]
        _bjy = data["true_bjorkeny"][:]

        _casc = data["cascade_score_reco"][:]
        _track = data["track_score_reco"][:]
        _snow_storm_params = data["snowstorm_params"][:]
        _snowstorm_ref = data["snowstorm_ref"][:] # reference to which m frame we're using I think 
        Logger.Log("{} refs ones, {} snowstorm ones".format(len(_e_reco), len(_snowstorm_ref)))
        
        # TODO: put in some checker so that it makes sure all the events have the same number of snowstorm parameters 
        #  may also want it to keep track of the parameter names                 

        Logger.Log("Opened!")  
        i_max = len(_e_reco)
        while i_event < i_max:
            new_event = Event()
            new_event.setEnergy(  _e_reco[i_event] )
            new_event.setZenith( cos(_z_reco[i_event]) )
            new_event.setAzimuth( _a_reco[i_event] )
            new_event.setTopology( 0 if _track[i_event]>_casc[i_event] else 1 ) 
            new_event.setYear( 0 ) #TODO change this when you want to bin in time     
            if is_mc:
                new_event.setIsMC(True)
                new_event.setPrimaryEnergy(  _e_true[i_event] )
                new_event.setPrimaryAzimuth( _a_true[i_event] )
                new_event.setRawZenith( _z_true[i_event] ) 
                new_event.setPrimaryType(int(_prim[i_event]))                
                #new_event.setFinalType0(int(_fs0[i_event][5]))
                #new_event.setFinalType1(int(_fs1[i_event][5]))
                new_event.setNumEvents( i_max )
                new_event.setOneWeight(_weight[i_event] )
                new_event.setTotalColumnDepth(_tcd[i_event])
                # these are -1 for GR events 
                if _bjx[i_event]>0:
                    new_event.setIntX( _bjx[i_event])
                if _bjy[i_event]>0:
                    new_event.setIntY( _bjy[i_event])
                new_event.setSnowStormParams( list(_snow_storm_params[_snowstorm_ref[i_event]]) )
                # self._fillCache(new_event) <-- we need new weighters to do this 

            target_hist.add(new_event, new_event.energy, new_event.zenith, new_event.azimuth, new_event.topology, new_event.year)

            if i_event%25000==0:
                Logger.Log("Loaded {} Events so far".format(i_event))
            i_event+=1 
        data.close()

    def _loadFile_sterile(self, which_file, target_hist, is_mc):
        """
        used to load the "sterile" tag MC
        """
        Logger.Log("Opening {}".format(which_file))
        data = h5.File(which_file, 'r')
        i_event = 0

        # by doing this, we load the whole dataset into memory at once. Otherwise it loads only chunks, and slooooows down this process tremendously 
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
                contents = glob(os.path.join(filename, "*.hdf5"))
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
