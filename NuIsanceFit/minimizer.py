"""
Here we will build up the logic for making a machine that minimizes a set of parameters according to Data and MC

Look into 
Event
HistType
simpleLocalDataWeighterConstructor
WeighterMaker
CPrior
SAYLikelihood
"""

from param import paramPoint, PriorSet
from logger import Logger 
from weighter import Weighter
from weighter import WeighterMaker as simWeighterMaker
from nuthread import ThreadManager

from event import Event

from math import log, lgamma, log1p
#lgamma - log of gamma function
#log1p - log(1+x)

import numpy as np
from numbers import Number 


"""
These three likelihood functions are from Phystools, don't have much to say about them. 
"""
def gammaPriorPoissonLikelihood(k, alpha, beta):
    val = alpha*log(beta)
    val += lgamma(k+alpha)
    val += -lgamma(k+1)
    val += -(k+alpha)*log1p(beta)
    val += -lgamma(alpha)
    return val

def poissonLikelihood(dataCount, lambdaVal, dtype=float):
    zero = dtype()
    one = dtype(1.0)
    if lambdaVal==zero:
        return( zero if dataCount==zero else -np.inf )
    else:
        sum_val = lambdaVal + lgamma(dataCount+1)
        return(dataCount*log(lambdaVal) - sum_val)


def SAYLikelihood(k, w_sum, w2_sum, dtype=float):
    zero = dtype()
    one = dtype(1.0)

    if (w_sum<=zero or w2_sum<zero):
        return( zero if k==zero else -np.inf)
    if (w2_sum==zero):
        return poissonLikelihood(k, w_sum, dtype)

    if (w_sum==zero):
        if (k==zero):
            return zero
        else:
            return -np.inf

    alpha = w_sum*w_sum/w2_sum + one 
    beta = w_sum/w2_sum

    return gammaPriorPoissonLikelihood(k,alpha,beta)


"""
make it 
setEvaluatinoThreadCount

we need to do evaluateLikelihood

for minLLH
    setSeed
    DoFitLBFGSB (minimizer.minimize


The do fit thing uses our overall likelihood problem and the LBFGSB_Driver minimizer
"""
_event_type = Event
_hist_type = np.ndarray
_bin_type = np.ndarray
_array_dims = 5
def _verify_array_shape(array):
    """
    This function makes sure that whatever we're sending in as our simulation or observation is all formatted correctly.
    This ensures we know the exact kinda data we're working with 
    """
    if not isinstance(array, _hist_type):
        Logger.Warn("Incorrect Histogram Type: {}".format(type_arry))
        return False

    shape = np.shape(array)
    if len(shape)!=_array_dims:
        Logger.Warn("Incorrect shape: {}".format(shape))
        return False

    # Guaranteed 5-d array
    # each bin needs to hold bins
    if array.dtype!=_bin_type:
        Logger.Warn("Incorrect bin type: {}".format(array.dtype))
        return False

    # the bins should be iterable... 
    for entry in array.flat:
        if not isinstance(entry, _bin_type):
            Logger.Warn("This should be unreachable... since this should've already been found! ")
            return False
        if entry.dtype!=_event_type:
            Logger.Warn("Bin Event type should be {}, found {}".format(_event_type, entry.dtype))
            return False

    return True

class llhMachine:
    def __init__(self):
        """
        Machine for calculating the likelihood of an observation, given simulation, for a set of parameters  
        """
        self._minimum = None # only non-None type when minimized 
   
        self._prior = PriorSet()
        self._includePriors

        self._obshist = None
        self._simhist = None 

        self._dataWeighter = None
        self._simWeighter = None

        self._weighttype = float
        self._simWeighterMaker = None
        self.setSimWeighterMaker( simWeighterMaker )
        self._dataWeighterMaker = None

        self._simulation = None # list/tuple of data 
        self._observation = None # singular histogram of data

        self._seeds = None # list of seeds

        self._llhfunc = None
        self._llhdtype = float
        self.setLikelihoodFunc( SAYLikelihood )

    # ========================== Getters and Setters ===============================
    @property
    def prior(self):
        return self._prior


    # OG GolemFit uses a 5-dimensional array of bins. Bins contained events. So let's do that too
    def setSimulation(self, simulation):
        if not _verify_array_shape(simulation):
            Logger.Fatal("Cannot configure with this simulation histogram.")
        else:
            self._simulation = simulation
    @property 
    def simulation(self):
        return self._simulation

    def setObservation(self, observation):
        if not _verify_array_shape(observation):
            Logger.Fatal("Cannot configure with this observation histogram")
        else:
            self._observation = observation
    @property
    def observation(self):
        return self._observation 


    def setSimWeighterMaker(self, weighter):
        pass
    @property
    def simWeighterMaker(self):
        return self._simWeighterMaker

    def setDataWeighterMaker(self, weighter):
        pass 
    @property
    def dataWeighterMaker(self):
        return self._dataWeighterMaker

    def setLikelihoodFunc(self, llhfunc):
        if not hasattr(llhfunc, "__call__"):
            Logger.Fatal("LikelihoodFunc needs to be callable",TypeError)

        self._llhfunc = llhfunc
    @property
    def likelihoodFunc(self):
        return self._llhfunc


    def _validate(self):
        if not isinstance(self._simweighter, Weighter):
            Logger.Fatal("SimWeighter should be {}, not {}".format(Weighter, type(self._simweighter)), TypeError)
        
        #TODO  need to validate the other things!

    # ================================ Parts that actually do things =====================

    def _likelihoodCore(self, pairs):
        """
        This is the function called by the threader. 

        It gets passed a stack of observation and simulation bins! 
        """
        llh = self._llhdtype(0.0)

        for entry in pairs:
            if len(entry)!=2:
                Logger.Fatal("Found unexpected length for sim/obs pair, {}".format(len(entry)), ValueError)
            # these are bin objects, they have events 
            this_obs = entry[0] 
            this_sim = entry[1]

            # get total weight of events oveserved in this bin
            observationAmount = sum([self._dataWeighter(event) for event in this_obs])

            # get the expectation here 
            expectationWeights = [self._weightType() for event in this_sim]
            expectationSqWeights = [self._weightType() for event in this_sim]

            n_events = 0
            for i_event in range(len(his_sim)):
                event = this_sim[i_event]
                w = self.simWeighter(event)
                assert(w>=0)
                w2 = w*w
                assert(w2>=0)
                n_events += event.num_events
                expectationWeights[i_event] = w
                expectationSqWeights[i_event] = w2

                if np.isnan(w):
                    Logger.Warn("Bad Weight {} for Event {}".format(w, event))
                if np.isnan(event.num_events):
                    Logger.Warn("Bad num_events {} for Event {}".format(event.num_events, event))

            # using numpy sum since python default sum only works on number-likes
            # this should work on anything with a defined "+" operation 
            w_sum = np.sum(expectationWeights)
            w2_sum = np.sum(expectationSqWeights)

            llh += self.likelihoodFunc(observationAmount, w_sum, w2_sum, self._llhdtype)
        
        return llh

    def _evaluateLikelihood(self, params):
        """
        Here, we take our binned histograms. We separate them along the first axis, flatten along the other four axes. 
        Then we use the core function 
        """
        Logger.Thread("Making sim/data Weighters")
        self._dataWeighter = self.dataWeighterMaker(params)
        self._simWeighter = self.simWeighterMaker(params)
 
        # If this is outside our valid parameter space, BAIL OUT
        prior_param = self.prior(params)
        if np.isnan(prior_param):
            return(-np.inf) 

        # we flatten these out into big stacks of bins 
        # may want to change this a bit if it turns out the first axis only has a length ~2-6 
        if len(self.observation)<10:
            Logger.Warn("Observation and Simulation axis0 is only length {}".format(len(self.observation)))

        llh = prior_param if self._includePriors else self._llhdtype(0.0)

        # this is a deterministic process, so the bin ordering shouldn't be affected 
        flat_obs = np.array([axis0.flatten() for axis0 in self.observation])
        flat_sim = np.array([axis0.flatten() for axis0 in self.simulation])
       
        Logger.Thread("Starting up Threader for LLH calculation")
        # now prepare these into pairs of stacks for the threading     
        llh += ThreadManager( self._likelihoodCore, np.transpose([flat_obs, flat_sim]))
        
        return(llh)

    def minimize(self):
        """
        Finds the minimized point

        Need to use a minimization driver 
        """


        return paramPoint()

    def __call__(self, params):
        """
        Gets the likelihood of the provided parameter point
        """
        if not isinstance(params, paramPoint):
            Logger.Fatal("Cannot evaluate LLH for object of type {}".format(type(params)), TypeError)

    
