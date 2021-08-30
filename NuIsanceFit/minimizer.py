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

from NuIsanceFit.param import ParamPoint, PriorSet
from NuIsanceFit.logger import Logger 
from NuIsanceFit.weighter import SimWeighter, SimpleDataWeighter
from NuIsanceFit.histogram import bHist, eventBin, flatten, transpose
from NuIsanceFit.nuthread import ThreadManager
from NuIsanceFit.event import Event
from NuIsanceFit.data import Data

from math import log, lgamma, log1p
#lgamma - log of gamma function
#log1p - log(1+x)

import numpy as np
from numbers import Number 

from torch import Tensor, optim

from torch._C import Value
import param


"""
These three likelihood functions are from Phystools, they were defined over here
    https://arxiv.org/abs/1901.04645
see the effective likelihood material
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
    """
    This takes the
        k - observation amount 
        w - sum of weights in the bin
        w - sum of *squares* of weights in the bin (sort of)
    """
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
_hist_type = bHist
_bin_type = eventBin

class llhMachine:
    def __init__(self, data_obj, steering):
        """
        Machine for calculating the likelihood of an observation, given simulation, for a set of parameters  
        """
        
        self._steering = steering
        self._minimum = None # only non-None type when minimized 
   
        self._prior = PriorSet()
        self._includePriors = True

        self._dataWeighter = SimpleDataWeighter()
        self._simWeighter = SimWeighter(self._steering)

        self._weighttype = float
        self._dataWeighterMaker = None

        self._simulation = data_obj.simulation
        self._observation = data_obj.data

        self._seeds = None # list of seeds

        self._llhfunc = None
        self._llhdtype = float

        # we set this one aside to make sure the params we're passing aruond don't add or lose keys 
        # also verifies the order is unchanged
        self._templated_params=[]

        self.setLikelihoodFunc( SAYLikelihood )

    # ========================== Getters and Setters ===============================

    @property
    def prior(self):
        return self._prior
    @property 
    def simulation(self):
        return self._simulation
    @property
    def observation(self):
        return self._observation 


    def setDataWeighterMaker(self, weighter):
        self._dataWeighterMaker = weighter
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
        if not isinstance(self._simWeighter, SimWeighter):
            Logger.Fatal("SimWeighter should be {}, not {}".format(SimWeighter, type(self._simWeighter)), TypeError)
        
        #TODO  need to validate the other things!

    # ================================ Parts that actually do things =====================

    def _likelihoodCore(self, this_obs:eventBin, this_sim: eventBin):
        """
        This is the function called by the threader. 

        It gets passed two eventBins, one for sim one for obs
        """
        llh = self._llhdtype(0.0)

        if len(this_sim)==0: # no expectation
            return llh

        # get total weight of events oveserved in this bin
        observationAmount = sum([self._dataWeighter(event) for event in this_obs])

        # get the expectation here 
        expectationWeights = [self._weighttype() for event in this_sim]
        expectationSqWeights = [self._weighttype() for event in this_sim]

        n_events = 0
        for i_event in range(len(this_sim)):
            event = this_sim[i_event]
            w = self._simWeighter(event)
            assert(w>=0)
            w2 = w*w
            assert(w2>=0)
            n_events += event.num_events
            expectationWeights[i_event] = w
            expectationSqWeights[i_event] = w2/event.num_events

            if not event.is_mc:
                Logger.Fatal("Doing simulation weighting on Data! Something is very wrong")

            if np.isnan(w) or np.isinf(w):
                Logger.Warn("Bad Weight {} for Event {}".format(w, event))
            if np.isnan(w) or np.isinf(w):
                Logger.Warn("Bad WeightSq {} for Event {}".format(w, event))
            if np.isnan(event.num_events):
                Logger.Warn("Bad num_events {} for Event {}".format(event.num_events, event))

        # using numpy sum since python default sum only works on number-likes
        # this should work on anything with a defined "+" operation 
        w_sum = np.sum(expectationWeights)
        w2_sum = np.sum(expectationSqWeights)
        if (observationAmount>0 and w_sum<=0):
            Logger.Warn("Bad Bin! Printing Weights!")
            for w in expectationWeights:
                Logger.Warn("    {}".format(w))
            Logger.Warn("Events")
            for event in this_sim:
                Logger.Warn("    {}".format(event))
            return llh

        this_llh = self.likelihoodFunc(observationAmount, w_sum, w2_sum, self._llhdtype)
        if np.isnan(this_llh) or np.isinf(this_llh):
            Logger.Warn("Bad llh {} found! From obs {}, w_sum {}, w_2 {}".format(this_llh, observationAmount, w_sum, w2_sum))

        llh += this_llh
        
        return llh

    def _evaluateLikelihood(self, _params:Tensor)->Tensor:
        """
        Here, we take our binned histograms. We separate them along the first axis, flatten along the other four axes. 
        Then we use the core function 
        """
        params=self.convert_tensor_to_params(_params)
        Logger.Trace("Making sim/data Weighters")
        self._dataWeighter = SimpleDataWeighter()
        self._simWeighter.configure(params.as_dict())
 
        # If this is outside our valid parameter space (and we're using the priors), BAIL OUT
        if self._includePriors:
            llh = self.prior(params)
            if np.isnan(llh):
                Logger.Warn("nan Prior!")
                return(-np.inf)    
        else:
            llh = self._llhdtype(0.0)

        # we flatten these out into big stacks of bins 
        # may want to change this a bit if it turns out the first axis only has a length ~2-6 
        if len(self.observation)<10:
            Logger.Warn("Observation and Simulation axis0 is only length {}".format(len(self.observation)))


        # this is a deterministic process, so the bin ordering shouldn't be affected 
        flat_obs = [flatten(axis0) for axis0 in self.observation]
        flat_sim = [flatten(axis0) for axis0 in self.simulation]
        pairs = [[flat_obs[i], flat_sim[i]] for i in range(len(flat_obs))]
       
        Logger.Trace("Starting up Threader for LLH calculation")
        # now prepare these into pairs of stacks for the threading
        llh += sum([self._likelihoodCore(pair) for pair in pairs])
        
        return(Tensor([-llh]))

    def minimize(self):
        """
        Finds the minimized point

        Need to use a minimization driver 
        """

        if self._minimum is None:
            initial_guess = ParamPoint()
        else:
            initial_guess = self._minimum
        
        initial_model = self.convert_params_to_tensor(initial_guess)
        optimizer = optim.lbfgs(params=initial_guess.as_dict(), lr=0.01, max_iter=20000)

        def closeure():
            optimizer.zero_grad()
            loss = self._evaluateLikelihood(initial_model)
            loss.backward()
            return loss
        optimizer.step(closeure)


        return ParamPoint()

    def convert_params_to_tensor(self, params:ParamPoint):
        """
        Converts the params into a pytorch Tensor for interfacing with PyTorch 
        """
        keys = params.valid_keys()
        # set the template, check it against the other one otherwise 
        if len(self._templated_params) == 0:
            self._templated_params=keys
        else:
            # this might slow things down, so we may only want to do a length-check 
            if not keys==self._templated_params:
                Logger.Fatal("It looks like the params changed at some point. From {} to {}".format(self._templated_params, keys),ValueError)

        return Tensor([params[key] for key in keys])

    def convert_tensor_to_params(self, tensor:Tensor):
        if len(self._templated_params)==0:
            Logger.Fatal("Cannot convert {} to params object, don't have a template yet.".format(tensor), ValueError)
        if not len(tensor)==len(self._templated_params):
            Logger.Fatal("tensor {} doens't match templated keys {}".format(tensor, self._templated_params), ValueError)

        #inline build the dict, pass it as the kwargs to the ParamPoint constructor
        return ParamPoint(**{self._templated_params[i]:tensor[i] for i in range(len(self._templated_params))} )


    def __call__(self, params):
        """
        Gets the likelihood of the provided parameter point
        """
        if not isinstance(params, ParamPoint):
            Logger.Fatal("Cannot evaluate LLH for object of type {}".format(type(params)), TypeError)

        return self._evaluateLikelihood(self.convert_params_to_tensor(params))

