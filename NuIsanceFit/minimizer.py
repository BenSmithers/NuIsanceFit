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

from param import paramPoint
from logger import Logger 
from weighter import Weighter
from weighter import WeighterMaker as simWeighterMaker

from event import Event

from math import log, lgamma, log1p
#lgamma - log of gamma function
#log1p - log(1+x)

import numpy as np
from numbers import Number 


def gammaPriorPoissonLikelihood(k, alpha, beta):
    val = alpha*log(beta)
    val += lgamma(k+alpha)
    val += -lgamma(k+1)
    val += -(k+alpha)*log1p(beta)
    val += -lgamma(alpha)
    return val

def poissonLikelihood(dataCount, lambdaVal):
    if lambdaVal==0:
        return( 0 if dataCount==0 else -np.inf )
    else:
        sum_val = lambdaVal + lgamma(dataCount+1)
        return(dataCount*log(lambdaVal) - sum_val)


def SAYLikelihood(k, w_sum, w2_sum):
    if (w_sum<=0 or w2_sum<0):
        return( 0 if k==0 else -np.inf)
    if (w2_sum==0):
        return poissonLikelihood(k, w_sum)

    if (w_sum==0):
        if (k==0):
            return 0.0
        else:
            return -np.inf

    alpha = w_sum*w_sum/w2_sum + 1.0
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
        Take in the 
        """
        self._minimum = None # only non-None type when minimized 
   
        self._obshist = None
        self._simhist = None 

        self._simWeighterMaker = None
        self.setSimWeighterMaker( simWeighterMaker )
        self._dataWeighterMaker = None

        self._simulation = None # list/tuple of data 
        self._observation = None # singular histogram of data

        self._seeds = None # list of seeds

        self._llhfunc = None
        self.setLikelihoodFunc( SAYLikelihood )

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
        return self._simweighter

    def setDataWeighterMaker(self, weighter):
        pass 
    @property
    def dataWeighterMaker(self):
        return self._dataweighter

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

    def _likelihoodCore(self):
        pass

    def _evaluateLikelihood(self):
       pass 

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

    
