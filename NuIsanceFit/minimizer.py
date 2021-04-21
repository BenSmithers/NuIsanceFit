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

class llhMachine:
    def __init__(self):
        """
        Take in the 
        """
        self._minimum = None # only non-None type when minimized 
    
        self._simweighter = None
        self._dataweighter = None
        self._simulation = None


    def validate(self):
        if not isinstance(self._simweighter, Weighter):
            Logger.Fatal("SimWeighter should be {}, not {}".format(Weighter, type(self._simweighter)), TypeError)
        
        #TODO  need to validate the other things!

    def likelihoodCore(self):
        pass

    def evaluateLikelihood(self):
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

    
