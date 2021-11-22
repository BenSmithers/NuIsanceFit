
import torch
from torch import tensor
from torch import lgamma as torchlgamma
from torch import log as torchlog
from torch import log1p as torchlog1p
from torch import nn
from torch.functional import F as tFunctinal

from NuIsanceFit.data import Data
from NuIsanceFit.param import ParamPoint
from NuIsanceFit.histogram import bHist, eventBin
from NuIsanceFit.weighter import SimReWeighter, SimpleDataWeighter


import numpy as np

def torchGammaPriorPoissonLikelihood(k:tensor, alpha:tensor, beta:tensor)->tensor:
    val = alpha*torchlog(beta)
    val += torchlgamma(k+alpha)
    val += -torchlgamma(k+1)
    val += -(k+alpha)*torchlog1p(beta)
    val += -torchlgamma(alpha)
    return val


def torchPoissonLikelihood(dataCount:tensor, lambdaVal:tensor)->tensor:
    total = dataCount*torchlog(lambdaVal) - lambdaVal  - torchlgamma(dataCount+1)
    # where both are zero, we get nans, 

    # this thing filters out the nans, and swaps them with zeros 
    total[total!=total]=0.0
    return total


def SAYLikelihood(k:tensor, w_sum:tensor, w2_sum:tensor)->tensor:
    """
    This takes the
        k - observation amount 
        w - sum of weights in the bin
        w - sum of *squares* of weights in the bin (sort of)
    """

    alpha = w_sum*w_sum/w2_sum + 1. 
    beta = w_sum/w2_sum

    retval = torchGammaPriorPoissonLikelihood(k, alpha, beta)
    # all these conditionals! Yuck! 
    retval[(w_sum<=0. or w2_sum<0.) and k==0.] = 0.
    retval[(w_sum<=0. or w2_sum<0.) and k!=0.] = torch.inf
    retval[w2_sum==0.] = torchPoissonLikelihood(k, w_sum)
    retval[w_sum==0. and k==0] = 0.
    retval[w_sum==0. and k!=0] = -torch.inf
    return retval


_hist_type = bHist
_bin_type = eventBin



class Model(nn.Module):
    def __init__(self, data_obj:Data, steering:dict) -> None:
        super().__init__()

        self._simulation = data_obj.simulation
        self._observation = data_obj.data

        self._dataWeighter = SimpleDataWeighter()
        self._SimReWeighter = SimReWeighter(self._steering)

        obs_shape = np.shape(self._observation)
        self._k_vals = tensor(np.zeros(shape=obs_shape), requires_grad=True)
        for i_e in range(obs_shape[0]): #energy
            for i_c in range(obs_shape[1]): # costheta
                for i_a in range(obs_shape[2]): # azimuth
                    for i_t in range(obs_shape[3]): # topology 
                        for i_y in range(obs_shape[4]): # year/time
                            self._k_vals = torch.sum([self._dataWeighter(event) for event in self._observation[i_e,i_c,i_a,i_t,i_y]])
                    
        # load in the parameters
        self._templated_params=ParamPoint().valid_keys
        _params = ParamPoint(reseed=False)
        self._params = nn.Parameter(tensor([_params[key] for key in self._templated_params], requires_grad=True))



    def _extract_likelihood(self)->tuple:
        """
        Returns the w_sum and w2_sum in each bin
        """
        # hard-coding this, I don't like that 
        shape = np.shape(self._simulation.fill)
        assert(len(shape == 5))

        weights = tensor(np.zeros(shape=shape))
        weightsq = tensor(np.zeros(shape=shape))

        for i_e in range(shape[0]): #energy
            for i_c in range(shape[1]): # costheta
                for i_a in range(shape[2]): # azimuth
                    for i_t in range(shape[3]): # topology 
                        for i_y in range(shape[4]): # year/time
                            for event in self._simulation[i_e,i_c,i_a,i_t,i_y]:
                                w = self._SimReWeighter(event)
                                weights[i_e,i_c,i_a,i_t,i_y] += w
                                weightsq[i_e,i_c,i_a,i_t,i_y] += w*w/event.num_events
       
        return SAYLikelihood(self._k_vals, weights, weightsq)

    def _eval_llh(self):
        llh_data = self._extract_likelihood()

    def forward(self):
        """
            Evaluate the current state of things given the currently saved set of parameters


        """
        pass
