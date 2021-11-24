
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
from NuIsanceFit.torchweighter import SimReWeighter


import numpy as np

def torchGammaPriorPoissonLikelihood(k:tensor, alpha:tensor, beta:tensor)->tensor:
    val = alpha*torchlog(beta)
    val += torchlgamma(k+alpha)
    val += -torchlgamma(1.+k)
    val += -(k+alpha)*torchlog1p(beta)
    val += -torchlgamma(alpha)
    return val


def torchPoissonLikelihood(dataCount:tensor, lambdaVal:tensor)->tensor:
    total = dataCount*torchlog(lambdaVal) - lambdaVal -torchlgamma(1.+dataCount)
    # where both are zero, we get nans, 

    # this thing filters out the nans, and swaps them with zeros 
    total[total!=total]=0.0
    print(np.shape(total))
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
    retval[torch.logical_and(torch.logical_or(w_sum<=0., w2_sum<0.), k==0.)] = 0.
    retval[torch.logical_and(torch.logical_or(w_sum<=0., w2_sum<0.), k!=0.)] = torch.inf

    #retval[(w2_sum==0)] = torchPoissonLikelihood(k, w_sum)
    torch.where(w2_sum==0., w2_sum,  torchPoissonLikelihood(k, w_sum))
    retval[torch.logical_and(w_sum==0.,  k==0)] = 0.
    retval[torch.logical_and(w_sum==0.,  k!=0)] = -torch.inf
    return retval


_hist_type = bHist
_bin_type = eventBin



class GolemModel(nn.Module):
    def __init__(self, data_obj:Data) -> None:
        super().__init__()

        self._observation = data_obj.data
        self._steering = data_obj.steering

        self._SimReWeighter = SimReWeighter(data_obj)

        obs_shape = np.shape(self._observation)
        self._k_vals = tensor(np.zeros(shape=obs_shape))
        for i_e in range(obs_shape[0]): #energy
            for i_c in range(obs_shape[1]): # costheta
                for i_a in range(obs_shape[2]): # azimuth
                    for i_t in range(obs_shape[3]): # topology 
                        for i_y in range(obs_shape[4]): # year/time
                            self._k_vals[i_e,i_c,i_a,i_t,i_y] = sum([1. for event in self._observation[i_e][i_c][i_a][i_t][i_y]])
                    
        # load in the parameters
        self._templated_params=ParamPoint().valid_keys
        _params = ParamPoint(reseed=False)
        self.weights = nn.Parameter(tensor([_params[key] for key in self._templated_params]))

    def get_llh(self)->tensor:
        """
        Returns overall likelihood!
        """
        weights = self._SimReWeighter(self.weights)
        weightsq = weights*weights
        weightsq = torch.sum(weightsq.to_dense(), 5)
        weights = torch.sum(weights.to_dense(), 5)
       
        assert(np.shape(self._k_vals) == np.shape(weights))

        return -1*torch.sum(SAYLikelihood(self._k_vals, weights, weightsq))


    def forward(self):
        """
            Evaluate the current state of things given the currently saved set of parameters


        """
        pass
