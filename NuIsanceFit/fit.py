from numpy.core.multiarray import bincount
from NuIsanceFit.param import ParamPoint
from NuIsanceFit.data import Data
from NuIsanceFit.minimizer import llhMachine
from NuIsanceFit import Logger

from NuIsanceFit.nuthread import ThreadManager

import numpy as np

class Fitter:
    def __init__(self, steering):
        self._data = Data(steering)
        self._llhobj = llhMachine(self._data, steering)

    def get_centers(self):
        return self._data.simulation.centers

    def get_expectation(self,params):
        if not isinstance(params, ParamPoint):
            Logger.Fatal("Arg 'param' must be {}, not {}".format(ParamPoint, type(params)))

        self._llhobj._simWeighter.configure(params)


        sim = self._data.simulation

        shape = (len(sim.edges[0])-1, len(sim.edges[1])-1)
        result = np.zeros(shape=shape)

        dims = tuple([len(sim.edges[i])-1 for i in range(len(sim.edges))])
        
        #def getter

        for i in range(dims[0]):
            for j in range(dims[1]):
                # let's just fix the other dimensions for now! 
                for event in sim[i][j][0][0][0]:
                    result[i][j]+=self._llhobj._simWeighter(event)

        return result



