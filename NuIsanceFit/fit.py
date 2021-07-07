from numpy.core.multiarray import bincount
from NuIsanceFit.param import ParamPoint
from NuIsanceFit.data import Data
from NuIsanceFit.minimizer import llhMachine
from NuIsanceFit import Logger

from NuIsanceFit.nuthread import ThreadManager

import numpy as np
from time import time

from multiprocessing import Pool as ThreadPool
from multiprocessing.dummy import Pool as ThreadPool

class Fitter:
    def __init__(self, steering):
        self._data = Data(steering)
        self._llhobj = llhMachine(self._data, steering)

    def get_centers(self):
        return self._data.simulation.centers

    def get_expectation(self,params):
        if not isinstance(params, ParamPoint):
            Logger.Fatal("Arg 'param' must be {}, not {}".format(ParamPoint, type(params)))

        self._llhobj._simWeighter.configure(params.as_dict())


        sim = self._data.simulation

        shape = (len(sim.edges[0])-1, len(sim.edges[1])-1)
        result = np.zeros(shape=shape)

        dims = tuple([len(sim.edges[i])-1 for i in range(len(sim.edges))])

        def wrapper(dataobj):
            result = [0 for j in range(len(dataobj))]
            for j in range(len(dataobj)):
                for event in dataobj[j][0][0][0]:
                    result[j] += self._llhobj._simWeighter(event)
            return result

        #for i in range(dims[0]):
        #    for j in range(dims[1]):
        #        # let's just fix the other dimensions for now! 
        #        for event in sim[i][j][0][0][0]:
        #            result[i][j]+=self._llhobj._simWeighter(event)

        #stepsize = int(dims[0]/4)
        #subsets = [sim.fill[stepsize*i:stepsize*(i+1)] for i in range(4)]
        #subsets[3] = sim.fill[3*stepsize:]

        pool = ThreadPool(1)
        result = pool.map(wrapper, sim)        


        #result = ThreadManager(wrapper, subsets)


        return result



