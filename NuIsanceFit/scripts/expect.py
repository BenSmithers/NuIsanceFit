import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import os
from NuIsanceFit.fit import Fitter
from NuIsanceFit import steering
from NuIsanceFit.param import ParamPoint
import time
fif = Fitter(steering)

edges = fif.get_centers()

# np.save("what.npy", [expect, edges])

def weplot(pp):
    start = time.time()
    expect = fif.get_expectation(pp)
    end = time.time()
    print("Took {} seconds to get expectation".format(end-start))
    plt.clf()
    plt.pcolormesh( edges[1], edges[0],np.log10(expect))
    plt.colorbar()
    plt.xlim([-1,0.2])
    plt.yscale('log')
    plt.xlabel("Cos(Zenith)", size=14)
    plt.ylabel("Reco Energy [GeV]",size=14)
    plt.show()
    plt.close()

default = ParamPoint()
weplot(default)
#print("Try different CRDeltaGamma")
#default.set("CRDeltaGamma", 0.2)
#weplot(default)

#default.set("CRDeltaGamma", 0.0)
#default.set("icegrad0", 1.1)
#weplot(default)

import sys 
sys.exit()

print("And let's see how the evaluat llh thing goes")
t1 = time.time()
fif._llhobj._evaluateLikelihood(default)
t2 = time.time()
print("Took {}".format(t2-t1))
if False:
    data = np.load("what.npy", allow_pickle=True)
    edges = data[1]
    expect = data[0]


