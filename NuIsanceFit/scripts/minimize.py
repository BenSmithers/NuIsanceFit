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

print("And let's see how the evaluat llh thing goes")
t1 = time.time()
value = fif.minLLH()
t2 = time.time()
print("Took {}".format(t2-t1))
print( value )

expect = fif.get_expectation(value)
edges = fif.get_centers()

plt.clf()
plt.pcolormesh( edges[1], edges[0],np.log10(expect))
plt.xlim([-1,0.2])
plt.yscale('log')
plt.xlabel("Cos(Zenith)", size=14)
plt.ylabel("Reco Energy [GeV]",size=14)
plt.show()
plt.close()


