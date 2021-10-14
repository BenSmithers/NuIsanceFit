import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.style.use("/home/benito/software/cascade/cascade/cascade.mplstyle")

import os
from NuIsanceFit.fit import Fitter
from NuIsanceFit import steering
from NuIsanceFit.param import ParamPoint
import time


fif = Fitter(steering)

e_grad, c_grad = fif.getGradients()
print("Energy {}".format(e_grad))
print("cth {}".format(c_grad))

#e_grad /= np.sum(e_grad)
#c_grad /= np.sum(c_grad)

edges = fif.get_edges()

# plot the energy edges 
plt.pcolormesh( edges[0], edges[0], e_grad, vmin=-0.02, vmax=0.02, cmap='cividis')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$E_{\nu}^{reco}$ [Gev]",size=14)
plt.ylabel(r"$E_{\nu}^{reco}$ [Gev]",size=14)
cbar = plt.colorbar()
cbar.set_label("Covariance",size=16)
#plt.tight_layouts()
plt.show()


# plot the energy edges 
plt.pcolormesh( edges[1], edges[1], c_grad, vmin=-0.01,vmax=0.01, cmap='cividis')
plt.xlabel(r"$\cos\theta_{z}^{reco}$",size=14)
plt.ylabel(r"$\cos\theta_{z}^{reco}$",size=14)
cbar = plt.colorbar()
cbar.set_label("Covariance",size=16)
#plt.tight_layouts()
plt.show()

