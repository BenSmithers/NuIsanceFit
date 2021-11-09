"""
We'll use this script to create utilities not dependent on anything else in particular 
"""

import numpy as np
from math import pi

import matplotlib
import matplotlib.pyplot as plt 

def get_color(n, colormax=3.0, cmap="viridis"):
    """
    I use this script to get colors out of a colormarp. Super useful for plotting 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)

def shift_cmap(cmap, frac):
    """Shifts a colormap by a certain fraction.
    Keyword arguments:
    cmap -- the colormap to be shifted. Can be a colormap name or a Colormap object
    frac -- the fraction of the colorbar by which to shift (must be between 0 and 1)
    """
    N=256
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    n = cmap.name
    x = np.linspace(0,1,N)
    out = np.roll(x, int(N*frac))
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f'{n}_s', cmap(out))
    return new_cmap

class NewPhysicsParams:
    """
    This container class will hold the new physics stuff! 
        TODO develop something that can convert filenames and NPP into unique filenames and vice-versa
    """
    def __init__(self, **kwargs):
        self._accepted_kwargs = ["theta14", "theta24", "theta34", "deltam41"]
        self._n_neut = 4

         # only use this for the sterile params, ignore the 3-nu entries
         #      also, this'd be diagonal... but whatever 
        self._thetas = np.zeros(shape=(self._n_neut, self._n_neut))
        self._deltas = np.zeros(shape=(self._n_neut, 1))

        # check wargs
        for kwarg in kwargs:
            if kwarg in self._accepted_kwargs:
                pass
            else:
                raise ValueError("Unexpected kwarg: {}".format(kwarg))

    def set_mixing_angle(self, value:float, i1:int, i2=4):
        """
        Use this for setting the mixing angles! Currently only i2==4 is configured, so a 3+1 sterile nu model 

        If you're adding more nus, you may want to use the getattr and setattr functions; they aren't as fast
        """
        if i2<i1:
            return self.set_mixing_angle(value, i2,i1)
        else:
            if i2!=4:
                raise NotImplementedError("Currently only 3+1 neutrino models are supported")
            if i1<=0 or i1>self._n_neut:
                raise ValueError("Neutrino index shoudl be between {} and {}, i1 is {}".format(1, self._n_neut, i1))
            if i2<=0 or i2>self._n_neut:
                raise ValueError("Neutrino index shoudl be between {} and {}, i2 is {}".format(1, self._n_neut, i2))
            if value<0 or value>pi:
                raise ValueError("Invalid mixing angle found: {}".format(value))
            if i1==i2:
                raise ValueError("i1==i2, this must've been a mistake?")

            self._thetas[i1-1][i2-1] = value

    def get_mixing_angle(self, i1:int, i2:int)->float:
        """
        Generic function for accessing the neutrino mixing angles
            i1,i2 are the indices. Must be between [1, n_neut]
            where n_neut is the number of neutrinos 
        """
        if i1<=0 or i1>self._n_neut:
            raise ValueError("Neutrino index shoudl be between {} and {}, i1 is {}".format(1, self._n_neut, i1))
        if i2<=0 or i2>self._n_neut:
            raise ValueError("Neutrino index shoudl be between {} and {}, i2 is {}".format(1, self._n_neut, i2))
        if i2==i1:
            raise ValueError("i2==i1, this must've been a mistake?")

        return self._thetas[i1-1][i2-1] 

    @property
    def theta14(self):
        return self._thetas[0][3]
    @property
    def theta24(self):
        return self._thetas[1][3]
    @property
    def theta34(self):
        return self._thetas[2][3]
    @property
    def dm2(self):
        return self._deltas[3][0] # Delta_41^2

    def _parse_kwarg(self, kwarg):
        pass

    def __repr__(self):
        return("Sterile-nu Params Object")

    def __str__(self):
        return