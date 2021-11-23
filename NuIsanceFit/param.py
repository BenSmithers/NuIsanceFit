import numpy as np
import os, json
from numbers import Number # parent of all numbers 
from math import sqrt, pi, log
from numpy import random
from numpy.random import rand, randn

from NuIsanceFit.logger import Logger

over_root_two_pi = 1./sqrt(2*pi)

class Prior:
    """
    These objects represent possible prior distributions for the parameters.
    Implementations should derive from this base class and re-implement the initializer and call functions

    These objects return the probabiltity density for a given sampled point in LOG space 
    """
    def __init__(self):
        Logger.Fatal("Need to use derived class", NotImplementedError)
    def __call__(self):
        Logger.Fatal("Need to use derived class", NotImplementedError)
    def inverse(self)->float:
        Logger.Fatal("Need to use derived class", NotImplementedError)

class UniformPrior(Prior):
    """
    A prior which gives uniform probability for the parameter to be anywhere in a fixed range. 
    """
    def __init__(self, minval=-np.inf, maxval=np.inf):
        if not isinstance(minval, Number):
            Logger.Fatal("minval should be {}, not {}".format(float, type(minval)), TypeError)
        if not isinstance(maxval, Number):
            Logger.Fatal("maxval should be {}, not {}".format(float, type(maxval)), TypeError)
        self.min = minval
        self.max = maxval

    def inverse(self)->float:
        return rand()*(self.max - self.min) - self.min

    def __call__(self, x):
        if x<self.min or x>self.max:
            return -np.inf
        else:
            return 0.0

class GaussianPrior(Prior):
    """
    A prior which uses a Gaussian distribution for the prior 
    """
    def __init__(self, mean, stddev):
        if not isinstance(mean, Number):
            Logger.Fatal("mean should be {}, got {}".format(float, type(mean)), TypeError)
        if not isinstance(stddev, Number):
            Logger.Fatal("stddev should be {}, got {}".format(float, type(stddev)), TypeError)

        self.mean = mean
        self.stddev = stddev
        self.norm = over_root_two_pi/stddev
        
    def __call__(self, x):
        if self.norm==0.0:
            return(0.0)
        z = (x-self.mean)/self.stddev 
        return log(self.norm)-(z*z)/2 

    def inverse(self) -> float:
        if np.isinf(self.stddev):
            return self.mean
        else:
            return self.stddev*randn() + self.mean

class LimitedGaussianPrior(Prior):
    def __init__(self, mean, stddev, minval, maxval):
        # these two will enforce type-ing! 
        self.limits = UniformPrior(minval, maxval)
        self.prior = GaussianPrior(mean, stddev)
        if mean <minval or mean>maxval:
            Logger.Warn("Making Limited Gaussian prior where the mean {} is outside the bounds ({}, {})".format(mean, minval, maxval))

    def __call__(self,x):
        return self.limits(x) + self.prior(x)

    def inverse(self) -> float:
        max_iter = 10000

        counter = 0
        value = self.limits.min
        while (value<=self.limits.min or value>=self.limits.max):
            value = self.prior.inverse()
            counter += 1 
            if counter >= max_iter:
                print("mean {}, stdev {}, min {}, max {}".format(self.prior.mean, self.prior.stddev, self.limits.min, self.limits.max ))    
                raise ValueError("Infinite loop caught, killed")
        return value 

class Gaussian2DPrior(Prior):
    """
    A 2D Gaussian prior 
    """
    def __init__(self, mean0, mean1, stddev0, stddev1, correlation):
        if not isinstance(mean0, Number):
            Logger.Fatal("Expected {} for mean0, got {}".format(float, type(mean0)), TypeError)
        if not isinstance(mean1, Number):
            Logger.Fatal("Expected {} for mean1, got {}".format(float, type(mean1)), TypeError)
        if not isinstance(stddev0, Number):
            Logger.Fatal("Expected {} for stddev0, got {}".format(float, type(stddev0)), TypeError)
        if not isinstance(stddev1, Number):
            Logger.Fatal("Expected {} for stddev1, got {}".format(float, type(stddev1)), TypeError)
        if not isinstance(correlation, Number):
            Logger.Fatal("Expecte {} for correlation, got {}".format(float, type(correlation)), TypeError) 

        self.mean0 = mean0
        self.mean1 = mean1 
        self.stddev0 = stddev0
        self.stddev1 = stddev1
        self.correlation = correlation 
        
        if (np.isinf(self.stddev0) or np.isinf(self.stddev1) or np.isnan(stddev0) or np.isnan(stddev1) or np.isnan(correlation)):
            self.lnorm=0.0
            self.prefactor=0.0
        else:
            self.lnorm = log(over_root_two_pi/(self.stddev0*stddev1*sqrt(1.0-self.correlation*self.correlation)))
            self.prefactor = -1.0/(2.0*1.0-correlation*correlation)

    def __call__(self,x0, x1):
        if self.prefactor==0.0:
            return self.lnorm
        else:
            z0 = (x0-self.mean0)/self.stddev0
            z1 = (x1-self.mean1)/self.stddev1
            return(self.lnorm + self.prefactor*(z0*z0 + z1*z1 - 2.0*self.correlation*z0*z1))
    
class LimitedGaussian2DPrior(Prior):
    def __init__(self, mean0, mean1, stddev0, stddev1, correlation, min0, max0, min1,max1):
        self.limits0  = UniformPrior(min0,max0)
        self.limits1  = UniformPrior(min1,max1)
        self.prior = Gaussian2DPrior(mean0, mean1, stddev0, stddev1, correlation)

    def __call__(self, x):
        return self.limits0(x) + self.limits1(x) + self.prior(x)


class Param:
    """
    Basic Fit parameter datatype
    """
    def __init__(self, json_entry):
        """
        This accepts any number of arguments in any order. Right now it only works with 
            name     -   a string representing the name of this parameter 
            center   -   a prior representing the center of the distribution
            width
            max, min
            fit
        """

        # Defaults 
        self._center = 0.0
        self._width  = 1.0
        self._min    = -np.inf
        self._max    = np.inf

        # should this be used in the fit? 
        self._fit = False

        # try assigning these from the keywords passed to this constructor
        for key in json_entry:
            # we only care about a few of these, and we want to do some basic checks

            if (key=="center" or key=="width" or key=="min" or key=="max"):
                if not isinstance(json_entry[key], Number):
                    Logger.Fatal("{} must be {}, received {}".format(key, float, type(json_entry[key])), TypeError)    
            elif key=="fit":
                if not isinstance(json_entry[key], bool):
                    Logger.Fatal("{} should be {}, received {}".format(key, bool, type(json_entry[key])), TypeError)
            else:
                Logger.Warn("Found unrecognized key: {}".format(key))

            setattr(self, "_"+key, json_entry[key])

        if self.width == -1 or self.width==-1.0:
            self._width = np.inf


        # make sure that the assigned values pass a very simple check
        if not self.width <= abs(self.max-self.min):
            Logger.Fatal("Width of distribution is greater than width of allowed values. This must be wrong...", ValueError)

        if self.center<self.min or self.center>self.max:
            Logger.Fatal("Center of distribution is outside the range of allowed values.", ValueError)

        if self.max<self.min:
            Logger.Fatal("Minimum is greater than maximum...", ValueError)

        self._value = self._center

        if not (np.isinf(self.min) or np.isinf(self.max)):
            self.prior = GaussianPrior(self.center, self.width)
        else:
            self.prior = LimitedGaussianPrior(self.center, self.width, self.min, self.max)

    @property 
    def center(self):
        return self._center

    @property 
    def width(self):
        return self._width

    @property 
    def min(self):
        return self._min

    @property 
    def max(self):
        return self._max

    def inverse(self):
        return self.prior.inverse()

    def __str__(self):
        return( "Param: center {}; width {}; min {}, max {}. Will {}Fit\n".format( self.center, self.width, self.min, self.max, "" if self._fit else "not "))

    def __repr__(self):
        return self.__str__()



_params_filepath = os.path.join(os.path.dirname(__file__),"resources","parameters.json")
_params_file = open(_params_filepath, 'r')
_params_raw = json.load(_params_file)

params = {}
for entry in _params_raw.keys():
    Logger.Trace("Read new param: {}".format(str(entry)))
    params[str(entry)] = Param(_params_raw[entry])

class ParamPoint:
    """
    Set of values corresponding to the loaded parameters. Has a value for each parameter

    This is used to represent a point in nuisance parameter space! 
    """
    def __init__(self,reseed=False, **kwargs):
        """
        Create this object, set the attributes corresponding to the centers loaded in from the json file 
        """
        self.valid_keys = list(params.keys())

        self.values = {}

        for key in params.keys():
            # reseed True causes this thing to linger and never finish for some damn reason! What the hell! 
            if reseed:
                self.values[str(key)] = params[key].inverse()
            else:
                self.values[str(key)] = params[key].center

        # now set any non-default arguments 
        for kwarg in kwargs:
            if kwarg in self.values:
                self.values[kwarg] = kwargs[kwarg]

    def set(self, attr, value):
        """
        Sets the attribute "attr" to the value "value"
        """
        if not (attr in self.values):
            Logger.Fatal("Unfamiliar parameter: {}".format(attr))

        # ensure the type is valid 
        if not isinstance( value, type(self.values[attr])):
            Logger.Fatal("Value {} is of invalid type {}, not {}".format(value, type(value), type(self.values[attr])))
    
        self.values[attr]=value

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        if not isinstance(key, str):
            Logger.Fatal("Object {} is not subscriptable with type {}, use {}".format(ParamPoint, type(key), str), TypeError)
        if key in self.values:
            return self.values[key]
        else:
            Logger.Fatal("Object does not have attribute {}".format(key),KeyError)

    def as_dict(self):
        return self.values
    
    def __str__(self):
        value = ""
        for key in self.values:
            value+= "{}: {} from {}\n".format(key, self.values[key], params[key].center)
        return value

class PriorSet:
    """
    Holds the priors, is used to calculate a prior LLH 
    """
    def __init__(self):
        self.astro_cor = 0.70
        self.icegrad_cor = 5.091035738186185100e-02
        self.ice_gradient_joined_prior = Gaussian2DPrior(params["icegrad0"].center, params["icegrad1"].center, 
                                                         params["icegrad0"].width, params["icegrad1"].width,
                                                         self.icegrad_cor)
        #self.astro_correlated_prior    = Gaussian2DPrior(params["astro1Comp2DNorm"].center, params["astro1Comp2DDeltaGamma"].center,
        #                                                 params["astro1Comp2DNorm"].width,  params["astro1Comp2DDeltaGamma"].width, 
        #                                                 self.astro_cor)

    def __call__(self, these_params:ParamPoint):
        value = 0.0
        # add up contributions from the procedural ones
        for key in params:
            value += params[key].prior( these_params[key] )

        # add up contributions from the 2D ones 
        value += self.ice_gradient_joined_prior( these_params["icegrad0"], these_params["icegrad1"] )
        #value += self.astro_correlated_prior( these_params["astro1Comp2DNorm"], these_params["astro1Comp2DDeltaGamma"])

        return value


