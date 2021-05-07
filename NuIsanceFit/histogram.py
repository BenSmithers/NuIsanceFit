from NuIsanceFit.logger import Logger
from NuIsanceFit.event import Event 

import numpy as np
from math import log10

from collections import deque

def sci(number, precision=4):
    """
    Returns a string representing the number in scientific notation
    """
    if not isinstance(number, (int, float)):
        raise TypeError("Expected {}, not {}".format(float, type(number)))
    if not isinstance(precision,int):
        raise TypeError("Precision must be {}, not {}".format(int, type(precision)))
    try:
        power = int(log10(abs(number)))
    except ValueError:
        return("0.0")

    return("{0:.{1}f}".format(number/(10**power), precision)+"e{}".format( power))

class eventBin:
    def __init__(self):
        self._contains = deque()

    def add(self, event):
        if not isinstance(event, Event):
            Logger.Fatal("Cannot bin event of type {}")
        self._contains.append( event )
       
    def __iadd__(self, event):
        if isinstance(event, Event):
            self._contains.append(event)
        elif isinstance(event, eventBin):
            self._contains.extend( event._contains )
        else:
            Logger.Fatal("Cannot perform '+' with {}".format(type(event)))

    def __add__(self, event):
        if isinstance(event, Event):
            new_obj = eventBin()
            new_obj._contains = self._contains
            new_obj._contains.append(event)
            return new_obj
        elif isinstance(event, eventBin):
            new_obj = eventBin()
            new_obj._contains = self._contains
            new_obj._contains.extend( event._contains )
            return new_obj
        else:
            Logger.Fatal("Cannot perform '+' with {}".format(type(event)))

    def __iter__(self):
        return iter(self._contains)

    def __getitem__(self, index):
        return self._contains[index]

    def __len__(self):
        return len(self._contains)

def build_arbitrary_list(dims, dtype):
    """
    This allows us to build an array of arbitrary dimension (dims)
    Filled with objects of datatype dtype.

    Uses default constructor! 
        dtype() needs to work


    It's recursive, which I don't like. But it works pretty well!
    """
    if not isinstance(dims, (list,tuple,np.ndarray)):
        Logger.Fatal("Need iterable dims! Got {}".format(type(dims)), TypeError)
    if not isinstance(dtype, type):
        Logger.Fatal("dtype should be type {}, got {}".format(type, type(dtype)),TypeError)
    if len(dims)==1:
        return [dtype() for i in range(dims[0])]
    else:
        return [ build_arbitrary_list(dims[1:], dtype) for i in range(dims[0]) ]

def itemset(source, amount, binloc):
    """
    Sets the entry in source at coordinates -binloc- to amount
    """
    if len(binloc)==1:
        source[binloc[0]] = amount
    else:
        itemset(source[binloc[0]], amount, binloc[1:])

def itemadd(source,amount, binloc):
    """
    Adds 'amount' to the entry in source at coordinates -binloc-
    """
    if len(binloc)==1:
        assert(isinstance(binloc[0],int))
        source[binloc[0]] = source[binloc[0]] + amount
    else:
        itemadd(source[binloc[0]], amount, binloc[1:])

def itemget(source, binloc):
    """
    Gets the entry in source at coordinates -binloc- 
    """
    if len(binloc)==1:
        return source[binloc[0]]
    else:
        return itemget(source[binloc[0]], binloc[1:])

def get_loc(x, domain, just_left=False):
    """
    Returns the indices of the entries in domain that border 'x' 
    Raises exception if x is outside the range of domain 

    Assumes 'domain' is sorted! And this _only_ works if the domain is length 2 or above 

    Uses a binary search algorithm
    """
    if not isinstance(domain, (tuple,list,np.ndarray)):
        raise TypeError("'domain' has unrecognized type {}, try {}".format(type(domain), list))
    if not isinstance(x, (float,int)):
        raise TypeError("'x' should be number-like, not {}".format(type(x)))

    if len(domain)<=1:
        raise ValueError("get_loc function only works on domains of length>1. This is length {}".format(len(domain)))

    if x<domain[0] or x>domain[-1]:
        Logger.Trace("x={} and is outside the domain: ({}, {})".format(sci(x), sci(domain[0]), sci(domain[-1])))
        return

    min_abs = 0
    max_abs = len(domain)-1

    lower_bin = int(abs(max_abs-min_abs)/2)
    upper_bin = lower_bin+1

    while not (domain[lower_bin]<=x and domain[upper_bin]>=x):
        if abs(max_abs-min_abs)<=1:
            Logger.Log("Was {} in {}".format(x, domain))
            Logger.Fatal("get_loc failed. Was the data unsorted?",Exception)

        if x<domain[lower_bin]:
            max_abs = lower_bin
        if x>domain[upper_bin]:
            min_abs = upper_bin

        # now choose a new middle point for the upper and lower things
        lower_bin = min_abs + int(abs(max_abs-min_abs)/2)
        upper_bin = lower_bin + 1

    assert(x>=domain[lower_bin] and x<=domain[upper_bin])
    if just_left:
        return lower_bin
    else:
        return(lower_bin, upper_bin)

# one note - I'm manually entering in these flatten and transpose arrays, and you might be wondering why I didn't just use numpy arrays.
# Numpy arrays don't really like having non-numbers inside, and so they had issues with the Event objects
# so, I had to use lists. And since I'm using lists, we need these special functions 
def flatten(bhist):
    """
    Takes a bhist, or list, and returns the flattened version
    """
    if isinstance(bhist, list):
        target = bhist
    elif isinstance(bhist, bHist):
        target = bhist._fill

    out = []
    for item in target:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out

def transpose(obj):
    """
    Takes a 2D array and transposes it 
    """
    return list(zip(*obj))

class bHist:
    """
    Binned Histogram type, or Bhist     

    I made this so I could have a binned histogram that could be used for adding more stuff at arbitrary places according to some "edges" it has. The object would handle figuring out which of its bins would hold the stuff along an arbitrary number of axes 

    We also have discrete bin and data types. The only requirement is that the bin is that 
        bin + data 
    is a valid operation that modifies the bins. Both could be floats, or lists, or whatever. 
    """
    def __init__(self,edges, bintype = list, datatype=float):
        """
        Arg 'edges' should be a tuple of length 1 or 2  

        The type-checking could use a bit of work... Right now for 1D histograms you need to give it a length-1 list. 
        """

        if not (isinstance(edges, list) or isinstance(edges, tuple) or isinstance(edges, np.ndarray)):
            raise TypeError("Arg 'edges' must be {}, got {}".format(list, type(edges)))

        for entry in edges:
            if not (isinstance(entry, list) or isinstance(entry, tuple) or isinstance(entry, np.ndarray)):
                raise TypeError("Each entry in 'edges' should be list-like, found {}".format(type(entry)))
            if len(entry)<2:
                raise ValueError("Entries in 'edges' must be at least length 2, got {}".format(len(entry)))

        self._edges = [np.sort(edge) for edge in edges] # make sure the edges are all sorted 
        self._bintype = bintype 
        self._datatype = datatype

        # build the function needed to register additions to the histograms.
        dims = tuple([len(self._edges[i])-1 for i in range(len(self._edges))])
        self._fill = build_arbitrary_list( dims, self._bintype ) 


        self._counter = 0

    def add(self, amt, *args):
        """
        Tries to bin some data passed to the bhist. Arbitrarily dimensioned cause I was moving from 2D-3D and this seemed like a good opportunity 
            amt is the amount to add
            *args specifies the coordinates in our binned space 

        """
        if not len(args)==len(self._edges):
            raise ValueError("Wrong number of args to register! Got {}, not {}".format(len(args), len(self._edges)))
        if not isinstance(amt, self._datatype):
            try:
                amount = self._datatype(amt)
            except TypeError:
                raise TypeError("Expected {}, got {}. Tried casting to {}, but failed.".format(self._datatype, type(amt), self._datatype))
        else:
            amount = amt
        
        # note: get_loc returns the indices of the edges that border this point. So we grab the left-edge; the bin number
        bin_loc = tuple([get_loc( args[i], self._edges[i], True) for i in range(len(args))]) # get the bin for each dimension

        # Verifies that nothing in the list is None-type
        if all([x is not None for x in bin_loc]):
            # itemset works like ( *bins, amount )
            itemadd(self._fill, amount, bin_loc)
            
            return tuple(bin_loc)

    def __getitem__(self, index):
        if not isinstance(index, int):
            Logger.Fatal("Cannot access entry of type {}".format(type(index)),TypeError)
        return self._fill[index]

    def __iter__(self):
        return iter(self._fill)

    def __len__(self):
        return len(self._fill)


    @property
    def dtype(self):
        return self._datatype

    # some access properties. Note these aren't function calls. They are accessed like "object.centers" 
    @property
    def centers(self):
        complete = [ [0.5*(subedge[i+1]+subedge[i]) for i in range(len(subedge)-1)] for subedge in self._edges]
        return(complete[0] if len(self._edges)==1 else complete)
    @property
    def edges(self):
        complete = [[value for value in subedge] for subedge in self._edges]
        return(complete[0] if len(self._edges)==1 else complete)
    @property
    def widths(self):
        complete = [[abs(subedges[i+1]-subedges[i]) for i in range(len(subedges)-1)] for subedges in self._edges]
        return(complete[0] if len(self._edges)==1 else complete)
    @property
    def fill(self):
        return(self._fill)

