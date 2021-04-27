from logger import Logger
from event import Event 

from cascade.utils import get_loc

import numpy as np

class eventBin:
    def __init__(self):
        self._contains = []
    def add(self, event):
        if not isinstance(event, Event):
            Logger.Fatal("Cannot bin event of type {}")
        self._contains += event
        
    def __add__(self,event):
        if isinstance(event, Event):
            new_obj = eventBin()
            new_obj._contains= self._contains + [event]
            return new_obj
        elif isinstance(event, eventBin):
            new_obj = eventBin()
            new_obj._contains = self._contains + event._contains
            return new_obj
        else:
            Logger.Fatal("Cannot perform '+' with {}".format(type(event)))

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
        return [ _get_entry(dims[1:]) for i in range(dims[0]) ]

def itemset(source, amount, binloc):
    """
    Sets the entry in source at coordinates -binloc- to amount
    """
    if len(binloc)==1:
        source[binloc[0]] = amount
    else:
        itemset(source, amount, binloc[:1])

def itemadd(source,amount, binloc):
    """
    Adds 'amount' to the entry in source at coordinates -binloc-
    """
    if len(binloc)==1:
        source[binloc[0]]+=amount
    else:
        itemadd(source,amount, binloc[:1])

def itemget(source, binloc):
    """
    Gets the entry in source at coordinates -binloc- 
    """
    if len(binloc)==1:
        return source[binloc[0]]
    else:
        return itemget(source, binloc[:1])



class bhist:
    """
    Binned Histogram type, or Bhist     

    I made this so I could have a binned histogram that could be used for adding more stuff at arbitrary places according to some "edges" it has. The object would handle figuring out which of its bins would hold the stuff. 

    Also made with the potential to store integers, floats, or whatever can be added together and has both an additive rule and some kind of identity element correlated with the default constructor. 
        If a non-dtype entry is given, it will be explicitly cast to the dtype. 
    """
    def __init__(self,edges, dtype=float):
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

        self._edges = np.sort(edges) # make sure the edges are all sorted 
        self._dtype = dtype 

        # Ostensibly you can bin strings... not sure why you would, but you could  
        try:
            x = dtype() + dtype()
        except Exception:
            raise TypeError("It appears impossible to add {} together.".format(dtype))

        # build the function needed to register additions to the histograms.
        dims = tuple([len(self._edges[i])-1 for i in range(len(self._edges))])
        self._fill = build_arbitrary_list( dims, self._dtype ) 


        self._counter = 0

    def add(self, amt, *args):
        """
        Tries to bin some data passed to the bhist. Arbitrarily dimensioned cause I was moving from 2D-3D and this seemed like a good opportunity 
            amt is the amount to add
            *args specifies the coordinates in our binned space 

        """
        if not len(args)==len(self._edges):
            raise ValueError("Wrong number of args to register! Got {}, not {}".format(len(args), len(self._edges)))
        if not isinstance(amt, self._dtype):
            try:
                amount = self._dtype(amt)
            except TypeError:
                raise TypeError("Expected {}, got {}. Tried casting to {}, but failed.".format(self._dtype, type(amt), self._dtype))
        else:
            amount = amt

        bin_loc = tuple([get_loc( args[i], self._edges[i])[0] for i in range(len(args))]) # get the bin for each dimension
        # Verifies that nothing in the list is None-type
        if all([x is not None for x in bin_loc]):
            # itemset works like ( *bins, amount )
            itemadd(self._fill, amount, bin_loc)

            return tuple(bin_loc)



    def __iter__(self):
        return iter(self._fill)


    @property
    def dtype(self):
        return self._dtype

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

