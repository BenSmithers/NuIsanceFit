from threading import Thread
import time
import numpy as np

from NuIsanceFit.logger import Logger 


# these are global return intermediates
# we save the function returns here before actually putting them in the _return_vals array 
_t1_ret = None
_t2_ret = None
_t3_ret = None
_t4_ret = None

def ThreadManager( function, datalist):
    """
    function - what function you want to call
    datalist - a list of data. Each entry will be individually passed to the function 

    This is hard-coded in a kinda nasty way, but HEAR ME OUT

    We needed to be super careful about race conditions, and so I need to keep totally separate places in memory for these different intermediate return values.
    Python doens't like it when two threads access the samething. So, I needed to hard-code some of this. 

    Now you could make the threads in a procedural way without repeating code. That's doable! But the part where you rejoin the threads _would_ need to be hard-coded. 
    """
    if not hasattr(function, "__call__"):
        Logger.Fatal("Expected callable function.", TypeError)
    if not isinstance(datalist, (list, tuple, np.ndarray)):
        Logger.Fatal("Expected list-like, got {}".format(type(datalist)))
    #if not isinstance(dtype, type):
    #    Logger.Fatal("`dtype` should be a type definition, got {}".format(type(dtype)))

    Logger.Trace("Calling Thread Manager")
    
    _jobs = list(datalist)
    _function = function
    #_dtype = dtype
    
    _done = 0
    _return_vals = [None for i in range(len(datalist))]

    _thread_deets = {}

    while not all([entry is not None for entry in _return_vals]):
        
        Logger.Trace("Starting Thread 1")
        jobid = len(_jobs)-1
        t1 = Thread( target=metajob, args=(function,_jobs.pop(),jobid, 1, ))
        t1.start()

        t2_on = False
        t3_on = False
        t4_on = False

        if len(_jobs)>=1:
            Logger.Trace("Starting Thread 2")
            jobid = len(_jobs)-1
            t2 = Thread( target=metajob, args=(function,_jobs.pop(),jobid, 2, ))
            t2.start()
            t2_on = True

        if len(_jobs)>=1:
            Logger.Trace("Starting Thread 3")
            jobid = len(_jobs)-1
            t3 = Thread( target=metajob, args=(function,_jobs.pop(),jobid, 3, ))
            t3.start()
            t3_on = True
        
        if len(_jobs)>=1:
            Logger.Trace("Starting Thread 4")
            jobid = len(_jobs)-1
            t4 = Thread( target=metajob, args=(function,_jobs.pop(),jobid, 4, ))
            t4.start()
            t4_on = True

        t1.join()
        _return_vals[_t1_ret[0]] = _t1_ret[1]
        if t2_on:
            t2.join()
            _return_vals[_t2_ret[0]] = _t2_ret[1]
        if t3_on:
            t3.join()
            _return_vals[_t3_ret[0]] = _t3_ret[1]
        if t4_on:
            t4.join()
            _return_vals[_t4_ret[0]] = _t4_ret[1]

    return _return_vals

def metajob(function, data, jobid, threadno):
    """
    We create these 'metajobs' using the function and thread-specific values

    This way we can handle the return values without worrying about race-conditions
    All four will finish on their own time, and save the result to different return paramters 
    """
    ret_val = function(data)
    if threadno==1:
        global _t1_ret
        _t1_ret = [jobid, ret_val]
    elif threadno==2:
        global _t2_ret
        _t2_ret = [jobid, ret_val]
    elif threadno==3:
        global _t3_ret
        _t3_ret = [jobid, ret_val]
    elif threadno==4:
        global _t4_ret
        _t4_ret = [jobid, ret_val]
    else:
        raise Exception("Invalid Thread ID {}".format(threadno))
