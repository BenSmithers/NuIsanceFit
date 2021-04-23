from PyQt5 import QtCore 
from PyQt5.QtWidgets import QMainWindow,QApplication

import numpy as np
from logger import Logger 
import time

class ThreadManager(QMainWindow):
    def __init__(self, function, datalist, dtype=float):
        """
         - takes a callable called `function` that returns objects of datatype `dtype`
         - takes a list-like of data (datalist). 

         Creates threads to call the function independently with each entry in datalist

         Returns a list of len(datalist), with an entry for each function's return 
        """
        super(ThreadManager, self).__init__()
        if not hasattr(function, "__call__"):
            Logger.Fatal("Expected callable function.", TypeError)
        if not isinstance(datalist, (list, tuple, np.ndarray)):
            Logger.Fatal("Expected list-like, got {}".format(type(datalist)))
        if not isinstance(dtype, type):
            Logger.Fatal("`dtype` should be a type definition, got {}".format(type(dtype)))

        self.threadpool = QtCore.QThreadPool()
        
        # queries the CPU for how many threads we can have 
        self._mtp = self.threadpool.maxThreadCount()

        self._jobs = list(datalist)
        self._function = function
        self._dtype = dtype
        
        self._done = 0
        self._return_vals = [self._dtype() for i in range(len(datalist))]

        # make as many jobs as we can while there's work to do 
        while self.activeThreads<self._mtp and len(self._jobs)>=1:
            self.make_thread()

        
    @property
    def activeThreads(self):
        return int(self.threadpool.activeThreadCount())

    def make_thread(self):
        """
        Get one of the jobs we still have to do, create a thread for it, and let it rip.

        We can therefore have multiple sets of data being processed simultaneously! 
        """
        assert(self.activeThreads<self._mtp)
        assert(len(self._jobs)>=1)
        jobid = len(self._jobs)-1 # give each job an index so we know where to put it when the job's done 
        data = self._jobs.pop()
        newthread = Thread(self._function, data, jobid)
        newthread.signals.signal.connect(self.thread_done)
        self.threadpool.start(newthread) #calls the run function below 

    def thread_done(self, value):
        """
        Called when a thread finishes. 
        Appends the return value (these are NOT necessarily in order)
        """
        assert(isinstance(value[1], self._dtype))
        print("in done part")
        self._return_vals[value[0]] = value[1]
        
        # allow a closed thread to make another in its place 
        if self.activeThreads<self._mtp and len(self._jobs)>=1:
            self.make_thread()

        if self.activeThreads==0:
            self.destroy()

    def __call__(self):
        return self._return_vals 

class Signaler(QtCore.QObject):
    signal = QtCore.pyqtSignal(list)

class Thread(QtCore.QRunnable):
    def __init__(self, function, data, jobid):
        super(Thread, self).__init__()
        self._function = function
        self._data = data
        self.signals = Signaler()
        self.jobid = jobid
        self.setAutoDelete(True)

    @QtCore.pyqtSlot()
    def run(self):
        """
        Called when the thread manager starts the thread
        """
        self.result = self._function(self._data)
        print("emiutting {}".format(self.result))
        self.signals.signal.emit([self.jobid , self.result])
"""
    Leaving this here for later, 
        this is the context needed for executing one of these with the proper backend all there

    app = QApplication([])
    window = ThreadManager()
    app.exec_()
"""

from math import sqrt
datalist = np.linspace(0,50,11)

def funcy(number):
    time.sleep(1)
    return sqrt(number)

app = QApplication([])
window = ThreadManager(funcy, datalist)
app.exec_()

window()
window.close()
