import os, sys
from datetime import datetime

"""
Bringing in this logger from another [project of mine](https://github.com/BenSmithers/MultiHex/blob/Smithers/MapModeProto/MultiHex/logger.py)

This way we can keep a written file with the output from running this project 
In short, this opens up a file in the user's home or local directory, under a subfolder "NuIsanceFit"
In the code base, one can write messages to this log file at various levels of severity:
    Trace - call this basically whenever. It's used for debugging  (4)
    Log   - Sporadically call this to track the overall progress (3)
    Warn  - something might be wrong, but we can carry on (2)
    Fatal - raises an exception (1)

You can log by doing 
    Logger.Log("This is your message")
or by swapping out "Log" with a different level. Fatal should be called with an exception type 
    Logger.Fatal("Bad number error", ValueError) 

The logger will log (written to file, and to the terminal) anything at this log level or below
It includes the type of log, a timestamp, and the message
"""

log_level = 3

def get_base_dir():
    # set up the save directory
    if sys.platform=='linux':
        basedir = os.path.join(os.path.expandvars('$HOME'),'.local','NuIsanceFit')
    elif sys.platform=='darwin': #macOS
        basedir = os.path.join(os.path.expandvars('$HOME'),'NuIsanceFit')
    elif sys.platform=='win32' or sys.platform=='cygwin': # Windows and/or cygwin. Not actually sure if this works on cygwin
        basedir = os.path.join(os.path.expandvars('%AppData%'),'NuIsanceFit')
    else:
        Logger.Fatal("{} is not a supported OS".format(sys.platform), NotImplementedError)

    return(basedir)

if not os.path.exists(get_base_dir()):
    os.mkdir(get_base_dir())

logfile = os.path.join(get_base_dir(), "NuIsanceFit.log")
class LoggerClass:
    """
    This is the actual object that does the logging (CLI and file)
    """
    def __init__(self, level=2, visual=True):
        if not isinstance(level,int):
            self.visual = True
            self.Fatal("Logger passed level of type {}, not {}".format(type(level), int), TypeError)
        if not isinstance(visual, bool):
            self.visual = True
            self.Fatal("Logger passed level of type {}, not {}".format(type(level), int), TypeError)

        # the buffering is such that it flushes out the file after every line
        self.file = open(logfile,mode='wt', buffering=1)
        self.level = level
        self.visual = visual
        
        self.pipe = None
        self.Trace("Initializing Logger")


    def connect(self, target):
        """
        This way we can connect some other object to this logger, and potentially display the log output in some gui (or whatever)
        """
        if not hasattr(target, "__call__"):
            self.Fatal("Cannot pipe to a non-callable", TypeError)
         
        self.pipe = target
        try:
            self.Log("Successfully Connected Pipe")
        except Exception as err:
            self.pipe = None
            self.Warn("Failed to connect pipe. Exception {}".format(err))


    def _log(self,level,message):
        """
        Logs the message in the file, prints it out, and (optionally) pipes it throughto somethign else 
        """
        if level == 1:
            status = "ERROR "
        elif level==2:
            status = "WARN  "
        elif level==3:
            status = "LOG   "
        elif level==4:
            status = "TRACE "
        else:
            self.Fatal("Received invalid log level {}".format(level), ValueError)

        date_string = str(datetime.now())

        self.file.write(" ".join([date_string, status, message,"\n"]))

        if self.pipe is not None:
            self.pipe(" ".join([date_string, status, message,"\n"]))

    def Trace(self,message):
        if self.level>=4:
            if self.visual:
                print(message)
            self._log(4,message)

    def Log(self, message):
        if self.level>=3:
            if self.visual:
                print(message)
            self._log(3,message)

    def Warn(self, warning):
        if self.level>=2:
            if self.visual:
                print(warning)
            self._log(2,warning)

    def Fatal(self, error, exception=Exception):
        if self.visual:
            print(error)
        self._log(1,error)
        raise exception(error)


# may want to have these settings changed in some config file? 
Logger = LoggerClass(level=log_level,visual=True)