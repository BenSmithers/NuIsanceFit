import json
import os

from NuIsanceFit.logger import Logger
from NuIsanceFit.data import Data

f = open(os.path.join(os.path.dirname(__file__),"resources/steering.json"),'r')
steering = json.load(f)
f.close()

# If no resource_dir is specified, then we just use the default one, the _resources_ folder! 
if steering["resources"]["resource_dir"]=="":
    steering["resources"]["resource_dir"] = os.path.join(os.path.dirname(__file__),"resources")

# now, we prepend the root resource directory to the beginning of these file names / folder names! 
for key in steering["resources"]:
    if key =="resource_dir":
        continue
    
    # and also verify that they exist! 
    steering["resources"][key] = os.path.join( steering["resources"]["resource_dir"] , steering["resources"][key] )
    if not os.path.exists(steering["resources"][key]):
        Logger.Warn("Uh Oh! I looked for the {} file at {}, but didn't find it".format(key, steering["resources"][key])) #, IOError)


