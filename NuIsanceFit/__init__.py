import json
import os

from .data import Data

f = open(os.path.join(os.path.dirname(__file__),"resources/steering.json"),'r')
steering = json.load(f)
f.close()

if steering["resources"]["resource_dir"]=="":
    steering["resources"]["resource_dir"] = os.path.join(os.path.dirname(__file__),"resources")
