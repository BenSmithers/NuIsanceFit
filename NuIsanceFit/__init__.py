import json
import os

from .data import Data

f = open(os.path.join(os.path.dirname(__file__),"resources/steering.json"),'r')
steering = json.load(f)
f.close()
