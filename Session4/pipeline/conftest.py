import os
import sys

_here = os.path.dirname(__file__)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "widget")))
