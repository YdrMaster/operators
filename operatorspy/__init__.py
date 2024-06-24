import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from .liboperators import open_lib, to_tensor, CTensor
from .devices import DeviceEnum
