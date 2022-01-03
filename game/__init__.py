from .pinball import PinballModel
from .pinball2 import PinballModel as pinball2
from .gym_game import game as GYM
from .grid_game import getEnv as GRID
#from .control2gym_game import game as CGYM

REGISTRY = {}
REGISTRY["pinball"] = PinballModel
REGISTRY["pinball2"] = pinball2
REGISTRY["grid"] = GRID
REGISTRY['CartPole-v0'] = GYM
#REGISTRY['reacher'] = CGYM