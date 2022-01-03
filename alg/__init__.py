from .DQN import DeepQNetwork
from .Caps import CAPS
from .A3C import A3C
from .PTF_A3C import PTF_A3C
from .PTF_PPO import PTF_PPO
from .PPO import PPO

REGISTRY = {}

REGISTRY["dqn"] = DeepQNetwork
REGISTRY['caps'] = CAPS
REGISTRY["a3c"] = A3C
REGISTRY['ptf_a3c'] = PTF_A3C
REGISTRY['ppo'] = PPO
REGISTRY['ptf_ppo'] = PTF_PPO
