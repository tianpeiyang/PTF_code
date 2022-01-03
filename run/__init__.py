from .run_dqn import run as DQN_RUN
from .run_a3c import run as A3C_RUN
from .run_caps import run as CAPS_RUN
from .run_ptf_a3c import run as PTF_A3C_RUN
from .run_ppo import run as PPO_RUN
from .run_ptf_ppo import run as PTF_PPO_RN

REGISTRY = {}

REGISTRY["dqn"] = DQN_RUN
REGISTRY["a3c"] = A3C_RUN
REGISTRY["caps"] = CAPS_RUN
REGISTRY["ptf_a3c"] = PTF_A3C_RUN
REGISTRY["ppo"] = PPO_RUN
REGISTRY["ptf_ppo"] = PTF_PPO_RN
