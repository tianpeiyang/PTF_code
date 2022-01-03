import numpy as np


class PRM_actor:
    def __init__(self, action, action_dim):
        self.action_dim = action_dim
        self.action = action
        if type(action) is int:
            self.action_pro = np.zeros(self.action_dim, dtype=float)
            self.action_pro[action] = 1
        else:
            self.action_pro = 1

    def choose_acton_prob(self, s):
        return self.action_pro

    def choose_action_g(self, s):
        return self.action

    def is_from_source_actor(self, ob, action=None):
        return (np.array(self.action) == np.array(action)).all()
