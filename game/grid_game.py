'''


# gym.env-Gridworld[ammas2006]
@Eugene.Convolua.Ares

! MIT
License
~
for DD paper usage: 50
tasks
GridworldEnv.
'''

import os
import numpy as np
# gym
import gym
from gym import spaces


# a grid
class Grid (object):

    # (x,y), [none/start/fin/wall], R, V
    def __init__(self, x=None, y=None, type=0, reward=0.0, value=0.0):
        self.x = x
        self.y = y
        self.type = value  # Wall/ Wind/ Start/ End/ Fall
        self.reward = reward
        self.value = value

        self.name = None
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value{3}".format(self.x, self.y, self.type, self.reward, self.value,
                                                                    self.name)


# matrix
class GridMatrix (object):

    def __init__(self, n_width, n_height, default_type=0, default_reward=0.0, default_value=0.0):

        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x, y, self.default_type, self.default_reward, self.default_value))

    def get_grid(self, x, y=None):
        xx, yy = None, None
        if isinstance (x, int):
            xx, yy = x, y
        elif isinstance (x, tuple):
            xx, yy = x[0], x[1]
        # assert(xx>=0 and yy>=0 and xx < self.n_width and yy < self.n_height)
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid (x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise ("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid (x, y)
        if grid is not None:
            grid.value = value
        else:
            raise ("grid doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid (x, y)
        if grid is not None:
            grid.type = type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid (x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid (x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid (x, y)
        if grid is None:
            return None
        return grid.type


# mother environment
class GridworldEnv (gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, args, n_width=24, n_height=21, types=[]):
        self.args = args
        u_size = self.args['ui_size']
        self.u_size = u_size  #
        self.n_width = n_width  #
        self.n_height = n_height  #
        self.width = u_size * n_width  #
        self.height = u_size * n_height  #
        self.default_reward = self.args['default_reward']
        self.default_type = self.args['default_type']
        self.adjust_size()

        self.grids = GridMatrix (n_width=self.n_width,
                                 n_height=self.n_height,
                                 default_reward=self.default_reward,
                                 default_type=self.default_type,
                                 default_value=0.0)
        self.reward = 0  # for rendering
        self.action = None  # for rendering
        self.spec = None
        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_space = spaces.Discrete (4)
        # low high to observe
        features = self.n_width + self.n_height + 8
        self.observation_space = spaces.Discrete(features)
        self.directions = {0: (0, 1),
                           1: (0, -1),
                           2: (1, 0),
                           3: (-1, 0),
                           4: (1, 1),
                           5: (-1, -1),
                           6: (1, -1),
                           7: (-1, 1)}
        #
        self.types = types  #
        taskR = args['task'] // self.n_height
        taskC = args['task'] % self.n_height
        self.start = (0, 0)  # start /randomstart
        self.ends = [(taskR, taskC)]
        self.rewards = [(taskR, taskC, args['done_reward'])]
        self.refresh_setting()
        self.viewer = None  #
        # self._reset()  # reset 2 _reset
        self.count = 0

    def adjust_size(self):
        pass

    def step(self, action):
        # assert self.action_space.contains(action), \
        #  "%r (%s) invalid" % (action, type(action))

        self.action = action  # action for rendering
        old_x, old_y = self._state_to_xy (self.state)
        new_x, new_y = old_x, old_y

        if action == 0:
            new_y -= 1  # down
        elif action == 1:
            new_x -= 1  # left
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_x += 1  # right

        # boundary effect
        if new_x < 0: new_x = 0
        if new_x >= self.n_width: new_x = self.n_width - 1
        if new_y < 0: new_y = 0
        if new_y >= self.n_height: new_y = self.n_height - 1

        # wall effect:

        if self.grids.get_type (new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward (new_x, new_y)

        done = self._is_end_state (new_x, new_y)
        self.state = self._xy_to_state (new_x, new_y)
        #
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        x, y = self._state_to_xy (self.state)
        '''
    pixel_size = 16
    obversion = np.zeros([self.n_width * pixel_size, self.n_height * pixel_size, 3])
    for x in range(self.n_width):
      for y in range(self.n_height):
        if self.grids.get_type(x, y) == 1:  #
          obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [0, 0, 0]
        else:
          obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [255, 255, 255]
        if self._is_end_state(x, y):
          obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [0, 255, 255]
    obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [255, 255, 0]
    '''
        # return np.array([x, y]), self.reward, done
        # one hot
        one_hot_x = np.zeros(self.n_width)
        one_hot_y = np.zeros(self.n_height)
        one_hot_x[x] = 1
        one_hot_y[y] = 1
        walls = []
        for i in range(8):
            move = self.directions.get(i)
            position = np.sum(((x, y), move), axis=0)
            if (position[0], position[1], 1) in self.types:
                walls.append(1)
            else:
                walls.append(0)
        one_hot = np.concatenate((one_hot_x, one_hot_y, walls))
        return np.array(one_hot), self.reward, done, {}

    def set_task(self, task):
        taskR = task // self.n_height
        taskC = task % self.n_height
        for x, y in self.ends:
            self.grids.set_reward (x, y, self.default_reward)
        self.ends = [(taskR, taskC)]
        for x, y in self.ends:
            self.grids.set_reward (x, y, self.args['done_reward'])

    #
    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int ((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance (x, int):
            assert (isinstance (y, int)), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance (x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  #

    def refresh_setting(self):

        for x, y, r in self.rewards:
            self.grids.set_reward (x, y, r)
        for x, y, t in self.types:
            self.grids.set_type (x, y, t)

    def reset(self):
        if self.args['random_start']:
        # self.state = self._xy_to_state(self.start)
            self.reset_rand ()  # self.state = 55
        else:
            self.state = self._xy_to_state(self.args['start_position'])
        # self.state = 478
        x, y = self._state_to_xy (self.state)
        self.start = [x, y]
        '''
    pixel_size = 16
    obversion = np.zeros([self.n_width * pixel_size, self.n_height * pixel_size, 3])
    for x in range(self.n_width):
      for y in range(self.n_height):
        if self.grids.get_type(x, y) == 1:  #
          obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [0, 0, 0]
        else:
          obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [255, 255, 255]
        if self._is_end_state(x, y):
          obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [0, 255, 255]
    obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [255, 255, 0]
    '''
        # print([x, y, self.start[0], self.start[1]], self.grids.get_type(x, y))
        # return np.array([x, y])
        #print(np.array([x, y]))
        # original
        #return np.array([x, y])
        # one hot
        '''
        walls = []
        for i in range(8):
            move = self.directions.get(i)
            position = np.sum(((x, y), move), axis=0)
            if (position[0], position[1], 1) in self.types:
                walls.append(1)
            else:
                walls.append(0)
        '''
        one_hot_x = np.zeros(self.n_width)
        one_hot_y = np.zeros(self.n_height)
        one_hot_x[x] = 1
        one_hot_y[y] = 1
        walls = []
        for i in range(8):
            move = self.directions.get(i)
            position = np.sum(((x, y), move), axis=0)
            if (position[0], position[1], 1) in self.types:
                walls.append(1)
            else:
                walls.append(0)
        one_hot = np.concatenate((one_hot_x, one_hot_y, walls))
        return np.array(one_hot)

    def rand_reset(self):
        self.state = np.random.randint (0, self.n_width - 1)
        return self.state

    def reset_rand(self):
        self.state = np.random.randint (0,
                                        self.n_height * self.n_width)  # -2 for the last line is all wall, can't enter
        while ((self.state % self.n_width), (self.state // self.n_width), 1) in self.types:
            self.state = np.random.randint (0, self.n_height * self.n_width)
            # print(self.state)
        return self.state

    #
    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance (x, int):
            xx, yy = self._state_to_xy (x)
        else:
            assert (isinstance (x, tuple))
            xx, yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    # render
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close ()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2

        #
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer (self.width, self.height)

            #
            for x in range (self.n_width):
                for y in range (self.n_height):
                    v = [(x * u_size + m, y * u_size + m),
                         ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m),
                         (x * u_size + m, (y + 1) * u_size - m)]
                    # obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [255, 255, 255]
                    rect = rendering.FilledPolygon (v)

                    r = self.grids.get_reward (x, y)  # -0.01 for normal
                    if r == 0:
                        r = -1

                    if r < 0:
                        rect.set_color (0.5 - r, 0.5 + r, 0.1)  # orange
                        # obversion[x*pixel_size:(x+1)*pixel_size, y*pixel_size:(y+1)*pixel_size] = [255, 0, 0]
                    elif r > 0:
                        rect.set_color (0.5 - r, 0.5 + r, 0.1)  # green
                        # obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [0, 255, 0]
                    else:
                        rect.set_color (0.0, 0.0, 0.0)  # no reward - wall
                        # obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [255, 255, 255]
                    self.viewer.add_geom (rect)
                    #
                    v_outline = [(x * u_size + m, y * u_size + m),
                                 ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                 (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon (v_outline, False)
                    outline.set_linewidth (3)

                    if self._is_end_state (x, y):
                        #
                        outline.set_color (0.9, 0.9, 0)
                        # obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [240, 240, 0]
                        self.viewer.add_geom (outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color (0.5, 0.5, 0.8)
                        # obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [120, 120, 200]
                        self.viewer.add_geom (outline)
                    if self.grids.get_type (x, y) == 1:  #
                        # obversion[x * pixel_size:(x + 1) * pixel_size, y * pixel_size:(y + 1) * pixel_size] = [255, 255, 255]
                        rect.set_color (0.3, 0.3, 0.3)
                    else:
                        pass
            #
            self.agent = rendering.make_circle (u_size / 4, 30, True)
            self.agent.set_color (1.0, 1.0, 1.0)
            self.viewer.add_geom (self.agent)
            self.agent_trans = rendering.Transform ()
            self.agent.add_attr (self.agent_trans)

        x, y = self._state_to_xy (self.state)
        self.agent_trans.set_translation ((x + 0.5) * u_size, (y + 0.5) * u_size)
        return self.viewer.render (return_rgb_array=mode == 'rgb_array')


def getLayout(name):
    if not os.path.exists(name):
        return None
    f = open(name)
    try:
        layoutText = [line.strip() for line in f]
        width = len(layoutText[0])
        height = len(layoutText)
        types = []
        for x in range(width):
            for y in range(height):
                if layoutText[y][x] == '1':
                    types.append((x, height - y - 1, 1))
        return width, height, types
    finally:
        f.close()


# Environment for gridworld[ammas2006] in 50 tasks
def getEnv(args):
    width, height, types = getLayout(args['configuration'])
    env = GridworldEnv(args, n_width=width, n_height=height, types=types)

    # env._render()
    return env

'''
dic = {}
dic['configuration'] = 'grid_layout/change_layout.lay'
dic['task'] = 54
dic['ui_size'] = 40
dic['default_reward'] = 0
dic['default_type'] = 0
dic['done_reward'] = 5
dic['random_start'] = False
dic['start_position'] = (1,1)

env = getEnv(dic)
env.reset()
import time
while(True):
    time.sleep(2)
    action = np.random.randint(0, 4)
    env.step(action)
    env.render()
'''


