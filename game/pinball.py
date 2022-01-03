"""
.. module:: pinball
   :platform: Unix, Windows
   :synopsis: Pinball domain for reinforcement learning
.. moduleauthor:: Pierre-Luc Bacon <pierrelucbacon@gmail.com>


"""

import random
import argparse
import numpy as np
from itertools import *
import pygame
from gym import spaces
from past.utils import old_div


class BallModel:
    """ This class maintains the state of the ball
    in the pinball domain. It takes care of moving
    it according to the current velocity and drag coefficient.

    """
    DRAG = 0.995

    def __init__(self, start_position, radius):
        """
        :param start_position: The initial position
        :type start_position: float
        :param radius: The ball radius
        :type radius: float
        """
        self.position = start_position
        self.radius = radius
        self.xdot = 0.0
        self.ydot = 0.0

    def add_impulse(self, delta_xdot, delta_ydot):
        """ Change the momentum of the ball
        :param delta_xdot: The change in velocity in the x direction
        :type delta_xdot: float
        :param delta_ydot: The change in velocity in the y direction
        :type delta_ydot: float
        """
        '''old
        self.xdot += delta_xdot / 5.0
        self.ydot += delta_ydot / 5.0
        self._clip (self.xdot)
        self._clip (self.ydot)
        '''
        self.xdot += old_div(delta_xdot, 5.0)
        self.ydot += old_div(delta_ydot, 5.0)
        self.xdot = self._clip(self.xdot)
        self.ydot = self._clip(self.ydot)

    def add_drag(self):
        """ Add a fixed amount of drag to the current velocity """
        self.xdot *= self.DRAG
        self.ydot *= self.DRAG

    def step(self):
        """ Move the ball by one increment """
        self.position[0] += self.xdot * self.radius / 20.0
        self.position[1] += self.ydot * self.radius / 20.0

    def _clip(self, val, low=-2, high=2):  # new -2 to 2
        """ Clip a value in a given range """
        if val > high:
            val = high
        if val < low:
            val = low
        return val


class PinballObstacle:
    """ This class represents a single polygon obstacle in the pinball domain and detects
    when a :class:`BallModel` hits it.
    When a collision is detected, it also provides a way to
    compute the appropriate effect to apply on the ball.
    """

    def __init__(self, points):
        """
        :param points: A list of points defining the polygon
        :type points: list of lists
        """
        self.points = points
        self.min_x = min(self.points, key=lambda pt: pt[0])[0]
        self.max_x = max(self.points, key=lambda pt: pt[0])[0]
        self.min_y = min(self.points, key=lambda pt: pt[1])[1]
        self.max_y = max(self.points, key=lambda pt: pt[1])[1]

        self._double_collision = False
        self._intercept = None

    def collision(self, ball):
        """ Determines if the ball hits this obstacle
        :param ball: An instance of :class:`BallModel`
        :type ball: :class:`BallModel`
        """
        self._double_collision = False

        if ball.position[0] - ball.radius > self.max_x:
            return False
        if ball.position[0] + ball.radius < self.min_x:
            return False
        if ball.position[1] - ball.radius > self.max_y:
            return False
        if ball.position[1] + ball.radius < self.min_y:
            return False

        a, b = tee(np.vstack([np.array(self.points), self.points[0]]))
        next(b, None)
        intercept_found = False
        for pt_pair in list(zip(a, b)):
            if self._intercept_edge(pt_pair, ball):
                if intercept_found:
                    # Ball has hit a corner
                    self._intercept = self._select_edge(pt_pair, self._intercept, ball)
                    self._double_collision = True
                else:
                    self._intercept = pt_pair
                    intercept_found = True

        return intercept_found

    def isPointinPolygon(self, ball):
        # 判断是否在外包矩形内，如果不在，直接返回false
        if ball.position[0] - ball.radius > self.max_x:
            return False
        if ball.position[0] + ball.radius < self.min_x:
            return False
        if ball.position[1] - ball.radius > self.max_y:
            return False
        if ball.position[1] + ball.radius < self.min_y:
            return False

        rangelist = [i for i in self.points]
        rangelist.append(self.points[0])
        #print(rangelist, self.points)

        for p in range(17):
            scale = (1 + int(p / 4)) / 4.0
            if p == 16:
                point = [ball.position[0], ball.position[1]]
            elif p % 4 == 0:
                point = [ball.position[0] - scale * ball.radius, ball.position[1] - scale * ball.radius]
            elif p % 4 == 1:
                point = [ball.position[0] - scale * ball.radius, ball.position[1] + scale * ball.radius]
            elif p % 4 == 2:
                point = [ball.position[0] + scale * ball.radius, ball.position[1] - scale * ball.radius]
            elif p % 4 == 3:
                point = [ball.position[0] + scale * ball.radius, ball.position[1] + scale * ball.radius]
            else:
                point = [ball.position[0], ball.position[1]]

            count = 0
            point1 = rangelist[0]
            for i in range(1, len(rangelist)):
                point2 = rangelist[i]
                # 点与多边形顶点重合
                if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
                    return True
                # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
                if (point1[1] < point[1] and point2[1] > point[1]) or (point1[1] > point[1] and point2[1] < point[1]):
                    # 求线段与射线交点 再和lat比较
                    point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
                    # 点在多边形边上
                    if point12lng == point[0]:
                        return True
                    if point12lng < point[0]:
                        count += 1
                point1 = point2
            if count % 2 == 1:
                return True
        return False

    def collision_effect(self, ball):
        """ Based of the collision detection result triggered in :func:`PinballObstacle.collision`,
        compute the change in velocity.
        :param ball: An instance of :class:`BallModel`
        :type ball: :class:`BallModel`
        """
        if self._double_collision:
            return [-ball.xdot, -ball.ydot]

        # Normalize direction
        obstacle_vector = self._intercept[1] - self._intercept[0]
        if obstacle_vector[0] < 0:
            obstacle_vector = self._intercept[0] - self._intercept[1]

        velocity_vector = np.array([ball.xdot, ball.ydot])
        theta = self._angle(velocity_vector, obstacle_vector) - np.pi
        if theta < 0:
            theta += 2 * np.pi

        intercept_theta = self._angle([-1, 0], obstacle_vector)
        theta += intercept_theta

        if theta > 2 * np.pi:
            theta -= 2 * np.pi

        velocity = np.linalg.norm([ball.xdot, ball.ydot])

        return [velocity * np.cos(theta), velocity * np.sin(theta)]

    def _select_edge(self, intersect1, intersect2, ball):
        """ If the ball hits a corner, select one of two edges.
        :param intersect1: A pair of points defining an edge of the polygon
        :type intersect1: list of lists
        :param intersect2: A pair of points defining an edge of the polygon
        :type intersect2: list of lists
        :returns: The edge with the smallest angle with the velocity vector
        :rtype: list of lists
        """
        velocity = np.array([ball.xdot, ball.ydot])
        obstacle_vector1 = intersect1[1] - intersect1[0]
        obstacle_vector2 = intersect2[1] - intersect2[0]

        angle1 = self._angle(velocity, obstacle_vector1)
        if angle1 > np.pi:
            angle1 -= np.pi

        angle2 = self._angle(velocity, obstacle_vector2)
        if angle1 > np.pi:
            angle2 -= np.pi

        # if np.abs(angle1 - (np.pi / 2.0)) < np.abs (angle2 - (np.pi / 2.0)): # old
        if np.abs(angle1 - (old_div(np.pi, 2.0))) < np.abs(angle2 - (old_div(np.pi, 2.0))):
            return intersect1
        return intersect2

    def _angle(self, v1, v2):
        """ Compute the angle difference between two vectors
        :param v1: The x,y coordinates of the vector
        :type: v1: list
        :param v2: The x,y coordinates of the vector
        :type: v2: list
        :rtype: float
        """
        angle_diff = np.arctan2(v1[0], v1[1]) - np.arctan2(v2[0], v2[1])
        if angle_diff < 0:
            angle_diff += 2 * np.pi
        return angle_diff

    def _intercept_edge(self, pt_pair, ball):
        """ Compute the projection on and edge and find out if it intercept with the ball.
        :param pt_pair: The pair of points defining an edge
        :type pt_pair: list of lists
        :param ball: An instance of :class:`BallModel`
        :type ball: :class:`BallModel`
        :returns: True if the ball has hit an edge of the polygon
        :rtype: bool
        """
        # Find the projection on an edge
        obstacle_edge = pt_pair[1] - pt_pair[0]
        difference = np.array(ball.position) - pt_pair[0]
        scalar_proj = old_div(difference.dot(obstacle_edge), obstacle_edge.dot(obstacle_edge))
        # scalar_proj = difference.dot (obstacle_edge) / obstacle_edge.dot (obstacle_edge) # old
        if scalar_proj > 1.0:
            scalar_proj = 1.0
        elif scalar_proj < 0.0:
            scalar_proj = 0.0

        # Compute the distance to the closest point
        closest_pt = pt_pair[0] + obstacle_edge * scalar_proj
        obstacle_to_ball = ball.position - closest_pt
        distance = obstacle_to_ball.dot(obstacle_to_ball)

        if distance <= ball.radius * ball.radius:
            # A collision only if the ball is not already moving away
            velocity = np.array([ball.xdot, ball.ydot])
            ball_to_obstacle = closest_pt - ball.position

            angle = self._angle(ball_to_obstacle, velocity)
            if angle > np.pi:
                angle = 2 * np.pi - angle

            # if angle > np.pi / 1.99: # old
            if angle > old_div(np.pi, 1.99):
                return False

            return True
        else:
            return False


class PinballModel:
    """ This class is a self-contained model of the pinball
    domain for reinforcement learning.

    It can be used either over RL-Glue through the :class:`PinballRLGlue`
    adapter or interactively with :class:`PinballView`.

    """
    ACC_X = 0
    ACC_Y = 1
    DEC_X = 2
    DEC_Y = 3
    ACC_NONE = 4

    STEP_PENALTY = 0
    THRUST_PENALTY = 0
    END_EPISODE = 10

    def __init__(self, args):
        """ Read a configuration file for Pinball and draw the domain to screen
        :param configuration: a configuration file containing the polygons, source(s) and target location.
        :type configuration: str
        """
        self.args = args
        if args['sequential_state']:
            self.tmp = np.zeros(16)
            self.observation_space = 16
        else:
            self.observation_space = 4
        if args['continuous_action']:
            self.action_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)
        else:
            self.action_space = 5
        self.action_effects = {self.ACC_X: (1, 0), self.ACC_Y: (0, 1), self.DEC_X: (-1, 0), self.DEC_Y: (0, -1),
                               self.ACC_NONE: (0, 0)}
        self.pri_action = [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
        self.configuration = args['configuration']
        # Set up the environment according to the configuration
        self.obstacles = []
        self.target_pos = []
        self.target_rad = 0.01

        self.ball_rad = 0.01
        self.start_pos = []
        self.done = False
        with open(self.configuration) as fp:
            for line in fp.readlines():
                tokens = line.strip().split()
                if not len(tokens):
                    continue
                elif tokens[0] == 'polygon':
                    self.obstacles.append(
                        PinballObstacle(list(zip (*[iter(list(map(float, tokens[1:])))] * 2))))
                elif tokens[0] == 'target':
                    self.target_pos = [float(tokens[1]), float(tokens[2])]
                    self.target_rad = float(tokens[3])
                elif tokens[0] == 'start':
                    self.start_pos = list(zip (*[iter (list(map (float, tokens[1:])))] * 2))
                elif tokens[0] == 'ball':
                    self.ball_rad = float(tokens[1])
        self.target_pos = self.args['target_position']
        self.start_pos = self.args['start_position']
        self.ball = BallModel(list(random.choice(self.start_pos)), self.ball_rad)

        if args['run_test']:
            self.build_view()

    def build_view(self):
        pygame.init()
        pygame.display.set_caption('Pinball Domain')
        screen = pygame.display.set_mode([self.args['width'], self.args['height']])
        # print(args)
        self.environment_view = PinballView(screen, self)

    def quit_view(self):
        pygame.quit()

    def get_random_start(self):
        return [0, 0]

    def get_state(self):
        """ Access the current 4-dimensional state vector
        :returns: a list containing the x position, y position, xdot, ydot
        :rtype: list
        """
        if self.args['sequential_state']:
            self.tmp[:12] = self.tmp[4:]
            self.tmp[12:] = [self.ball.position[0], self.ball.position[1], self.ball.xdot, self.ball.ydot]
            return list(self.tmp)
        else:
            return [self.ball.position[0], self.ball.position[1], self.ball.xdot, self.ball.ydot]

    def take_action(self, action):
        """ Take a step in the environment
        :param action: The action to apply over the ball
        """
        for i in range(20):
            if i == 0:
                if self.args['continuous_action']:
                    action = np.array(action)
                    self.ball.add_impulse(action[0], action[1])
                else:
                    self.ball.add_impulse(*self.action_effects[action])

            self.ball.step()

            # Detect collisions
            ncollision = 0
            dxdy = np.array([0, 0])

            for obs in self.obstacles:
                if obs.collision(self.ball):
                    dxdy = dxdy + obs.collision_effect(self.ball)
                    ncollision += 1

            if ncollision == 1:
                self.ball.xdot = dxdy[0]
                self.ball.ydot = dxdy[1]
                if i == 19:
                    self.ball.step()
            elif ncollision > 1:
                self.ball.xdot = -self.ball.xdot
                self.ball.ydot = -self.ball.ydot

            if self.episode_ended():
                return self.END_EPISODE

        self.ball.add_drag()
        self._check_bounds()

        if not self.args['continuous_action'] and action == self.ACC_NONE:
            return self.STEP_PENALTY

        return self.THRUST_PENALTY

    def step(self, action):
        reward = self.take_action(action)
        next_state = self.get_state()
        done = self.episode_ended()
        return next_state, reward, done, {}

    def reset(self):
        self.tmp = np.zeros(16)
        if self.args['random_start']:
            while True:
                x = round(random.uniform(0.01, 0.99), 2)
                y = round(random.uniform(0.01, 0.99), 2)
                self.ball = BallModel(list([x, y]), self.ball_rad)
                is_collison = False
                for obs in self.obstacles:
                    if obs.isPointinPolygon(self.ball) or self.episode_ended():
                        is_collison = True
                        break
                if not is_collison:
                    break
        else:
            self.ball = BallModel(list(random.choice(self.start_pos)), self.ball_rad)

        #self.ball = BallModel(list([0.1, 0.01]), self.ball_rad)
        if self.args['run_test']:
            self.quit_view()
            self.build_view()

        #for obs in self.obstacles:
        #    print(obs.isPointinPolygon(self.ball))
        #print(self.ball.position)
        #return [self.ball.position[0], self.ball.position[1], self.ball.xdot, self.ball.ydot]
        #print(self.get_state())
        return self.get_state()

    def render(self):
        if not self.args['run_test']:
            return
        pygame.time.wait (50)
        self.environment_view.blit()
        pygame.display.flip()

    def episode_ended(self):
        """ Find out if the ball reached the target
        :returns: True if the ball reched the target position
        :rtype: bool
        """
        return np.linalg.norm(np.array(self.ball.position) - np.array(self.target_pos)) < self.target_rad

    def _check_bounds(self):
        """ Make sure that the ball stays within the environment """
        if self.ball.position[0] > 1.0:
            self.ball.position[0] = 0.95
        if self.ball.position[0] < 0.0:
            self.ball.position[0] = 0.05
        if self.ball.position[1] > 1.0:
            self.ball.position[1] = 0.95
        if self.ball.position[1] < 0.0:
            self.ball.position[1] = 0.05


class PinballView:
    """ This class displays a :class:`PinballModel`

    This class is used in conjunction with the :func:`run_pinballview`
    function, acting as a *controller*.

    We use `pygame <http://www.pygame.org/>` to draw the environment.

    """

    def __init__(self, screen, model):
        """
        :param screen: a pygame surface
        :type screen: :class:`pygame.Surface`
        :param model: an instance of a :class:`PinballModel`
        :type model: :class:`PinballModel`
        """
        self.screen = screen
        self.model = model

        self.DARK_GRAY = [64, 64, 64]
        self.LIGHT_GRAY = [232, 232, 232]
        self.BALL_COLOR = [0, 0, 255]
        self.TARGET_COLOR = [255, 0, 0]

        # Draw the background
        self.background_surface = pygame.Surface(screen.get_size())
        self.background_surface.fill(self.LIGHT_GRAY)
        for obs in model.obstacles:
            pygame.draw.polygon(self.background_surface, self.DARK_GRAY, list(map(self._to_pixels, obs.points)), 0)

        pygame.draw.circle(
            self.background_surface, self.TARGET_COLOR, self._to_pixels(self.model.target_pos),
            int(self.model.target_rad * self.screen.get_width()))

    def _to_pixels(self, pt):
        """ Converts from real units in the 0-1 range to pixel units
        :param pt: a point in real units
        :type pt: list
        :returns: the input point in pixel units
        :rtype: list
        """
        return [int(pt[0] * self.screen.get_width()), int(pt[1] * self.screen.get_height())]

    def blit(self):
        """ Blit the ball onto the background surface """
        self.screen.blit(self.background_surface, (0, 0))
        pygame.draw.circle(self.screen, self.BALL_COLOR,
                           self._to_pixels(self.model.ball.position),
                           int(self.model.ball.radius * self.screen.get_width()))


def run_pinballview(width, height, configuration, args):
    """ Controller function for a :class:`PinballView`

    :param width: The desired screen width in pixels
    :type widht: int
    :param height: The desired screen height in pixels
    :type height: int
    :param configuration: The path to a configuration file for a :class:`PinballModel`
    :type configuration: str

    """
    # Launch interactive pygame
    environment = PinballModel(args)

    step = 0
    reward = 0
    episode_reward = 0
    state = environment.reset ()
    start = state
    while True:
        # pygame.time.wait(10)
        rand = np.random.uniform(-20, 20)
        rand2 = np.random.uniform(-20, 20)
        # print(rand, rand2)
        # print (state)
        a = np.clip(np.array([rand, rand2]), -1, 1)
        state, r, done = environment.step(a)
        # environment.render()
        step += 1
        reward += r
        episode_reward = episode_reward + r * np.power(0.96, step)
        if done or step >= 500:
            print(step, reward, episode_reward, start)
            break
    # environment.quit_view()
    # pygame.quit ()


def get_env(args):
    run_pinballview(args["width"], args["height"], args["configuration"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser (description='Pinball domain')
    parser.add_argument('--configuration', default="pinball_hard_single.cfg",
                        help='the configuration file')
    parser.add_argument('--width', action='store', type=int,
                        default=500, help='screen width (default: 500)')
    parser.add_argument('--height', action='store', type=int,
                        default=500, help='screen height (default: 500)')
    parser.add_argument('-r', '--rlglue', action='store_true', help='expose the environment through RL-Glue')
    args = parser.parse_args()
    dic = dict()
    dic['width'] = args.width
    dic['height'] = args.height
    dic['configuration'] = args.configuration
    dic['random_start'] = False
    dic['target_position'] = [0.9, 0.2]
    dic['start_position'] = [[0.2, 0.9]]
    dic['run_test'] = True
    dic['continuous_action'] = True
    dic['sequential_state'] = False
    for i in range(5000):
        run_pinballview(args.width, args.height, args.configuration, dic)





