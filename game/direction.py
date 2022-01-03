import numpy as np
from matplotlib import pyplot as plt
from alg.DQN import DeepQNetwork
from game.grid_game import getEnv

'''
This is used to generate a map with directions(actions) of each state
Reload DQN model
Plot the action in each state in the grid world
'''

num_rows = 21  # number of rows
num_cols = 24  # number of columns

TASK = 459

dic = {}
dic['configuration'] = '../grid_layout/hard_layout.lay'
dic['task'] = 459
dic['ui_size'] = 40
dic['default_reward'] = 0
dic['default_type'] = 0
dic['done_reward'] = 5
dic['random_start'] = False
dic['start_position'] = (1, 1)
env = getEnv(dic)
N_S = env.observation_space.n
N_A = env.action_space.n
args = dict()
args["n_layer_1"] = 20
args["learning_rate"] = 3e-4
args["reward_decay"] = 0.99
args["e_greedy"] = 0.95
args["e_greedy_increment"] = 0.0005
args["replace_target_iter"] = 1000
args["memory_size"] = 200000
args["batch_size"] = 32
args["output_graph"] = False
args["graph_path"] = ' '
args["summary_output_times"] = 5000000
args["start_greedy"] = 0.0
args["optimizer"] = 'adam'
args["soft_update"] = False
DQN = DeepQNetwork(N_A, N_S, args, logger=None)
DQN.load_model("C://Users/yangtianpei/Desktop/ptf/source_policies/grid/459/459")


# The array image is going to be the output image to display
image = np.zeros((num_rows * 40, num_cols * 40), dtype=np.uint8)

types = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1), (0, 6, 1), (0, 7, 1), (0, 8, 1), (0, 9, 1),
         (0, 10, 1), (0, 11, 1), (0, 12, 1),
         (0, 13, 1), (0, 14, 1), (0, 15, 1), (0, 16, 1), (0, 17, 1), (0, 18, 1), (0, 19, 1), (0, 20, 1), (1, 0, 1),
         (1, 4, 1), (1, 7, 1), (1, 11, 1),
         (1, 16, 1), (1, 20, 1), (2, 0, 1), (2, 7, 1), (2, 11, 1), (2, 16, 1), (2, 20, 1), (3, 0, 1), (3, 4, 1),
         (3, 7, 1), (3, 11, 1), (3, 16, 1),
         (3, 20, 1), (4, 0, 1), (4, 4, 1), (4, 7, 1), (4, 8, 1), (4, 10, 1), (4, 11, 1), (4, 13, 1), (4, 14, 1),
         (4, 15, 1), (4, 16, 1), (4, 20, 1),
         (5, 0, 1), (5, 1, 1), (5, 2, 1), (5, 3, 1), (5, 4, 1), (5, 20, 1), (6, 0, 1), (6, 4, 1), (6, 16, 1),
         (6, 17, 1), (6, 18, 1), (6, 19, 1),
         (6, 20, 1), (7, 0, 1), (7, 7, 1), (7, 8, 1), (7, 9, 1), (7, 10, 1), (7, 11, 1), (7, 12, 1), (7, 13, 1),
         (7, 16, 1), (7, 20, 1), (8, 0, 1),
         (8, 4, 1), (8, 7, 1), (8, 10, 1), (8, 13, 1), (8, 16, 1), (8, 20, 1), (9, 0, 1), (9, 4, 1), (9, 10, 1),
         (9, 13, 1), (9, 20, 1), (10, 0, 1),
         (10, 4, 1), (10, 7, 1), (10, 10, 1), (10, 16, 1), (10, 20, 1), (11, 0, 1), (11, 4, 1), (11, 7, 1), (11, 8, 1),
         (11, 9, 1), (11, 10, 1),
         (11, 13, 1), (11, 16, 1), (11, 20, 1), (12, 0, 1), (12, 1, 1), (12, 2, 1), (12, 3, 1), (12, 4, 1), (12, 7, 1),
         (12, 10, 1), (12, 13, 1),
         (12, 16, 1), (12, 17, 1), (12, 18, 1), (12, 19, 1), (12, 20, 1), (13, 0, 1), (13, 4, 1), (13, 7, 1),
         (13, 10, 1), (13, 13, 1),
         (13, 20, 1), (14, 0, 1), (14, 4, 1), (14, 7, 1), (14, 10, 1), (14, 13, 1), (14, 16, 1), (14, 20, 1),
         (15, 0, 1), (15, 7, 1), (15, 10, 1),
         (15, 13, 1), (15, 16, 1), (15, 20, 1), (16, 0, 1), (16, 4, 1), (16, 7, 1), (16, 9, 1), (16, 10, 1),
         (16, 11, 1), (16, 12, 1), (16, 13, 1),
         (16, 16, 1), (16, 20, 1), (17, 0, 1), (17, 1, 1), (17, 3, 1), (17, 4, 1), (17, 16, 1), (17, 17, 1),
         (17, 19, 1), (17, 20, 1), (18, 0, 1),
         (18, 4, 1), (18, 16, 1), (18, 20, 1), (19, 0, 1), (19, 4, 1), (19, 7, 1), (19, 8, 1), (19, 9, 1), (19, 10, 1),
         (19, 11, 1), (19, 12, 1),
         (19, 13, 1), (19, 15, 1), (19, 16, 1), (19, 20, 1), (20, 0, 1), (20, 4, 1), (20, 7, 1), (20, 12, 1),
         (20, 16, 1), (20, 20, 1), (21, 0, 1),
         (21, 4, 1), (21, 12, 1), (21, 16, 1), (21, 20, 1), (22, 0, 1), (22, 4, 1), (22, 7, 1), (22, 12, 1),
         (22, 16, 1), (22, 20, 1), (23, 0, 1),
         (23, 1, 1), (23, 2, 1), (23, 3, 1), (23, 4, 1), (23, 5, 1), (23, 6, 1), (23, 7, 1), (23, 8, 1), (23, 9, 1),
         (23, 10, 1), (23, 11, 1), (23, 12, 1),
         (23, 13, 1), (23, 14, 1), (23, 15, 1), (23, 16, 1), (23, 17, 1), (23, 18, 1), (23, 19, 1), (23, 20, 1)]

# Generate the image for display
for row in range(0, num_rows):
    for col in range(0, num_cols):
        if (col, row, 1) in types:
            for j in range(40):
                image[40 * (num_rows - row - 1) + j, range(40 * col, 40 * col + 40)] = 255
directions = {0: (0, 1),
              1: (0, -1),
              2: (1, 0),
              3: (-1, 0),
              4: (1, 1),
              5: (-1, -1),
              6: (1, -1),
              7: (-1, 1)}
for row in range(0, num_rows):
    for col in range(0, num_cols):
        if (col, row, 1) not in types:
            one_hot_x = np.zeros(num_cols)
            one_hot_y = np.zeros(num_rows)
            one_hot_x[col] = 1
            one_hot_y[row] = 1
            one_hot = np.concatenate((one_hot_x, one_hot_y))

            walls = []

            for i in range(8):
                move = directions.get(i)
                position = np.sum(((col, row), move), axis=0)
                if (position[0], position[1], 1) in types:
                    walls.append(1)
                else:
                    walls.append(0)

            one_hot = np.concatenate((one_hot_x, one_hot_y, walls))
            stateonehot = np.array(one_hot)

            state = np.array([col, row])
            # a_prob = AC.action_prob(state)
            # action = np.argmax(a_prob)
            # if a_prob[action] < 0.7:
            # action = -1
            # print(state, a_prob, action)

            DQN.epsilon = 1
            action = DQN.choose_action(stateonehot)
            # action = np.random.randint(0, 4)
            # action = 0
            if action == 3:
                xy = (40 * col, 40 * (num_rows - row - 1) + 20)
                xytext = (40 * col + 40, 40 * (num_rows - row - 1) + 20)
                plt.annotate(s="", xy=xy, xytext=xytext, arrowprops={"arrowstyle": "<-"})
            if action == 2:
                xy = (40 * col + 20, 40 * (num_rows - row - 1))
                xytext = (40 * col + 20, 40 * (num_rows - row - 1) + 40)
                plt.annotate(s="", xy=xy, xytext=xytext, arrowprops={"arrowstyle": "->"})
            if action == 1:
                xy = (40 * col, 40 * (num_rows - row - 1) + 20)
                xytext = (40 * col + 40, 40 * (num_rows - row - 1) + 20)
                plt.annotate(s="", xy=xy, xytext=xytext, arrowprops={"arrowstyle": "->"})
            if action == 0:
                xy = (40 * col + 20, 40 * (num_rows - row - 1))
                xytext = (40 * col + 20, 40 * (num_rows - row - 1) + 40)
                plt.annotate(s="", xy=xy, xytext=xytext, arrowprops={"arrowstyle": "<-"})
            # image[10 * (num_rows-row-1)+j, range(10 * col, 10 * col + 10)] = 255

# print(image)
plt.imshow(image, interpolation='none')
plt.show()
