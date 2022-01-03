"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.7.3
"""
import sys
import numpy as np
import tensorflow as tf
from alg.optimizer import Optimizer

sys.path.append("../../")

tf.reset_default_graph()


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            args,
            logger
    ):
        self.args = args
        self.n_actions = n_actions
        self.n_features = n_features
        self.logger = logger
        self.lr = self.args['learning_rate']

        self.gamma = self.args['reward_decay']
        self.epsilon_max = self.args['e_greedy']
        self.replace_target_iter = self.args['replace_target_iter']
        self.memory_size = self.args['memory_size']
        self.batch_size = self.args['batch_size']
        self.output_graph = self.args['output_graph']
        self.graph_path = self.args['graph_path']
        self.learning_step = 0
        self.summary_output_times = self.args['summary_output_times']

        self.epsilon_increment = self.args['e_greedy_increment']
        self.epsilon = self.args['start_greedy'] if self.args['e_greedy_increment'] != 0 else self.epsilon_max

        self.n_layer_1 = self.args['n_layer_1']

        opt = Optimizer(self.args['optimizer'], self.args['learning_rate'])
        self.optimizer = opt.get_optimizer()

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # hard target update
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        '''
        # soft target update
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.args['target_network_decay'])
        self.target_update = ema.apply(e_params)
        target_net = [ema.average(x) for x in e_params]
        '''
        self.merged = tf.summary.merge_all()
        self.sess = tf.Session()

        #if self.output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            #self.summary_writer = tf.summary.FileWriter(args['results_path'] + args['graph_path'], self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.update_target()
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.n_layer_1, \
                tf.random_normal_initializer(0., 0.5), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1', reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            #with tf.variable_scope('l2', reuse=tf.AUTO_REUSE):
                #w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                #b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
                #l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer. collections is used later when assign to target net
            with tf.variable_scope('l2', reuse=tf.AUTO_REUSE):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
            self.net = [w1, b1, w2, b2]
            self.layer_1 = l1
            # regularizer loss of unit weights
            #self.regularizer_loss = self.regularizer*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval)) #+ self.regularizer_loss)
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            # self._train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=0.95, epsilon=0.01).minimize(self.loss)
            self._train_op = self.optimizer.minimize(self.loss)
            #self.grads = tf.gradients(self.loss, self.net)

            #tf.summary.histogram('grad', self.grads[0])
            #self._train_op = self.optimizer.apply_gradients(zip(self.grads, self.net))

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1', reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2', reuse=tf.AUTO_REUSE):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        # if not hasattr(self, 'memory_counter'):
        # self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
        # print(self.memory_counter)

    def choose_action(self, observation):
        observation = np.array(observation)
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        # print(self.epsilon)
        return action

    def update_target(self):
        #hard target update
        if not self.args['soft_update'] and self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
        elif self.args['soft_update']:
            #self.sess.run(self.target_update)
            self.sess.run(self.replace_target_op)

    def learn(self):
        # hard target update
        #if self.learn_step_counter % self.replace_target_iter == 0:
            #self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # train eval network
        _, cost, l1 = self.sess.run([self._train_op, self.loss, self.layer_1],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        if self.output_graph:
            self.learning_step += 1
            self.logger.write_tb_log('loss', cost, self.learning_step)
            self.logger.write_tb_log('l1', l1, self.learning_step)

        self.cost_his.append(cost)

        # update target
        self.update_target()

        # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return self.epsilon

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path + ".ckpt")

    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path + ".ckpt")

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def update_epsilon(self, episode):
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        if self.output_graph:
            self.logger.write_tb_log('epsilon', self.epsilon, episode)

    def is_from_source_actor(self, observation, action=None):
        return True
