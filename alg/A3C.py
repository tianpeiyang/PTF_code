import tensorflow as tf
import numpy as np
from alg.optimizer import Optimizer
import multiprocessing


class ACNet (object):
    def __init__(self, scope, args, sess, logger, globalAC=None):
        self.SESS = sess
        self.args = args
        self.logger = logger
        self.N_S = self.args['features']
        self.N_A = self.args['action_dim']
        self.name = scope
        opt1 = Optimizer (args['optimizer'], args['learning_rate_a'])
        self.OPT_A = opt1.get_optimizer()
        opt2 = Optimizer (args['optimizer'], args['learning_rate_c'])
        self.OPT_C = opt2.get_optimizer()

        if scope == self.args['GLOBAL_NET_SCOPE']:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
                # self.a_params, self.c_params = self._build_net(scope)[-2:]
                self.a_prob_g, self.v_g, self.a_params, self.c_params = self._build_net(scope)
                if self.args['continuous_action']:
                    self.A_G = tf.squeeze(self.a_prob_g.sample (1), axis=0)
                else:
                    self.A_G = self.a_prob_g
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.t = tf.placeholder(tf.float32, [None, 1], 't')
                self.e = tf.placeholder(tf.float32, (), 'e')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net (scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))  # critic的loss是平方loss
                    # print("c_loss:", self.c_loss)
                with tf.name_scope ('a_loss'):
                    # Q * log（
                    if self.args['continuous_action']:
                        self.mu = tf.placeholder(tf.float32, [None, self.N_A], 'output_mu')
                        self.sigma = tf.placeholder(tf.float32, [None, self.N_A], 'output_sigma')
                        self.a_his = tf.placeholder(tf.float32, [None, self.N_A], 'A')
                        self.A = tf.squeeze(self.a_prob.sample(1), axis=0)
                        log_prob = self.a_prob.log_prob(self.a_his)
                        entropy = self.a_prob.entropy()
                        self.oe = [[self.mu, self.sigma], self.out]
                    else:
                        self.s_a_prob = tf.placeholder(tf.float32, [None, self.N_A], 's_a_prob')
                        self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                        log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-10) * tf.one_hot(self.a_his, self.N_A, dtype=tf.float32), axis=1, keepdims=True)

                        entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-10), axis=1, keepdims=True)  # encourage exploration

                    exp_v_t = log_prob * tf.stop_gradient(td)  # 这里的td不再求导，当作是常数
                    self.exp_v_t = tf.reduce_mean(exp_v_t)

                    entropy = tf.reduce_mean(entropy)
                    self.entropy = args['ENTROPY_BETA'] * entropy

                    self.exp_v = self.entropy + self.exp_v_t
                    self.a_loss = - self.exp_v

                with tf.name_scope('localx_grad'):

                    a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.a_grads, _ = tf.clip_by_global_norm(a_grads, args['clip_value'])
                    c_grads = tf.gradients(self.c_loss, self.c_params)
                    self.c_grads, _ = tf.clip_by_global_norm(c_grads, args['clip_value'])

                    # self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    # self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):  # 把主网络的参数赋予各子网络
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):  # 使用子网络的梯度对主网络参数进行更新
                    self.update_a_op = self.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    # self.update_a_op = self.OPT_A.apply_gradients(self.a_grads)
                    # self.update_c_op = self.OPT_C.apply_gradients(self.c_grads)

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .01)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, self.args['n_layer_a_1'], tf.nn.relu6, kernel_initializer=w_init, name='la')
            if self.args['continuous_action']:
                mu = self.args['action_clip'] * tf.layers.dense(l_a, self.N_A, tf.nn.tanh, kernel_initializer=w_init,
                                                                name='mu')
                sigma = tf.layers.dense(l_a, self.N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                sigma = sigma
                self.out = [mu, sigma]
                a_prob = tf.distributions.Normal(loc=mu, scale=sigma)
            else:
                a_prob = tf.layers.dense(l_a, self.N_A, tf.nn.softmax, kernel_initializer=w_init,
                                          name='ap')  # 得到每个动作的选择概率
            # a_prob = tf.layers.dense(l_a, self.N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap') # 得到每个动作的选择概率
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, self.args['n_layer_c_1'], tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # 得到每个状态的价值函数
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        # print("entropy:", SESS.run([self.a_loss], feed_dict))
        return self.SESS.run(
            [self.a_loss, self.c_loss, self.exp_v_t, self.entropy, self.a_grads, self.c_grads,
             self.update_a_op, self.update_c_op],
            feed_dict)  # local grads applies to global net

    def update(self, buffer_s, buffer_a, buffer_r, done, s_, t, epi):
        # print(r, GLOBAL_EP, total_step, self.name)
        if done:
            v_s_ = 0  # terminal
        else:
            v_s_ = self.SESS.run(self.v, {self.s: s_[np.newaxis, :]})[0, 0]
        buffer_v_target = []

        for r in buffer_r[::-1]:  # reverse buffer r
            v_s_ = r + self.args['reward_decay'] * v_s_  # 使用v(s) = r + v(s+1)计算target_v
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
        feed_dict = {
            self.s: buffer_s,
            self.a_his: buffer_a,
            self.v_target: buffer_v_target,
        }
        a_loss, c_loss, exp_v, entropy, _, _, _, _ = self.update_global(feed_dict)
        self.logger.write_tb_log('a_loss', a_loss, t)
        self.logger.write_tb_log('c_loss', c_loss, t)
        self.logger.write_tb_log('exp_v_t', exp_v, t)
        self.logger.write_tb_log('entropy', entropy, t)

        self.pull_global()
        return a_loss, c_loss, exp_v, entropy

    def pull_global(self):  # run by a local
        self.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        if self.args['continuous_action']:
            action = self.SESS.run(self.A, feed_dict={self.s: s[np.newaxis, :]})[0]
            action = np.clip(action, -self.args['action_clip'], self.args['action_clip'])
        else:
            prob_weights = self.SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
            action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def choose_prob(self, s):  # run by a local
        if self.args['continuous_action']:
            action = self.SESS.run(self.out, feed_dict={self.s: s[np.newaxis, :]})
            return [action[0][0], action[1][0]]
        else:
            return self.SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})

    def choose_action_g(self, s):
        prob_weights = self.SESS.run(self.A_G, feed_dict={self.s: s[np.newaxis, :]})
        if self.args['continuous_action']:
            action = np.clip(prob_weights[0], -self.args['action_clip'], self.args['action_clip'])
        else:
            action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def is_from_source_actor(self, observation, action):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        if self.args['continuous_action']:
            actions_value = self.SESS.run (self.a_prob, feed_dict={self.s: observation})
            mu, sigma = actions_value[0][0], actions_value[1][0]
            #mu, sigma = action[0], action[1]
            #Entropy = self.SESS.run(self.Entropy, feed_dict={self.s: observation, self.input_mu: [mu], self.input_sigma: [sigma]})
            #if Entropy < 5:
            #    return True
            #else:
            #    return False
            is_in = True
            for i in range(len(action)):
                if action[i] < mu[i] - sigma[i] * 1 or action[i] > sigma[i] * 1 + mu[i]:
                    is_in = False
                    break
            return is_in

        else:
            prob_weights = self.SESS.run(self.A_G, feed_dict={self.s: observation})
            a = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
            if a == action:
                return True
            else:
                return False


class A3C:
    def __init__(self, n_actions, features, args, logger):
        self.args = args
        self.SESS = tf.Session()
        if args['USE_CPU_COUNT']:
            self.N_WORKERS = multiprocessing.cpu_count()
        else:
            self.N_WORKERS = args['N_WORKERS']
        print(self.N_WORKERS)

        with tf.device("/cpu:0"):
            self.GLOBAL_AC = ACNet(args['GLOBAL_NET_SCOPE'], args, self.SESS, logger)  # we only need its params
            self.thread_AC = []
            # Create worker
            for i in range(self.N_WORKERS):
                i_name = 'W_%i' % i  # worker name
                self.thread_AC.append(ACNet(i_name, args, self.SESS, logger, self.GLOBAL_AC))

        # Coordinator类用来管理在Session中的多个线程，
        # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
        self.COORD = tf.train.Coordinator()
        self.SESS.run(tf.global_variables_initializer())
        self.variables = tf.global_variables()
        '''
        if args['load_model']:
            self.load_model(args['load_model_path'])
            for i in range(self.N_WORKERS):
                self.thread_AC[i].pull_global()
        '''
        # SESS.graph.finalize()

    def save_model(self, path):
        variables_to_restore = [v for v in self.variables if v.name.split('/')[0] == self.args['GLOBAL_NET_SCOPE']]
        saver = tf.train.Saver(variables_to_restore)
        saver.save(self.SESS, path + ".ckpt")

    def load_model(self, path):
        variables_to_restore = [v for v in self.variables if v.name.split('/')[0] == self.args['GLOBAL_NET_SCOPE']and
                                v.name.split('/')[-1].split('_')[0] != 'Adam']
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(self.SESS, path + ".ckpt")

    def choose_action(self, s):
        return self.GLOBAL_AC.choose_action_g(s)
