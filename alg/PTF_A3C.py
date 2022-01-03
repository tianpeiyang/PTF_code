import tensorflow as tf
import numpy as np
from alg.optimizer import Optimizer
import multiprocessing
from alg.source_actor import SourceActor as SA
from alg.PRM_actor import PRM_actor as PRM_AC
from util.ReplayBuffer import ReplayBuffer
import tensorflow.contrib.layers as layers


class ACNet(object):
    def __init__(self, scope, args, sess, logger, ot, globalAC=None):
        self.SESS = sess
        self.args = args
        self.logger = logger
        self.OT = ot
        self.N_S = self.args['features']
        self.N_A = self.args['action_dim']
        self.name = scope
        opt1 = Optimizer(args['optimizer'], args['learning_rate_a'])
        self.OPT_A = opt1.get_optimizer()
        opt2 = Optimizer(args['optimizer'], args['learning_rate_c'])
        self.OPT_C = opt2.get_optimizer()

        if scope == self.args['GLOBAL_NET_SCOPE']:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
                #self.a_params, self.c_params = self._build_net(scope)[-2:]
                self.a_prob_g, self.v_g, self.a_params, self.c_params = self._build_net(scope)
                if self.args['continuous_action']:
                    self.A_G = tf.squeeze(self.a_prob_g.sample(1), axis=0)
                else:
                    self.A_G = self.a_prob_g
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.t = tf.placeholder(tf.float32, [None, 1], 't')
                self.e = tf.placeholder(tf.float32, (), 'e')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td)) # critic的loss是平方loss
                    #print("c_loss:", self.c_loss)
                with tf.name_scope('a_loss'):
                    # Q * log（
                    if self.args['continuous_action']:
                        self.mu = tf.placeholder(tf.float32, [None, self.N_A], 'output_mu')
                        self.sigma = tf.placeholder(tf.float32, [None, self.N_A], 'output_sigma')
                        self.a_his = tf.placeholder(tf.float32, [None, self.N_A], 'A')
                        self.A = tf.squeeze(self.a_prob.sample(1), axis=0)
                        log_prob = self.a_prob.log_prob(self.a_his)
                        entropy = self.a_prob.entropy()
                        otherNormal = tf.distributions.Normal(self.mu, self.sigma)
                        otherEntropy = otherNormal.cross_entropy(self.a_prob)
                        #otherEntropy = -otherNormal.prob(self.a_his) * tf.log(self.a_prob.prob(self.a_his) + 1e-5)
                        self.oe = [[self.mu, self.sigma], self.out]
                    else:
                        self.s_a_prob = tf.placeholder(tf.float32, [None, self.N_A], 's_a_prob')
                        self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                        log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-10) *
                                                 tf.one_hot(self.a_his, self.N_A, dtype=tf.float32),
                                                 axis=1, keepdims=True)

                        entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-10),
                                                 axis=1, keepdims=True)  # encourage exploration
                        otherEntropy = -self.s_a_prob * tf.log(self.a_prob + 1e-10)

                    exp_v_t = log_prob * tf.stop_gradient(td)  # 这里的td不再求导，当作是常数
                    self.exp_v_t = tf.reduce_mean(exp_v_t)

                    entropy = tf.reduce_mean(entropy)
                    self.entropy = args['ENTROPY_BETA'] * entropy

                    entropyTS = tf.reduce_sum(otherEntropy, axis=1, keepdims=True)
                    #entropyTS = -tf.reduce_sum((1 - self.t) * self.s_a_prob * tf.log(self.a_prob + 1e-10), axis=1, keepdims=True)
                    weight = 0.5 + tf.tanh(3 - self.args['c3'] * self.e) / 2
                    entropyTS = entropyTS * weight * self.args['c1']
                    self.entropyTS = tf.reduce_mean(entropyTS)

                    self.exp_v = self.entropy + self.exp_v_t
                    self.a_loss = - self.exp_v + self.entropyTS

                with tf.name_scope('localx_grad'):
                    '''
                    self.a_grads = self.OPT_A.compute_gradients(self.a_loss, var_list=self.a_params)
                    for i, (grad, var) in enumerate(self.a_grads):
                        if grad is not None:
                            self.a_grads[i] = (tf.clip_by_norm(grad, args['clip_value']), var)
                    self.c_grads = self.OPT_C.compute_gradients(self.c_loss, var_list=self.c_params)
                    for i, (grad, var) in enumerate(self.c_grads):
                        if grad is not None:
                            self.c_grads[i] = (tf.clip_by_norm(grad, args['clip_value']), var)
                    '''

                    a_grads= tf.gradients(self.a_loss, self.a_params)
                    self.a_grads, _ = tf.clip_by_global_norm(a_grads, args['clip_value'])
                    c_grads = tf.gradients(self.c_loss, self.c_params)
                    self.c_grads, _ = tf.clip_by_global_norm(c_grads, args['clip_value'])

                    #self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    #self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'): # 把主网络的参数赋予各子网络
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'): # 使用子网络的梯度对主网络参数进行更新
                    self.update_a_op = self.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    #self.update_a_op = self.OPT_A.apply_gradients(self.a_grads)
                    #self.update_c_op = self.OPT_C.apply_gradients(self.c_grads)

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .01)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, self.args['n_layer_a_1'], tf.nn.relu6, kernel_initializer=w_init, name='la')
            if self.args['continuous_action']:
                mu = self.args['action_clip'] * tf.layers.dense(l_a, self.N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a, self.N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                sigma = sigma
                self.out = [mu, sigma]
                a_prob = tf.distributions.Normal(loc=mu, scale=sigma)
            else:
                a_prob = tf.layers.dense(l_a, self.N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap') # 得到每个动作的选择概率
            #a_prob = tf.layers.dense(l_a, self.N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap') # 得到每个动作的选择概率
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, self.args['n_layer_c_1'], tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # 得到每个状态的价值函数
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        #print("entropy:", SESS.run([self.a_loss], feed_dict))
        return self.SESS.run (
            [self.a_loss, self.c_loss, self.exp_v_t, self.entropy, self.entropyTS, self.a_grads, self.c_grads, self.update_a_op, self.update_c_op],
            feed_dict)  # local grads applies to global net

    def update(self, actor, buffer_s, buffer_a, buffer_r, buffer_o, buffer_t, done, s_, t, epi):
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

        #term_probs = self.OT.get_term_prob(buffer_s)

        buffer_a_prob_target = []
        mu = []
        sigma = []
        #buffer_t = []
        for i, o in enumerate(buffer_o):
            o = int(o)
            if self.args['continuous_action']:
                a_prob = actor[o].choose_acton_prob(buffer_s[i])
                mu.append(a_prob[0])
                sigma.append(a_prob[1])
            else:
                a_prob = actor[o].choose_acton_prob(buffer_s[i])
                buffer_a_prob_target.append(a_prob)
        if self.args['continuous_action']:
            buffer_s, buffer_a, buffer_v_target, buffer_t, buffer_mu, buffer_sigma = np.vstack(buffer_s), np.array(
                buffer_a), np.vstack(buffer_v_target), np.array(buffer_t), np.vstack(mu), np.vstack(sigma)
            feed_dict = {self.s: buffer_s, self.a_his: buffer_a, self.v_target: buffer_v_target, self.t: buffer_t,
                         self.mu: buffer_mu, self.sigma: buffer_sigma, self.e: epi}
        else:
            buffer_s, buffer_a, buffer_v_target, buffer_t, buffer_a_prob_target = np.vstack(buffer_s), np.array(
                buffer_a), np.vstack(buffer_v_target), np.array(buffer_t), np.vstack(buffer_a_prob_target)
            feed_dict = {self.s: buffer_s, self.a_his: buffer_a, self.v_target: buffer_v_target, self.t: buffer_t,
                         self.s_a_prob: buffer_a_prob_target, self.e: epi}
        a_loss, c_loss, v, e, ets, a_grads, c_grads, _, __ = self.update_global(feed_dict)
        #if epi % 10 == 0:
            #print('loss: ', self.name, a_loss, c_loss, v, e, ets)
        self.logger.write_tb_log('a_loss', a_loss, t)
        self.logger.write_tb_log('c_loss', c_loss, t)
        self.logger.write_tb_log('exp_v_t', v, t)
        self.logger.write_tb_log('entropy', e, t)
        self.logger.write_tb_log('entropyTS', ets, t)
        #self.logger.write_tb_log('a_grads', a_grads, t)
        #self.logger.write_tb_log('c_grads', c_grads, t)
        self.pull_global()

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


class Option:
    def __init__(self, option_dim, n_features, args, sess, logger):
        self.args = args
        self.option_dim = option_dim
        self.n_features = n_features
        self.logger = logger

        self.update_step = 0
        self.run_step = 0
        self.replace_target_iter = args['replace_target_iter']
        self.e_greedy = args['e_greedy']
        self.epsilon_increment = args['e_greedy_increment']
        self.epsilon = args['start_greedy'] if args['e_greedy_increment'] != 0 else self.e_greedy

        opt0 = Optimizer(args['optimizer'], args['learning_rate_o'])
        self.Opt_O = opt0.get_optimizer()
        opt1 = Optimizer(args['optimizer'], args['learning_rate_t'])
        self.Opt_T = opt1.get_optimizer()

        self.replay_buffer = ReplayBuffer(args['memory_size'])

        with tf.variable_scope('train_input'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
            self.option_o = tf.placeholder(tf.int32, [None])
            self.option_a_t = tf.placeholder(tf.float32, [None, self.option_dim])
            self.reward = tf.placeholder(tf.float32, [None])
            self.done = tf.placeholder(tf.float32, [None])

        self.q_omega_current, self.term_current = self._build_net('q_net', self.s)
        self.q_omega_target, self.term_target = self._build_net('q_target', self.s_)
        self.q_omega_next_current, self.term_next_current = self._build_net('q_net', self.s_, reuse=True)

        self.q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net')
        self.target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')

        with tf.variable_scope('q_omega_value'):
            term_val_next = tf.reduce_sum(self.term_next_current * tf.one_hot(self.option_o, self.option_dim), axis=-1)
            q_omega_val_next = tf.reduce_sum(self.q_omega_next_current * tf.one_hot(self.option_o, self.option_dim), axis=-1)
            max_q_omega_next = tf.reduce_max(self.q_omega_next_current, axis=-1)
            max_q_omega_next_targ = tf.reduce_sum(
                self.q_omega_target * tf.one_hot(tf.argmax(self.q_omega_next_current, axis=-1), self.option_dim), axis=-1)

        with tf.variable_scope('q_omega_loss'):
            u_next_raw = (1 - self.term_next_current) * self.q_omega_target + self.term_next_current * max_q_omega_next_targ[..., None]
            u_next = tf.stop_gradient(u_next_raw * (1 - self.done)[..., None])
            self.q_omega_loss = tf.reduce_mean(tf.reduce_sum(self.option_a_t
                * tf.losses.mean_squared_error(self.reward[..., None] + self.args['reward_decay'] * u_next, self.q_omega_current, reduction=tf.losses.Reduction.NONE), axis=-1))

        with tf.variable_scope('term_loss'):
            if self.args['xi'] == 0:
                xi = 0.8 * (max_q_omega_next - tf.nn.top_k(self.q_omega_next_current, 2)[0][:, 1])
            else:
                xi = self.args['xi']
            advantage_go = q_omega_val_next - max_q_omega_next + xi
            advantage = tf.stop_gradient(advantage_go)
            # total_error_term = term_val_next * advantage * (1 - done_mask_ph)
            self.total_error_term = term_val_next * advantage

        with tf.name_scope('grad'):

            gradients = self.Opt_O.compute_gradients(self.q_omega_loss, var_list=self.q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, args['clip_value']), var)
            self.update_o = self.Opt_O.apply_gradients(gradients)
            gradients_t = self.Opt_T.compute_gradients(self.total_error_term, var_list=self.q_func_vars)
            for i, (grad, var) in enumerate(gradients_t):
                if grad is not None:
                    gradients_t[i] = (tf.clip_by_norm(grad, args['clip_value']), var)
            self.update_t = self.Opt_T.apply_gradients(gradients_t)

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.target_q_func_vars, self.q_func_vars)]
        self.sess = sess

    def _build_net(self, scope, s, reuse=False):
        w_init = tf.random_normal_initializer(0., .01)
        with tf.variable_scope(scope, reuse=reuse):
            l_a = tf.layers.dense(s, self.args['option_layer_1'], tf.nn.relu6, kernel_initializer=w_init, name='la')
            with tf.variable_scope("option_value"):
                q_omega = tf.layers.dense(l_a, self.option_dim, tf.nn.tanh, kernel_initializer=w_init, name='omega_value')
            with tf.variable_scope("termination_prob"):
                term_prob = tf.layers.dense(l_a, self.option_dim, tf.sigmoid, kernel_initializer=w_init, name='term_prob')
        return q_omega, term_prob

    def store_transition(self, observation, action, reward, done, observation_, opa):
        self.replay_buffer.add(observation, action, reward, done, observation_, opa)

    def update_e(self):
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.e_greedy else self.e_greedy

    def choose_o(self, s):
        if np.random.uniform() < self.epsilon:
            options = self.sess.run(self.q_omega_current, feed_dict={self.s: s[np.newaxis, :]})
            options = options[0]
            # print(options)
            return np.argmax(options)
        else:
            return np.random.randint(0, self.option_dim)

    def get_t(self, s_, option):
        terminations = self.sess.run(self.term_next_current, feed_dict={self.s_: s_[np.newaxis, :]})
        return terminations[0][option]

    def get_term_prob(self, s):
        return self.sess.run(self.term_current, feed_dict={self.s: s})

    def update(self, observation, option, done, observation_):
        if self.update_step % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
        for i in range(self.args['learning_times']):
            self.update_step += 1

            if not done:
                loss_term, _ = self.sess.run([self.total_error_term, self.update_t], feed_dict={
                    self.s: observation[np.newaxis, :],
                    self.option_o: [option],
                    # opa_t_ph: [option_running],
                    # rew_t_ph: [r],
                    self.s_: observation_[np.newaxis, :],
                    self.done: [1.0 if done == True else 0.0]
                })
                self.logger.write_tb_log('t_loss', loss_term[0], self.update_step)

            minibatch = self.replay_buffer.get_batch(self.args['option_batch_size'])
            state_batch = np.asarray([data[0] for data in minibatch])
            action_batch = np.asarray([data[1] for data in minibatch])
            reward_batch = np.asarray([data[2] for data in minibatch])
            done_batch = np.array([1.0 if data[3] else 0.0 for data in minibatch], dtype=np.float32)
            next_state_batch = np.asarray([data[4] for data in minibatch])
            opa_batch = np.asarray([data[5] for data in minibatch])

            loss_q_omega, _ = self.sess.run([self.q_omega_loss, self.update_o], feed_dict={
                self.s: state_batch,
                self.reward: reward_batch,
                self.s_: next_state_batch,
                self.option_o: [option],
                self.done: done_batch,
                self.option_a_t: opa_batch
            })
            self.logger.write_tb_log('o_loss', loss_q_omega, self.update_step)


class PTF_A3C:
    def __init__(self, n_actions, features, args, logger):
        self.args = args
        self.SESS = tf.Session()
        if args['USE_CPU_COUNT']:
            self.N_WORKERS = multiprocessing.cpu_count()
        else:
            self.N_WORKERS = args['N_WORKERS']
        SOURCE_TASK = args['option_model_path']
        N_O = len(SOURCE_TASK)
        print(N_O)
        g = [tf.Graph() for i in range(N_O)]
        actor_sess = [tf.Session(graph=i) for i in g]
        self.actor = []
        for i in range(len(SOURCE_TASK)):
            with actor_sess[i].as_default():
                with g[i].as_default():
                    dqn = SA(SOURCE_TASK[i], args, actor_sess[i])
                    self.actor.append(dqn)

        with tf.device("/cpu:0"):
            self.OT = Option(N_O, features, args, self.SESS, logger)
            self.GLOBAL_AC = ACNet(args['GLOBAL_NET_SCOPE'], args, self.SESS, logger, self.OT)  # we only need its params
            self.thread_AC = []
            # Create worker
            for i in range(self.N_WORKERS):
                i_name = 'W_%i' % i  # worker name
                self.thread_AC.append(ACNet(i_name, args, self.SESS, logger, self.OT, self.GLOBAL_AC))

        # Coordinator类用来管理在Session中的多个线程，
        # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
        self.COORD = tf.train.Coordinator()
        self.SESS.run(tf.global_variables_initializer())
        self.variables = tf.global_variables()
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
