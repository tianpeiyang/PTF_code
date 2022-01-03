import tensorflow as tf
import numpy as np
from alg.optimizer import Optimizer
import tensorflow.contrib.layers as layers
from util.ReplayBuffer import ReplayBuffer


class CAPS:
    def __init__(self, n_action, n_features, args, logger):
        self.args = args
        if args['continuous_action']:
            self.option_dim = len(args['option_model_path']) + 5
        else:
            self.option_dim = len(args['option_model_path']) + n_action
        self.n_features = n_features
        self.logger = logger

        self.update_step = 0
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
            #self.o_grads = tf.gradients(self.q_omega_loss, self.q_func_vars)
            #self.t_grads = tf.gradients(self.total_error_term, self.q_func_vars)
            #self.update_o = self.Opt_O.apply_gradients(zip(self.o_grads, self.q_func_vars))
            #self.update_t = self.Opt_T.apply_gradients(zip(self.t_grads, self.q_func_vars))

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.target_q_func_vars, self.q_func_vars)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, scope, s, reuse=False):
        w_init = tf.random_normal_initializer(0., .01)
        with tf.variable_scope(scope, reuse=reuse):
            l_a = tf.layers.dense(s, self.args['option_layer_1'], tf.nn.relu6, kernel_initializer=w_init, name='la')
            with tf.variable_scope("option_value"):
                q_omega = tf.layers.dense(l_a, self.option_dim, tf.nn.tanh, kernel_initializer=w_init, name='omega_value')
                #q_omega = layers.fully_connected(s, num_outputs=self.option_dim, activation_fn=None)
            with tf.variable_scope("termination_prob"):
                term_prob = tf.layers.dense(l_a, self.option_dim, tf.sigmoid, kernel_initializer=w_init,
                                          name='term_prob')
                #term_prob = layers.fully_connected(s, num_outputs=self.option_dim, activation_fn=tf.sigmoid)
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

    def update(self, observation, option, reward, done, observation_):
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
                self.logger.write_tb_log('t_loss', loss_term, self.update_step)

            minibatch = self.replay_buffer.get_batch(self.args['batch_size'])
            state_batch = np.asarray([data[0] for data in minibatch])
            action_batch = np.asarray([data[1] for data in minibatch])
            reward_batch = np.asarray([data[2] for data in minibatch])
            done_batch = np.array([1.0 if data[3] else 0.0 for data in minibatch], dtype=np.float32)
            next_state_batch = np.asarray([data[4] for data in minibatch])
            opa_batch = np.asarray([data[5] for data in minibatch])
            #print(done_batch)

            loss_q_omega, _ = self.sess.run([self.q_omega_loss, self.update_o], feed_dict={
                self.s: state_batch,
                self.reward: reward_batch,
                self.s_: next_state_batch,
                self.option_o: [option],
                self.done: done_batch,
                self.option_a_t: opa_batch
            })

            self.logger.write_tb_log('o_loss', loss_q_omega, self.update_step)

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path + ".ckpt")

    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path + ".ckpt")



