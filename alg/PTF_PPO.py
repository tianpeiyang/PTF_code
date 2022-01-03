import copy
import numpy as np
import tensorflow as tf
from alg.optimizer import Optimizer
from alg.source_actor import SourceActor as SA
from alg.PRM_actor import PRM_actor as PRM_AC
from util.ReplayBuffer import ReplayBuffer


class PPO:
    def __init__(self, args, sess, logger):
        self.args = args
        self.n_actions = self.args['action_dim']
        self.n_features = self.args['features']
        self.logger = logger
        self.learning_step = 0

        self.obs = tf.placeholder(tf.float32, [None, self.n_features], 's')

        self.act_probs, self.policy_param = self.build_actor_net(self.args['policy'])
        self.o_act_probs, self.o_policy_param = self.build_actor_net(self.args['old_policy'])
        self.v_preds, self.v_param = self.build_critic_net('critic')

        if self.args['continuous_action']:
            self.sample_action = tf.squeeze(self.act_probs.sample(1), axis=0)
        else:
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])
            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

        self.replace_op = [tf.assign(t, e) for t, e in zip(self.o_policy_param, self.policy_param)]

        opt = Optimizer(args['optimizer'], args['learning_rate_a'])
        self.optimizer = opt.get_optimizer()
        opt_c = Optimizer(args['optimizer'], args['learning_rate_c'])
        self.optimizer_c = opt_c.get_optimizer()

        with tf.variable_scope('train_inp'):
            if self.args['continuous_action']:
                self.actions = tf.placeholder(tf.float32, [None, self.n_actions], 'action')
                self.mu = tf.placeholder(tf.float32, [None, self.n_actions], 'input_mu')
                self.sigma = tf.placeholder(tf.float32, [None, self.n_actions], 'input_sigma')
            else:
                self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
                self.s_a_prob = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name='s_a_prob')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='gaes')
            self.term = tf.placeholder(dtype=tf.float32, shape=[None], name='term')
            self.e = tf.placeholder(tf.float32, (), 'e')

        self.build_loss()

        self.sess = sess

    def build_actor_net(self, scope, trainable=True):
        with tf.variable_scope(scope):
            layer_1 = tf.layers.dense(inputs=self.obs, units=self.args['n_layer_a_1'], activation=tf.nn.relu, trainable=trainable)
            if self.args['continuous_action']:
                mu = self.args['action_clip'] * tf.layers.dense(inputs=layer_1, units=self.n_actions, activation=tf.nn.tanh, trainable=trainable)
                sigma = tf.layers.dense(inputs=layer_1, units=self.n_actions, activation=tf.nn.softplus, trainable=trainable)
                act_probs = tf.distributions.Normal(loc=mu, scale=sigma + 1e-9)
            else:
                act_probs = tf.layers.dense(inputs=layer_1, units=self.n_actions, activation=tf.nn.softmax)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return act_probs, params

    def build_critic_net(self, scope):
        with tf.variable_scope(scope):
            layer_1 = tf.layers.dense(inputs=self.obs, units=self.args['n_layer_c_1'], activation=tf.nn.relu)
            v_preds = tf.layers.dense(inputs=layer_1, units=1, activation=None)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return v_preds, params

    def build_loss(self):
        with tf.variable_scope('update_critic'):
            self.advantage = self.rewards - self.v_preds
            self.c_loss = tf.reduce_mean(tf.square(self.advantage))
            self.train_c_op = self.optimizer_c.minimize(self.c_loss, var_list=self.v_param)

        with tf.variable_scope('update_actor'):
            if self.args['continuous_action']:
                act_probs = self.act_probs.prob(self.actions)
                act_probs_old = self.o_act_probs.prob(self.actions)
                entropy = self.act_probs.entropy()
                otherNormal = tf.distributions.Normal(self.mu, self.sigma)
                otherEntroy = otherNormal.cross_entropy(self.act_probs)
            else:
                act_probs = self.act_probs * tf.one_hot(indices=self.actions, depth=self.act_probs.shape[1])
                act_probs = tf.reduce_sum(act_probs, axis=1)
                # probabilities of actions which agent took with old policy
                act_probs_old = self.o_act_probs * tf.one_hot(indices=self.actions, depth=self.o_act_probs.shape[1])
                act_probs_old = tf.reduce_sum(act_probs_old, axis=1)
                entropy = -tf.reduce_sum(self.act_probs *
                                         tf.log(tf.clip_by_value(self.act_probs, 1e-5, 1.0)), axis=1)
                otherEntroy = -self.s_a_prob * tf.log(self.act_probs + 1e-5)

            with tf.variable_scope('loss/clip'):
                # ratios = tf.divide(act_probs, act_probs_old)
                ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.args['clip_value'],
                                                  clip_value_max=1 + self.args['clip_value'])
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                self.loss_clip = tf.reduce_mean(loss_clip)

                self.entropy = self.args['c2'] * tf.reduce_mean(entropy)  # mean of entropy of pi(obs)

                t = tf.reshape(self.term, shape=[-1, 1])
                entropyTS = tf.reduce_sum(otherEntroy, axis=1,
                                           keepdims=True)
                weight = 0.5 + tf.tanh(3 - self.args['c3'] * self.e) / 2
                entropyTS = entropyTS * weight * self.args['c1']
                self.entropyTS = tf.reduce_mean(entropyTS)

                self.a_loss = -(self.loss_clip + self.entropy) + self.entropyTS
                self.train_a_op = self.optimizer.minimize(self.a_loss, var_list=self.policy_param)

    def choose_action(self, obs):
        obs = np.array(obs)
        obs = obs[np.newaxis, :]
        if self.args['continuous_action']:
            actions, v_preds = self.sess.run([self.sample_action, self.v_preds], {self.obs: obs})
            return np.clip(actions[0], -self.args['action_clip'], self.args['action_clip']), v_preds[0]
        else:
            if self.args['stochastic']:
                actions, v_preds, p = self.sess.run([self.act_stochastic, self.v_preds, self.act_probs], feed_dict={self.obs: obs})
                return np.asscalar(actions), np.asscalar(v_preds)
            else:
                actions, v_preds = self.sess.run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})
                return np.asscalar(actions)

    def get_v(self, s):
        s = np.array(s)
        return self.sess.run(self.v_preds, {self.obs: s[np.newaxis, :]})[0, 0]

    def update(self, actor, s, a, r, options, terms, epi):
        self.sess.run(self.replace_op)

        source_actor_prob = []
        mu = []
        sigma = []
        for i, o in enumerate(options):
            o = int(o)
            if self.args['continuous_action']:
                a_prob = actor[o].choose_acton_prob(s[i], a[i])
                mu.append(a_prob[0])
                sigma.append(a_prob[1])
            else:
                a_prob = actor[o].choose_acton_prob(s[i])
                source_actor_prob.append(a_prob)

        adv = self.sess.run(self.advantage, {self.obs: s, self.rewards: r})
        if self.args['continuous_action']:
            [self.sess.run(self.train_a_op, {self.obs: s, self.actions: a, self.gaes: adv, self.term: terms,
                                             self.mu: mu, self.sigma:sigma, self.e: epi}) for _ in
             range(self.args['epi_train_times'])]
        else:
            [self.sess.run(self.train_a_op, {self.obs: s, self.actions: a, self.gaes: adv, self.term: terms,
                                             self.s_a_prob: source_actor_prob, self.e: epi}) for _ in
             range(self.args['epi_train_times'])]
        [self.sess.run(self.train_c_op, {self.obs: s, self.rewards: r}) for _ in range(self.args['epi_train_times'])]


class CAPS:
    def __init__(self, option_dim, n_features, args, sess, logger):
        self.args = args
        self.option_dim = option_dim
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

        self.sess = sess

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

    def get_term_prob(self, s):
        return self.sess.run(self.term_current, feed_dict={self.s: s})

    def update(self, observation, option, done, observation_):
        if self.update_step % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
        self.update_step += 1

        if not done:
            loss_term, _ = self.sess.run([self.total_error_term, self.update_t], feed_dict={
                self.s: observation[np.newaxis, :],
                self.option_o: [option],
                # opa_t_ph: [option_running],
                # rew_t_ph: [r],
                self.s_: observation_[np.newaxis, :],
                self.done: [1.0 if done is True else 0.0]
            })
            self.logger.write_tb_log('t_loss', loss_term, self.update_step)

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


class PTF_PPO:
    def __init__(self, n_actions, features, args, logger):
        self.args = args
        self.SESS = tf.Session()
        SOURCE_TASK = args['option_model_path']
        N_O = len(SOURCE_TASK)
        g = [tf.Graph() for i in range(N_O)]
        actor_sess = [tf.Session(graph=i) for i in g]
        self.actor = []
        for i in range(len(SOURCE_TASK)):
            with actor_sess[i].as_default():
                with g[i].as_default():
                    dqn = SA(SOURCE_TASK[i], args, actor_sess[i])
                    self.actor.append(dqn)

        self.OT = CAPS(N_O, features, args, self.SESS, logger)
        self.PPO = PPO(args, self.SESS, logger)

        self.SESS.run(tf.global_variables_initializer())
        self.variables = tf.global_variables()
        # SESS.graph.finalize()

    def choose_action(self, s):
        return self.PPO.choose_action(s)

    def save_model(self, path):
        saver = tf.train.Saver(self.variables)
        saver.save(self.SESS, path + ".ckpt")

    def load_model(self, path):
        saver = tf.train.Saver(self.variables)
        saver.restore(self.SESS, path + ".ckpt")
