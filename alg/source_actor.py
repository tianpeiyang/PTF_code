import numpy as np
import tensorflow as tf
import os
import json


class SourceActor:
    def __init__(self, policy_path, args, sess):
        self.args = args
        self.sess = sess
        self.actor = self.build_source_actor(policy_path)
        self.actor.load_net(policy_path)

    def build_source_actor(self, policy_path):
        par_path = os.path.dirname(policy_path)
        file_name = ''
        for dirPath, dirNames, fileNames in os.walk(par_path):
            #print(fileNames)
            for fileName in fileNames:
                if fileName == 'args.json':
                    file_name = fileName
                    break
            if file_name != '':
                break
        file_path = par_path + "/" + file_name
        with open(file_path, 'r') as f:
            source_args = json.load(f)
        source_policy = self.args['source_policy']
        if source_policy == 'dqn':
            return DeepQNetwork(self.args['action_dim'], self.args['features'], source_args['n_layer_1'], self.sess)
        elif source_policy == 'a3c':
            return ACNet(self.args['action_dim'], self.args['features'], source_args, self.sess)
        elif source_policy == 'ppo':
            return PPO(self.args['action_dim'], self.args['features'], source_args, self.sess)
        else:
            raise Exception('no such source_policy named: ' + str(source_policy))

    def choose_action_g(self, ob):
        return self.actor.choose_action_g(ob)

    def choose_acton_prob(self, ob, action=None):
        return self.actor.choose_acton_prob(ob, action)

    def is_from_source_actor(self, ob, action=None):
        return self.actor.is_from_source_actor(ob, action)



# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            n_layer,
            sess
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_layer = n_layer
        self.sess = sess

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        with tf.variable_scope('eval_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.n_layer, \
                tf.random_normal_initializer(0., 0.5), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1', reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2', reuse=tf.AUTO_REUSE):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

    def choose_action_g(self, observation):
        observation = np.array(observation)
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        #print(self.epsilon)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})[0]
        action = np.argmax(actions_value)
        return action

    def choose_acton_prob(self, observation, action=None):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        # print(self.epsilon)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action_arg = np.argmax(actions_value)
        action_prob = np.zeros(self.n_actions)
        action_prob[action_arg] = 1
        return action_prob

    def is_from_source_actor(self, observation, action=None):
        return True

    def load_net(self, path):
        saver = tf.train.Saver()
        #print(path)
        saver.restore(self.sess, path + ".ckpt")


class ACNet:
    def __init__(self, action_dim, features, args, sess):
        self.args = args
        self.N_S = features
        self.N_A = action_dim
        self.SESS = sess

        with tf.variable_scope(self.args['GLOBAL_NET_SCOPE']):
            self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
            self.input_mu = tf.placeholder(tf.float32, [None, self.N_A], 'input_mu')
            self.input_sigma = tf.placeholder(tf.float32, [None, self.N_A], 'input_sigma')
            # self.a_params, self.c_params = self._build_net(scope)[-2:]
            ap_g, self.a_params = self._build_net(self.args['GLOBAL_NET_SCOPE'])
            if self.args['continuous_action']:
                mu, sigma = ap_g[0] * self.args['action_clip'], ap_g[1] + 1e-10
                normal_dist = tf.distributions.Normal(mu, sigma)
                self.A_G = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0),
                                            -self.args['action_clip'], self.args['action_clip'])
                self.a_prob = [mu, sigma]
                otherNormal = tf.distributions.Normal(self.input_mu, self.input_sigma)
                Entropy = normal_dist.cross_entropy(otherNormal)
                self.Entropy = tf.reduce_mean(Entropy)
            else:
                self.A_G = ap_g
                self.a_prob = ap_g

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .01)
        # w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, self.args['n_layer_a_1'], tf.nn.relu6, kernel_initializer=w_init,
                                  name='la')
            # l_a_2 = tf.layers.dense(l_a, self.args['n_layer_a_2'], tf.nn.relu6, kernel_initializer=w_init, name='la2')
            if self.args['continuous_action']:
                mu = tf.layers.dense(l_a, self.N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a, self.N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                a_prob = [mu, sigma]
            else:
                a_prob = tf.layers.dense(l_a, self.N_A, tf.nn.softmax, kernel_initializer=w_init,
                                         name='ap')  # 得到每个动作的选择概率
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        return a_prob, a_params

    def choose_action_g(self, observation):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        if self.args['continuous_action']:
            action = self.SESS.run(self.A_G, feed_dict={self.s: observation})[0]
            return np.clip(action, -self.args['action_clip'], self.args['action_clip'])
        else:
            prob_weights = self.SESS.run(self.A_G, feed_dict={self.s: observation})
            action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def choose_acton_prob(self, observation, action):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        if self.args['continuous_action']:
            actions_value = self.SESS.run(self.a_prob, feed_dict={self.s: observation})
            actions_value = [actions_value[0][0], actions_value[1][0]]
        else:
            actions_value = self.SESS.run(self.a_prob, feed_dict={self.s: observation})[0]
        return actions_value

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

    def load_net(self, path):
        variables = tf.global_variables()
        variables_to_restore = [v for v in variables if v.name.split('/')[0] == self.args['GLOBAL_NET_SCOPE']]
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(self.SESS, path + ".ckpt")


class PPO:
    def __init__(self, action_dim, features, args, sess):
        self.args = args
        self.n_features = features
        self.n_actions = action_dim
        self.SESS = sess

        self.obs = tf.placeholder(tf.float32, [None, self.n_features], 's')
        self.act_probs, self.policy_param = self._build_net(self.args['policy'])

        if self.args['continuous_action']:
            normal_dist = tf.distributions.Normal(self.act_probs[0], self.act_probs[1])
            self.sample_action = tf.squeeze(normal_dist.sample(1), axis=0)
            self.a_prob = self.act_probs
        else:
            self.a_prob = self.act_probs
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])
            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            layer_1 = tf.layers.dense(inputs=self.obs, units=self.args['n_layer_a_1'], activation=tf.nn.relu)
            if self.args['continuous_action']:
                mu = self.args['action_clip'] * tf.layers.dense(inputs=layer_1, units=self.n_actions, activation=tf.nn.tanh)
                sigma = tf.layers.dense(inputs=layer_1, units=self.n_actions, activation=tf.nn.softplus)
                sigma = sigma + 1e-5
                act_probs = [mu, sigma]
            else:
                act_probs = tf.layers.dense(inputs=layer_1, units=self.n_actions, activation=tf.nn.softmax)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return act_probs, params

    def choose_action_g(self, observation):
        obs = np.array(observation)
        obs = obs[np.newaxis, :]
        if self.args['continuous_action']:
            actions = self.SESS.run(self.sample_action, {self.obs: obs})[0]
            return np.clip(actions, -self.args['action_clip'], self.args['action_clip'])
        else:
            if self.args['stochastic']:
                actions = self.SESS.run(self.act_stochastic, feed_dict={self.obs: obs})
                return np.asscalar(actions)
            else:
                actions = self.SESS.run(self.act_deterministic, feed_dict={self.obs: obs})
                return np.asscalar(actions)

    def choose_acton_prob(self, observation, action):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        if self.args['continuous_action']:
            actions_value = self.SESS.run(self.a_prob, feed_dict={self.obs: observation})
            actions_value = [actions_value[0][0], actions_value[1][0]]
        else:
            actions_value = self.SESS.run(self.a_prob, feed_dict={self.obs: observation})[0]
        return actions_value

    def is_from_source_actor(self, observation, action):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        if self.args['continuous_action']:
            actions_value = self.SESS.run(self.a_prob, feed_dict={self.obs: observation})
            sigma, mu = actions_value[0][0], actions_value[1][0]
            is_in = True
            for i in range(len(action)):
                if action[i] < sigma[i] - mu[i] or action[i] > sigma[i] + mu[i]:
                    is_in = False
                    break
            return is_in

        else:
            if self.args['stochastic']:
                actions = self.SESS.run(self.act_stochastic, feed_dict={self.obs: observation})
            else:
                actions = self.SESS.run(self.act_deterministic, feed_dict={self.obs: observation})
            a = np.asscalar(actions)
            if a == action:
                return True
            else:
                return False

    def load_net(self, path):
        variables = tf.global_variables()
        variables_to_restore = [v for v in variables if v.name.split('/')[0] == self.args['policy']]
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(self.SESS, path + ".ckpt")

if __name__ == '__main__':
    args = {}
    args['source_policy'] = 'a3c'
    args['action_dim'] = 2
    args['features'] = 4
    args['configuration'] = '../game/pinball_hard_single.cfg'
    args['width'] = 500
    args['height'] = 500
    args['start_position'] = [[0.6, 0.4]]
    args['target_position'] = [0.9, 0.2]
    args['continuous_action'] = True
    args['action_clip'] = 1
    args['reward_normalize'] = True
    args['done_reward'] = 10000
    args['random_start'] = False
    args['run_test'] = True
    args['sequential_state'] = False

    from game.pinball import PinballModel
    env = PinballModel(args)
    s = env.reset()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    path = '../source_policies/a3c/0.90.2/model'
    actor = SourceActor(path, args, sess)
    import time
    while (True):
        time.sleep(0.01)
        action = actor.choose_action_g(s)
        prob = actor.is_from_source_actor(s, action)
        #print(action, np.argmax(prob), prob)
        s_, r, done = env.step(action)
        env.render()
        s = s_
        if done:
            s = env.reset()




