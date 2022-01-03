import threading
import numpy as np
from util.output_json import OutputJson
import copy


class Worker(object):
    def __init__(self, option, OT, actor, env, i, args, logger, OJ):
        self.env = copy.deepcopy(env)
        self.name = option.thread_AC[i].name
        self.AC = option.thread_AC[i]
        self.COORD = option.COORD
        self.args = args
        self.logger = logger
        self.OJ = OJ
        self.OT = OT
        self.actor = actor
        self.option = option

    def work(self):
        global memory, GLOBAL_EP, GLOBAL_STEP, discount_memory
        buffer_s, buffer_a, buffer_r, buffer_o, buffer_t = [], [], [], [], []
        while not self.COORD.should_stop() and GLOBAL_EP < self.args['numGames']:
            s = self.env.reset()
            s = np.array(s)
            option = self.OT.choose_o(s)
            termination = self.OT.get_t(s, option)
            episode_reward = 0
            episode_discount_reward = 0
            step = 0
            while True:
                if GLOBAL_EP < -1:
                    a = self.actor[option].choose_action_g(s)
                else:
                    a = self.AC.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                #    done = True
                s_ = np.array(s_)
                if self.args['continuous_action']:
                    opa = np.array(
                        [1 if i == option or self.actor[i].is_from_source_actor(s, a) else 0 for i in
                         range(len(self.actor))]
                    )
                else:
                    opa = np.array(
                        [1 if i == option or self.actor[i].choose_action_g(s) == a else 0 for i in range(len(self.actor))]
                    )
                if self.args['reward_normalize']:
                    normalize_reward = r * 1.0 / self.args['done_reward']
                else:
                    normalize_reward = r
                self.OT.store_transition(s, a, normalize_reward, done, s_, opa)

                episode_discount_reward = episode_discount_reward + round(
                    r * np.power(self.args['reward_decay'], step), 8)
                episode_reward = episode_reward + r

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(normalize_reward)
                buffer_o.append(option)
                buffer_t.append([termination])

                if GLOBAL_STEP > self.args['learning_step']:
                    self.OT.update(s, option, done, s_)

                if GLOBAL_EP >= -1 and ((step != 0 and step % self.args['batch_size'] == 0) or step == self.args['epi_step'] or done):
                    if len(buffer_s) != 0:
                        self.AC.update(self.actor, buffer_s, buffer_a, buffer_r, buffer_o, buffer_t, done, s_,
                                      GLOBAL_STEP,
                                      GLOBAL_EP)
                        buffer_s, buffer_a, buffer_r, buffer_o, buffer_t = [], [], [], [], []

                termination = self.OT.get_t(s_, option)
                if termination > np.random.uniform():
                    # print(term)
                    option = self.OT.choose_o(s_)
                if done or step == self.args['epi_step']:   # update global and assign to local net
                    memory[GLOBAL_EP % self.args['reward_memory']] = episode_reward
                    discount_memory[GLOBAL_EP % self.args['reward_memory']] = episode_discount_reward
                    mean_memory = np.mean(memory)
                    discount_mean_memory = np.mean(discount_memory)
                    self.OJ.update([done, step, self.OT.epsilon, episode_discount_reward, discount_mean_memory, episode_reward,
                                    mean_memory, GLOBAL_EP])
                    self.OJ.print_first()

                    self.logger.write_tb_log('discount_reward', episode_discount_reward, GLOBAL_EP)
                    self.logger.write_tb_log('discount_reward_mean', discount_mean_memory, GLOBAL_EP)
                    self.logger.write_tb_log('undiscounted reward', episode_reward, GLOBAL_EP)
                    self.logger.write_tb_log('reward_mean', mean_memory, GLOBAL_EP)

                    if GLOBAL_STEP > self.args['learning_step']:
                        self.OT.update_e()

                    GLOBAL_EP += 1
                    break

                s = s_
                GLOBAL_STEP += 1
                step += 1

            if self.args['save_model'] and (GLOBAL_EP % self.args['save_per_episodes'] == 1):
                self.option.save_model(
                    self.args['results_path'] + self.args['SAVE_PATH'] + "/model" + "_" + str(GLOBAL_EP-1))
                self.OJ.save(self.args['results_path'] + self.args['reward_output'], self.args['output_filename'])


def run(args, env, option, logger):
    global memory, GLOBAL_EP, GLOBAL_STEP, discount_memory
    memory = np.zeros(args['reward_memory'])
    discount_memory = np.zeros(args['reward_memory'])
    GLOBAL_EP = 0
    GLOBAL_STEP = 0
    field = ['win', 'step', 'epsilon', 'discounted_reward', 'discount_reward_mean', 'undiscounted_reward', 'reward_mean', 'episode']
    OJ = OutputJson(field)

    workers = []
    for i in range(option.N_WORKERS):
        workers.append(Worker(option, option.OT, option.actor, env, i, args, logger, OJ))
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)  # 创建一个线程，并分配其工作
        t.start()  # 开启线程
        worker_threads.append(t)
    option.COORD.join(worker_threads)  # 把开启的线程加入主线程，等待threads结束

    OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])
    if args['save_model']:
        option.save_model(args['results_path'] + args['SAVE_PATH']+"/model")
