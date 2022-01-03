import tensorflow as tf
from alg.PRM_actor import PRM_actor as PRM_AC
from util.output_json import OutputJson
from alg.source_actor import SourceActor as SA
import numpy as np


def run(args, env, CA, logger):
    SOURCE_TASK = args['option_model_path']

    if args['continuous_action']:
        action_dim = len(env.pri_action)
    else:
        action_dim = args['action_dim']
    N_O = len(args['option_model_path']) + action_dim

    g = [tf.Graph() for i in range(N_O)]
    actor_sess = [tf.Session(graph=i) for i in g]
    actor = []
    for i in range(len(SOURCE_TASK)):
        with actor_sess[i].as_default():
            with g[i].as_default():
                dqn = SA(SOURCE_TASK[i], args, actor_sess[i])
                actor.append(dqn)
    if args['continuous_action']:
        for i in range(len(env.pri_action)):
            with actor_sess[i].as_default():
                with g[i].as_default():
                    AC = PRM_AC(env.pri_action[i], action_dim)
                    actor.append(AC)
    else:
        for i in range(action_dim):
            with actor_sess[i].as_default():
                with g[i].as_default():
                    AC = PRM_AC(i, action_dim)
                    actor.append(AC)

    total_step = 0
    memory = np.zeros(args['reward_memory'])
    discount_memory = np.zeros (args['reward_memory'])
    numGames = args['numGames']
    totalreward = np.zeros(numGames)

    field = ['win', 'step', 'discounted_reward', 'discount_reward_mean', 'undiscounted_reward', 'reward_mean', 'e_greedy', 'episode']
    OJ = OutputJson(field)

    for episode in range(numGames):
        # initial observation
        observation = env.reset()
        observation = np.array(observation)
        option = CA.choose_o(observation)

        step = 0
        episode_reward = 0
        episode_discount_reward = 0
        while True:
            action = actor[option].choose_action_g(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            observation_ = np.array(observation_)
            if args['continuous_action']:
                opa = np.array(
                    [1 if i == option or actor[i].is_from_source_actor(observation, action) else 0 for i in range(N_O)]
                )
            else:
                opa = np.array(
                    [1 if i == option or actor[i].choose_action_g(observation) == action else 0 for i in range(N_O)]
                )
            if args['reward_normalize']:
                normalize_reward = reward * 1.0 / args['done_reward']
            else:
                normalize_reward = reward
            CA.store_transition(observation, action, normalize_reward, done, observation_, opa)
            if total_step > args['learning_step']:
                CA.update(observation, option, normalize_reward, done, observation_)
            # swap observation
            observation = observation_

            episode_discount_reward = episode_discount_reward + round(reward * np.power(args['reward_decay'], step),
                                                                      8)
            episode_reward = episode_reward + reward

            term = CA.get_t(observation_, option)
            # if episode % 10 == 0:
                # print(term)
            if term > np.random.uniform():
                #print(term)
                option = CA.choose_o(observation_)

            # break while loop when end of this episode
            if done or step > args['epi_step']:
                discount_memory[episode % args['reward_memory']] = episode_discount_reward
                memory[episode % args['reward_memory']] = episode_reward
                totalreward[episode] = episode_reward
                mean_memory = np.mean(memory)
                discount_mean_memory = np.mean(discount_memory)
                # print(RL.memory_counter)

                OJ.update([done, step, episode_discount_reward, discount_mean_memory, episode_reward, mean_memory, CA.epsilon, episode])
                OJ.print_first()

                logger.write_tb_log('discount_reward', episode_discount_reward, episode)
                logger.write_tb_log('discount_reward_mean', discount_mean_memory, episode)
                logger.write_tb_log('undiscounted reward', episode_reward, episode)
                logger.write_tb_log('reward_mean', mean_memory, episode)
                CA.update_e()

                break

            step += 1
            total_step += 1

        if args['save_model'] and episode % args['save_per_episodes'] == 0:
            CA.save_model(args['results_path'] + args['SAVE_PATH'] + "/model" + "_" + str(episode))
            OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

    OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

    if args['save_model']:
        CA.save_model(args['results_path'] + args['SAVE_PATH']+"/model")
