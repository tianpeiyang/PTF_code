import tensorflow as tf
from util.output_json import OutputJson
import numpy as np


def run(args, env, alg, logger):
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
        option = alg.OT.choose_o(observation)
        termination = alg.OT.get_t(observation, option)
        step = 0
        episode_reward = 0
        episode_discount_reward = 0
        buffer_s, buffer_a, buffer_r, buffer_o, buffer_t = [], [], [], [], []
        while True:
            action, v_pred = alg.PPO.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            if args['continuous_action']:
                opa = np.array(
                    [1 if i == option or alg.actor[i].is_from_source_actor(observation, action) else 0 for i in
                     range(len(alg.actor))]
                )
            else:
                opa = np.array(
                    [1 if i == option or alg.actor[i].choose_action_g(observation) == action else 0 for i in range(len(alg.actor))]
                )
            if args['reward_normalize']:
                normalize_reward = reward * 1.0 / args['done_reward']
            else:
                normalize_reward = reward

            buffer_s.append(observation)
            buffer_a.append(action)
            buffer_r.append(normalize_reward)
            buffer_o.append(option)
            buffer_t.append(termination)

            if (step != 0 and step % args['batch_size'] == 0) or step > args['epi_step'] or done:
                if done:
                    v_s_ = 0
                else:
                    v_s_ = alg.PPO.get_v(observation_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + args['reward_decay'] * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br, bo, bt = np.vstack(buffer_s), buffer_a, np.array(discounted_r)[:, np.newaxis], buffer_o, buffer_t
                buffer_s, buffer_a, buffer_r, buffer_o, buffer_t = [], [], [], [], []
                alg.PPO.update(alg.actor, bs, ba, br, bo, bt, episode)

            observation_ = np.array(observation_)
            alg.OT.store_transition(observation, action, normalize_reward, done, observation_, opa)

            if total_step > args['learning_step']:
                alg.OT.update(observation, option, done, observation_)

            termination = alg.OT.get_t(observation_, option)
            if np.random.uniform() < termination:
                option = alg.OT.choose_o(observation_)

            # swap observation
            observation = observation_

            episode_discount_reward = episode_discount_reward + round(reward * np.power(args['reward_decay'], step),
                                                                      8)
            episode_reward = episode_reward + reward

            # break while loop when end of this episode
            if done or step > args['epi_step']:
                discount_memory[episode % args['reward_memory']] = episode_discount_reward
                memory[episode % args['reward_memory']] = episode_reward
                totalreward[episode] = episode_reward
                mean_memory = np.mean(memory)
                discount_mean_memory = np.mean(discount_memory)
                # print(RL.memory_counter)

                OJ.update([done, step, episode_discount_reward, discount_mean_memory, episode_reward, mean_memory, alg.OT.epsilon, episode])
                OJ.print_first()

                logger.write_tb_log('discount_reward', episode_discount_reward, episode)
                logger.write_tb_log('discount_reward_mean', discount_mean_memory, episode)
                logger.write_tb_log('undiscounted reward', episode_reward, episode)
                logger.write_tb_log('reward_mean', mean_memory, episode)

                if total_step > args['learning_step']:
                    alg.OT.update_e()

                break

            step += 1
            total_step += 1

        if args['save_model'] and episode % args['save_per_episodes'] == 0:
            alg.save_model(args['results_path'] + args['SAVE_PATH'] + "/model" + "_" + str(episode))
            OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

    OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

    if args['save_model']:
        alg.save_model(args['results_path'] + args['SAVE_PATH']+"/model")
