import tensorflow as tf
from util.output_json import OutputJson
import numpy as np


def run(args, env, PPO, logger):
    total_step = 0
    memory = np.zeros(args['reward_memory'])
    discount_memory = np.zeros(args['reward_memory'])
    numGames = args['numGames']
    totalreward = np.zeros(numGames)

    field = ['win', 'step', 'discounted_reward', 'discount_reward_mean', 'undiscounted_reward', 'reward_mean', 'episode']
    OJ = OutputJson(field)

    #if args['load_model']:
        #PPO.load_model(args['load_model_path'])

    for episode in range(numGames):
        # initial observation
        observation = env.reset()
        episode_reward = 0
        episode_discount_reward = 0
        step = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        while True:
            # RL choose action based on observation
            act, v_pred = PPO.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, d, _ = env.step(act)
            done = d
            if args['reward_normalize']:
                normalize_reward = reward / args['done_reward']
            else:
                normalize_reward = reward
            # swap observation

            buffer_s.append(observation)
            buffer_a.append(act)
            buffer_r.append(normalize_reward)

            if (step != 0 and (step % args['batch_size'] == 0)) or done:
                if done:
                    v_s_ = 0
                else:
                    v_s_ = PPO.get_v(observation_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + args['reward_decay'] * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br = np.vstack(buffer_s), buffer_a, np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                PPO.update(bs, ba, br)

            observation = observation_
            episode_reward = episode_reward + reward
            episode_discount_reward = episode_discount_reward + round(reward * np.power(args['reward_decay'], step), 8)
            # break while loop when end of this episode
            if done or step > args['epi_step']:
                discount_memory[episode % args['reward_memory']] = episode_discount_reward
                memory[episode % args['reward_memory']] = episode_reward
                totalreward[episode] = episode_reward
                mean_memory = np.mean(memory)
                discount_mean_memory = np.mean(discount_memory)
                # print(RL.memory_counter)

                OJ.update([done, step, episode_discount_reward, discount_mean_memory, episode_reward, mean_memory, episode])
                OJ.print_first()

                logger.write_tb_log('discount_reward', episode_discount_reward, episode)
                logger.write_tb_log('discount_reward_mean', discount_mean_memory, episode)
                logger.write_tb_log('undiscounted reward', episode_reward, episode)
                logger.write_tb_log('reward_mean', mean_memory, episode)

                break

            step += 1
            total_step += 1

        if args['save_model'] and episode % args['save_per_episodes'] == 0:
            PPO.save_model(args['results_path'] + args['SAVE_PATH'] + "/model" + "_" + str(episode))
            OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

    OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

    if args['save_model']:
        PPO.save_model(args['results_path'] + args['SAVE_PATH']+"/model")
