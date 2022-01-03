import numpy as np
from util.output_json import OutputJson


def run(args, env, RL, logger):
    total_step = 0
    memory = np.zeros(args['reward_memory'])
    discount_memory = np.zeros(args['reward_memory'])
    numGames = args['numGames']
    totalreward = np.zeros(numGames)

    field = ['win', 'step', 'discounted_reward', 'discount_reward_mean', 'undiscounted_reward', 'reward_mean', 'e_greedy', 'episode']
    OJ = OutputJson(field)


    if args['load_model']:
        RL.load_model(args['load_model_path'])

    for episode in range(numGames):
        # initial observation
        #start = time.clock()
        observation = env.reset()
        #end = time.clock()
        #print("reset", end-start)
        # print(observation)
        step = 0
        episode_reward = 0
        episode_discount_reward = 0
        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            if args['reward_normalize']:
                normalize_reward = reward * 1.0 / args['done_reward']
            else:
                normalize_reward = reward
            RL.store_transition(observation, action, normalize_reward, observation_)
            if total_step > args['learning_step']:
                lr = RL.learn()
            # swap observation
            observation = observation_
            episode_discount_reward = episode_discount_reward + round(reward * np.power(args['reward_decay'], step),
                                                                      8)
            episode_reward = episode_reward + reward
            # break while loop when end of this episode
            if done or step > args['epi_step']:
                memory[episode % args['reward_memory']] = episode_reward
                discount_memory[episode % args['reward_memory']] = episode_discount_reward
                totalreward[episode] = episode_reward
                mean_memory = np.mean(memory)
                discount_mean_memory = np.mean(discount_memory)
                # print(RL.memory_counter)

                OJ.update([done, step, episode_discount_reward, discount_mean_memory, episode_reward, mean_memory, RL.epsilon, episode])
                OJ.print_first()

                logger.write_tb_log('discount_reward', episode_discount_reward, episode)
                logger.write_tb_log ('discount_reward_mean', discount_mean_memory, episode)
                logger.write_tb_log('undiscounted reward', episode_reward, episode)
                logger.write_tb_log('reward_mean', mean_memory, episode)

                RL.update_epsilon(episode)
                break

            step += 1
            total_step += 1

        if episode % args['save_per_episodes'] == 0:
            RL.save_model(args['results_path'] + args['SAVE_PATH'] + "/model" + "_" + str(episode))

    OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

    if args['save_model']:
        RL.save_model(args['results_path'] + args['SAVE_PATH']+"/model")

    import matplotlib.pyplot as plt
    ax1 = plt.subplot(221)
    plt.sca(ax1)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(np.arange(len(totalreward)), totalreward)
    # end of game
    print('game over')
    # env.destroy()
    ax2 = plt.subplot(222)
    plt.sca(ax2)
    RL.plot_cost()
