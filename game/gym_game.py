import gym


def game(args):
    env = gym.make(args['game'])
    env.seed(args['seed'])
    return env
