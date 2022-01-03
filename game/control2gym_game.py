import dm_control2gym


def game(args):
    env = dm_control2gym.make(domain_name="reacher", task_name=args['task'])  # easy, hard
    env.seed(args['seed'])
    return env
