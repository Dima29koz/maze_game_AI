import supersuit as ss

from maze_game.multiagent.maze_multi_agent_env import MAMazeGameEnv, create_env
from maze_game.multiagent.actions import action_to_action_space


def manual_policy(env, agent=None, observation=None):
    act = None
    act_pl_abilities = env.unwrapped.game.get_allowed_abilities(env.unwrapped.game.get_current_player())
    while not act:
        act, state = env.unwrapped.gui.get_action(act_pl_abilities)
    return action_to_action_space(act)


def random_policy(env, agent, observation):
    return env.action_space(agent).sample(observation["action_mask"])


def run(num_resets=1):
    # env = create_env(render_mode="human", num_players=2)
    env = MAMazeGameEnv(render_mode="human", num_players=4)
    env.metadata['render_fps'] = 10
    env.reset()
    env.render()
    for _ in range(num_resets):
        for agent in env.agent_iter():
            print(agent, 'turn:')
            observation, reward, termination, truncation, info = env.last()
            print('prev info:', info)
            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = random_policy(env, agent, observation)
                # action = manual_policy(env, agent, observation)

            env.step(action)

            env.render()
        env.reset()
    env.close()


if __name__ == "__main__":
    run(100)
