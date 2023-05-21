import argparse
import os
import time

from maze_game.multiagent.agent.ppo_agent import PPOAgent
from maze_game.multiagent.maze_multi_agent_env import MAMazeGameEnv, create_env
from maze_game.multiagent.actions import action_to_action_space

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="checkpoint path")


class RLGame:
    def __init__(self, num_players):
        self.num_players = num_players

        args = parser.parse_args()
        checkpoint_path = os.path.expanduser(args.path)

        self.ppo_agent = PPOAgent(self.num_players, checkpoint_path)
        self.env = None

    def manual_policy(self, agent=None, observation=None):
        if self.env.unwrapped.gui is None:
            self.env.render()
        act = None
        act_pl_abilities = self.env.unwrapped.game.get_allowed_abilities(self.env.unwrapped.game.get_current_player())
        while not act:
            act, state = self.env.unwrapped.gui.get_action(act_pl_abilities)
        return action_to_action_space(act)

    def random_policy(self, agent, observation):
        return self.env.action_space(agent).sample(observation["action_mask"])

    def run_performance_test(self, num_resets=100):
        self.env = create_env(render_mode=None, num_players=self.num_players)
        self.env.metadata['render_fps'] = 10

        all_times = []
        all_steps = []
        all_steps_tr = []
        sh = 0
        sh_s = 0
        for i in range(num_resets):
            self.env.reset()
            steps = 0
            times = []
            tr_step = 0
            print('reset number', i)
            for agent in self.env.agent_iter():
                time_start = time.time()
                observation, reward, termination, truncation, info = self.env.last()
                # print(info, f'reward={reward}')
                if termination or truncation:
                    action = None
                else:
                    # this is where you would insert your policy
                    # action = self.random_policy(agent, observation)
                    # action = self.manual_policy(agent, observation)
                    action = self.ppo_agent.compute_action(observation)

                self.env.step(action)
                time_end = time.time() - time_start
                times.append(time_end)
                if tr_step == 0 and info.get('turn_info').get('action') == 'swap_treasure' and info.get(
                        'resp') == 'подобрал клад':
                    tr_step = steps

                if info.get('turn_info').get('action') == 'shoot_bow':
                    sh += 1
                    if info.get('resp').find('ранены') != -1:
                        sh_s += 1
                steps += 1
            all_steps.append(steps)
            all_steps_tr.append(tr_step)
            all_times.append(times)
        self.env.close()
        return {
            'steps': all_steps,
            'tr_steps': all_steps_tr,
            'times': all_times,
            'shooting res': (sh, sh_s)
        }

    def run(self, num_resets=1):
        self.env = create_env(render_mode='human', num_players=self.num_players)
        self.env.metadata['render_fps'] = 10

        for i in range(num_resets):
            self.env.reset()

            print('reset number', i)
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                print(info, f'reward={reward}')
                if termination or truncation:
                    action = None
                else:
                    # this is where you would insert your policy
                    # action = self.random_policy(agent, observation)
                    # action = self.manual_policy(agent, observation)
                    action = self.ppo_agent.compute_action(observation)

                self.env.step(action)
                self.env.render()
        self.env.close()


if __name__ == "__main__":
    game = RLGame(num_players=4)
    game.run(100)
    # res = game.run_performance_test(100)
    # print(res)
