import argparse
import os
import time

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from maze_game.multiagent.config import policy_mapping_fn, policies, obs_space, act_space, gen_policy, num_players
from maze_game.multiagent.maze_multi_agent_env import MAMazeGameEnv, create_env
from maze_game.multiagent.actions import action_to_action_space
from maze_game.multiagent.models.action_masking import TorchActionMaskModel

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="checkpoint path")


def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return 'main'


class RLGame:
    def __init__(self):
        env_name = "maze_game_v2"
        self.num_players = num_players

        register_env(env_name, lambda config: PettingZooEnv(create_env(num_players=self.num_players)))
        ModelCatalog.register_custom_model("pa_model", TorchActionMaskModel)

        args = parser.parse_args()
        checkpoint_path = os.path.expanduser(args.path)

        ray.init(local_mode=True, num_cpus=0)
        config = (
            PPOConfig()
            .environment(env=env_name)
            .rollouts(
                num_rollout_workers=0,
            )
            .training(
                model={'custom_model': 'pa_model'}
            )
            .multi_agent(
                policies={f"main": gen_policy() for i in range(num_players)},
                # policy_mapping_fn=policy_mapping_fn,
                # policies={
                #     'main': PolicySpec(
                #         policy_class=None,
                #         observation_space=obs_space,
                #         action_space=act_space,
                #     ),
                # },
                policy_mapping_fn=new_policy_mapping_fn,
                policies_to_train=[],
            )
            .debugging(log_level="ERROR")
            .framework(framework="torch")
            .resources(num_gpus=1)
        )

        self.ppo_agent = PPO(config)
        self.ppo_agent.restore(checkpoint_path)
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

    def ppo_policy(self, agent, observation):
        # policy_id = policy_mapping_fn(agent)
        policy_id = new_policy_mapping_fn(agent, None, None)
        return self.ppo_agent.compute_single_action(observation, policy_id=policy_id)

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
                    action = self.ppo_policy(agent, observation)

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
                    action = self.ppo_policy(agent, observation)

                self.env.step(action)
                self.env.render()
        self.env.close()


if __name__ == "__main__":
    game = RLGame()
    game.run(100)
    # res = game.run_performance_test(100)
    # print(res)
