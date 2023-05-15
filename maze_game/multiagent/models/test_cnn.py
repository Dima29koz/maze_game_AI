import torch
from gymnasium.spaces import Dict, Discrete
from ray.rllib.env import PettingZooEnv

from maze_game.multiagent.maze_multi_agent_env import MAMazeGameEnv, create_env
from maze_game.multiagent.models.a2c_model import A2CNetwork
from maze_game.multiagent.models.action_masking import TorchActionMaskModel
from maze_game.multiagent.models.complex_input import ComplexInputNetwork
from maze_game.multiagent.models.maze_cnn import MazeCNN


def random_policy(env, agent, observation):
    return env.action_space(agent).sample(observation["action_mask"])


def run(num_resets=1):
    num_players = 4
    env = create_env(render_mode="human", num_players=num_players)
    # env = MAMazeGameEnv(render_mode="human", num_players=num_players)
    env.metadata['render_fps'] = 10
    env.reset()

    test_env = PettingZooEnv(create_env(num_players=num_players))
    obs_space: Dict = test_env.observation_space
    act_space: Discrete = test_env.action_space

    model_act = TorchActionMaskModel(
        obs_space, act_space, num_outputs=act_space.n, model_config={}, name='action_masking')
    # model_a2c = A2CNetwork(
    #     obs_space['observation'], act_space, num_outputs=act_space.n, model_config={}, name='a2c')
    # model_comp = ComplexInputNetwork(
    #     obs_space['observation'], act_space, model_config={}, name='complex_input')
    _type = 'treasures'
    # model_cnn = MazeCNN(
    #     obs_space['observation'][_type], act_space, model_config={'obs_type': _type}, name=f'cnn_{_type}')

    env.render()
    for _ in range(num_resets):
        for agent in env.agent_iter():
            print(agent, 'turn:')
            observation, reward, termination, truncation, info = env.last()
            print('prev info:', info)
            if termination or truncation:
                action = None
            else:
                # TorchActionMaskModel
                res, _ = model_act(
                    {
                        'obs': {
                            'observation': {
                                key: torch.Tensor([sub_obs] * 4) for key, sub_obs in
                                observation.get('observation').items()
                            },
                            'action_mask': torch.Tensor([observation.get('action_mask')] * 4)

                        }
                    },
                    [], None
                )
                # A2CNetwork / ComplexInputNetwork
                # res, _ = model_a2c(
                #     {
                #         'obs': {
                #             key: torch.Tensor([sub_obs] * 4) for key, sub_obs in observation.get('observation').items()
                #         }
                #     },
                #     [], None
                # )
                # MazeCNN(_type)
                # res, _ = model_cnn(
                #     {
                #         'obs': torch.Tensor([observation.get('observation').get(_type)] * 4)
                #     },
                #     [], None
                # )
                # res_vf = model_act.value_function()
                action = random_policy(env, agent, observation)

            env.step(action)

            env.render()
        env.reset()
    env.close()


if __name__ == "__main__":
    run(100)
