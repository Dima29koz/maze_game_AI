import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from maze_game import MazeGameEnv
from maze_game.sb3_tests import config as conf


def visualize():
    model_name = "PPO_field_dif"
    env = gym.make('env_maze/MazeGame-v0', render_mode='human')
    model = MaskablePPO.load(f"{conf.root_path}/{conf.model_name}_{conf.run_id}/model", env=env)
    # model = MaskablePPO.load(f"{conf.root_path}/{conf.model_name}_{conf.run_id}/save_4400000_steps", env=env)
    # model = MaskablePPO.load(f"{conf.root_path}/{model_name}/model", env=env)

    sum_reward = 0
    eps = 0
    wins = 0
    for episode in range(10_000):
        obs, info = env.reset()
        eps += 1
        while True:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=False)
            # action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            print(info, reward)
            env.render()
            done = terminated | truncated
            if terminated:
                wins += 1
                sum_reward += reward

            if done or env.gui is None:
                break
        print(f"eps={eps}, mean-rew={sum_reward / eps}, wins={wins}")
        if env.gui is None:
            break


if __name__ == "__main__":
    visualize()
