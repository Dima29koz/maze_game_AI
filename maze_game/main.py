import torch

from maze_game.test.policy import Policy
from maze_game.test.train import instantiate, train
import minigrid


max_env_steps = 50


# hyperparameters
class policy_params:
    policy_class = Policy  # Policy class to use (replaced later)
    hidden_dim = 32  # dimension of the hidden state in actor network
    learning_rate = 1e-3  # learning rate of policy update
    batch_size = 1024  # batch size for policy update
    policy_epochs = 4  # number of epochs per policy update
    entropy_coef = 0.001  # hyperparameter to vary the contribution of entropy loss


pol_par = dict(
    policy_class=Policy,  # Policy class to use (replaced later)
    hidden_dim=32,  # dimension of the hidden state in actor network
    learning_rate=1e-3,  # learning rate of policy update
    batch_size=1024,  # batch size for policy update
    policy_epochs=4,  # number of epochs per policy update
    entropy_coef=0.001,
)


class params:
    policy_params = pol_par
    rollout_size = 2050  # number of collected rollout steps per policy update
    num_updates = 50  # number of training policy iterations
    discount = 0.99  # discount factor
    plotting_iters = 10  # interval for logging graphs and policy rollouts
    env_name = 'MiniGrid-Empty-5x5-v0'  # we are using a tiny environment here for testing


env, rollouts, policy = instantiate(params)
rewards, success_rate = train(env, rollouts, policy, params)
print("Training completed!")


# print(rewards)
# print(success_rate)


eps = 0
wins = 0
sum_reward = 0
while True:
    obs, _ = env.reset()
    eps += 1
    while True:
        action, _ = policy.act(torch.tensor(obs, dtype=torch.float32))
        obs, reward, terminated, truncated, _ = env.step(int(action))
        env.render()
        done = terminated | truncated
        if terminated:
            wins += 1
            sum_reward += reward
        if done or env.gui is None:
            break
    print(f"eps={eps}, mean-rew={sum_reward/eps}, wins={wins}")
    if env.gui is None:
        break
