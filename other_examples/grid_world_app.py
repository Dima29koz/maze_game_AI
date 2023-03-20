import gymnasium

if __name__ == "__main__":
    env = gymnasium.make('env_example/GridWorld-v0', render_mode='human')

    done = False
    while not done:
        state = env.reset()
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
