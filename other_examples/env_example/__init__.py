from gymnasium.envs.registration import register

register(
    id="env_example/GridWorld-v0",
    entry_point="env_example.envs:GridWorldEnv",
)
