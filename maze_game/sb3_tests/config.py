from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_latest_run_id

from maze_game.sb3_tests.features_extractor import CustomCombinedExtractor
from maze_game.sb3_tests.utils import linear_schedule

n_envs = 16
root_path = "storage"
model_name = "PPO"
run_id = get_latest_run_id(root_path, model_name)

monitor_kwargs = dict(
    info_keywords=("is_success",)
)

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(cnn_output_dim=256),
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

model_kwargs = dict(
    learning_rate=linear_schedule(0.002),
    # learning_rate=0.001,
    n_epochs=4,
    batch_size=512,
    n_steps=128,
    ent_coef=0.01,
    vf_coef=0.25,
    clip_range=0.2,
)

learn_kwargs = dict(
    total_timesteps=2_000_000,
    tb_log_name=model_name,
    callback=CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1),
        save_path=f"{root_path}/{model_name}_{run_id + 1}",
        name_prefix="save",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=2
    )
)
