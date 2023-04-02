```bash
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_0 --stats --frames 1_000_000 --batch-size 512
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_0 --stats --frames 2_000_000 --lr 1e-4 --batch-size 512
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_0 --stats --frames 3_000_000 --lr 5e-5 --batch-size 512
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_0 --stats --frames 4_000_000 --lr 2.5e-5
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_0 --stats

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_2 --stats --frames 1_000_000 --batch-size 512
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_2 --stats --frames 2_000_000 --lr 1e-4 --batch-size 512
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_2 --stats --frames 3_000_000 --lr 5e-5 --batch-size 512
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_2 --stats

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_3f --stats --frames 1_000_000 --batch-size 512
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_3f --stats --frames 2_000_000 --lr 1e-4 --batch-size 512
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_3 --stats --frames 3_000_000 --lr 5e-5 --batch-size 512
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_3 --stats --frames 4_000_000 --lr 2.5e-5 --batch-size 512
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_3f --stats

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_4 --stats --frames 1_000_000 --lr 1e-3 --value-loss-coef 0.5
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_4 --stats --frames 2_000_000 --lr 1e-4 --value-loss-coef 0.5
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_4 --stats --frames 4_000_000 --lr 1e-5 --value-loss-coef 0.5
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_4 --stats

python -m scripts.train --algo a2c --env env_maze/MazeGame-v0 --model MazeGame_5 --stats --frames 1_000_000 --lr 1e-3
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_5 --stats


python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_7 --stats --lr 1e-2 --epochs 4 --frames-per-proc 256 --batch-size 512 --value-loss-coef 0.5 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_7 --stats


python -m scripts.train --algo a2c --env env_maze/MazeGame-v0 --model MazeGame_ --stats --epochs 10 --frames-per-proc 128 --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_ --stats

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_ppo_rec --recurrence 4 --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_ppo_rec --memory

python -m scripts.train --algo a2c --env env_maze/MazeGame-v0 --model MazeGame_a2c --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_a2c

tensorboard --logdir=maze_game/storage
tensorboard --logdir_spec sb3:maze_game/sb3_tests/storage,custom:maze_game/storage
tensorboard --logdir=maze_game/sb3_tests/storage 
```