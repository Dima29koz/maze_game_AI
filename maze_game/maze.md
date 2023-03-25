```bash
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame --save-interval 10 --frames 200_000
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame --save-interval 10 --frames 400_000
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_2 --stats --lr 2.5e-4 --epochs 10 --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_2 --stats

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_3 --stats --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_3 --stats

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_4 --stats --lr 2.5e-4 --epochs 10 --save-interval 10 --frames 100_000
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_4 --stats --lr 2.5e-4 --epochs 10 --save-interval 100 --frames 100_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_4 --stats

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_5 --stats --lr 1e-4 --epochs 10 --frames-per-proc 1000 --save-interval 10 --frames 100_000
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_5 --stats --lr 1e-4 --epochs 10 --frames-per-proc 1000 --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_5 --stats

max 100 steps
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_6   --stats --lr 1e-3 --epochs 10 --frames-per-proc 1000 --save-interval 10 --frames 1_000_000
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_6_0 --stats --lr 1e-3 --epochs 10 --frames-per-proc 1000 --batch-size 1024 --save-interval 10 --frames 1_000_000
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_6_1 --stats --lr 1e-3 --epochs 10 --frames-per-proc 1000 --save-interval 10 --frames 1_000_000
python -m scripts.train --algo a2c --env env_maze/MazeGame-v0 --model MazeGame_6_2 --stats --lr 1e-3 --epochs 10 --frames-per-proc 1000 --save-interval 10 --frames 1_000_000
python -m scripts.train --algo a2c --env env_maze/MazeGame-v0 --model MazeGame_6_3 --stats --lr 1e-3 --epochs 10 --frames-per-proc 200 --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_6 --stats

max 100 steps
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_6_1 --stats --lr 1e-4 --epochs 10 --frames-per-proc 100 --save-interval 10 --frames 1_000_000
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_6_1 --stats --lr 1e-4 --epochs 10 --frames-per-proc 100 --save-interval 10 --frames 2_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_6_1 --stats

max 100 steps
python -m scripts.train --algo a2c --env env_maze/MazeGame-v0 --model MazeGame_7 --stats --epochs 10 --frames-per-proc 128 --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_7 --stats

python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame_ppo_rec --recurrence 4 --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_ppo_rec --memory

python -m scripts.train --algo a2c --env env_maze/MazeGame-v0 --model MazeGame_a2c --save-interval 10 --frames 1_000_000
python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame_a2c

tensorboard --logdir=maze_game/storage
```