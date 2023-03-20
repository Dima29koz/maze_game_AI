```bash
python -m scripts.train --algo ppo --env env_maze/MazeGame-v0 --model MazeGame --save-interval 10 --frames 200_000

python -m scripts.visualize --env env_maze/MazeGame-v0 --model MazeGame
```