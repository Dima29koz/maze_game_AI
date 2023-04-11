from enum import IntEnum
import random
from typing import Generator
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from maze_game.game_map_encoder import one_hot_encode
from maze_game.game_core import SpectatorGUI, Game, Directions, Actions as Acts
from maze_game.game_core import base_rules as ru
from maze_game.game_core.game_engine.field import wall as w


class MazeGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    # Enumeration of possible actions
    class Actions(IntEnum):
        # move directions
        move_top = 0
        move_right = 1
        move_bottom = 2
        move_left = 3

        # bombing directions
        bomb_top = 4
        bomb_right = 5
        bomb_bottom = 6
        bomb_left = 7

        # Pick up an object
        swap_treasure = 8

        # get info
        # info = 5

    def __init__(self, render_mode=None, size=5, max_steps=250, seed=None):
        self.size = size
        self.max_steps = max_steps

        self.gui: SpectatorGUI | None = None
        self.rules = {}
        self.players = []
        self.turns: list | Generator = []

        self.last_walls_observations = deque(maxlen=4)
        self.last_stats_observations = deque(maxlen=4)

        self.step_count = 0

        self.response = None

        self._setup_game_local(seed=seed)
        self.game = Game(rules=self.rules)
        field = self.game.field
        for i, player in enumerate(self.players, 1):
            field.spawn_player(*player, turn=i)

        self._make_init_turns()

        field_observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(13, self.size + 2, self.size + 2),
            dtype=np.uint8,
        )
        walls_observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(12, 4, self.size + 2, self.size + 2),
            dtype=np.uint8,
        )
        treasures_observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(1, self.size + 2, self.size + 2),
            dtype=np.uint8,
        )

        self.observation_space = spaces.Dict(
            {
                "field": field_observation_space,
                "walls": walls_observation_space,
                "treasures": treasures_observation_space,
                "stats": spaces.Box(0, 7, shape=(6, 4), dtype=np.float32),
            }
        )

        self.actions = MazeGameEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.invalid_actions: list[int] = []

        self._action_space_to_action = {
            0: (Acts.move, Directions.top),
            1: (Acts.move, Directions.right),
            2: (Acts.move, Directions.bottom),
            3: (Acts.move, Directions.left),

            4: (Acts.throw_bomb, Directions.top),
            5: (Acts.throw_bomb, Directions.right),
            6: (Acts.throw_bomb, Directions.bottom),
            7: (Acts.throw_bomb, Directions.left),

            8: (Acts.swap_treasure, None),
            # 100: (Acts.info, None)
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _setup_game_local(self, seed=None):
        self.rules = ru
        # self.rules['generator_rules']['river_rules']['has_river'] = False
        # self.rules['generator_rules']['walls']['has_walls'] = False
        self.rules['generator_rules']['rows'] = self.size
        self.rules['generator_rules']['cols'] = self.size
        self.rules['generator_rules']['is_separated_armory'] = True
        self.rules['generator_rules']['seed'] = random.random() if seed is None else seed
        # self.rules['generator_rules']['seed'] = 1
        self.rules['gameplay_rules']['fast_win'] = False
        self.rules['gameplay_rules']['diff_outer_concrete_walls'] = True
        # spawn: dict[str, int] = {'x': 5, 'y': 3}
        spawn: dict[str, int] = {'x': random.randint(1, self.size), 'y': random.randint(1, self.size)}
        spawn2: dict[str, int] = {'x': 2, 'y': 4}
        spawn3: dict[str, int] = {'x': 2, 'y': 1}

        self.players = [
            (spawn, 'Skipper'),
            # (spawn2, 'Tester'),
            # (spawn3, 'player'),
        ]
        self.turns = []

    def _make_init_turns(self):
        for _ in self.players:
            act = (Acts.info, None)
            is_running = self._process_turn(*act)

    def _process_turn(self, action: Acts, direction: Directions):
        response, next_player = self.game.make_turn(action.name, direction.name if direction else None)
        self.success = False
        if response is not None:
            self.response = response
            self.success = True
        # print(response.get_turn_info(), response.get_info())
        # if self.bot:
        #     # print(response.get_raw_info())
        #     self.bot.process_turn_resp(response.get_raw_info())
        #     self.bot.turn_prepare(self.game.get_current_player().name)
        if self.game.is_win_condition(self.rules):
            return False
        return True

    def action_masks(self):
        current_player = self.game.get_current_player()
        act_pl_abilities = self.game.get_allowed_abilities(current_player)

        mask = [False] * len(self.actions)
        mask[-1] = act_pl_abilities.get(Acts.swap_treasure)
        for i, direction in enumerate(Directions):
            wall = current_player.cell.walls[direction]
            mask[i] = not wall.player_collision
            if act_pl_abilities.get(Acts.throw_bomb):
                mask[4 + i] = wall.breakable and type(wall) is not w.WallEmpty

        return mask

    def _get_obs(self):
        field, walls, treasures = self._get_field()
        return {
            "field": field,
            "walls": walls,
            "treasures": treasures,
            "stats": self._get_stats(),
        }

    def _get_info(self):
        return {
            "step": self.step_count,
            "turn_info": self.response.get_turn_info(),
            "info": self.response.get_info(),
            "is_success": False,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.rules = {}
        self.players = []
        self.turns: list | Generator = []

        self.step_count = 0

        self.response = None

        self._setup_game_local(seed=seed)
        self.game = Game(rules=self.rules)
        field = self.game.field
        for i, player in enumerate(self.players, 1):
            field.spawn_player(*player, turn=i)

        self._make_init_turns()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            if self.gui is None:
                self.gui = SpectatorGUI(self.game.field, None, self.metadata["render_fps"])
            self.gui.field = field

        for _ in range(4):
            self.last_walls_observations.append(observation['walls'])
            self.last_stats_observations.append(observation['stats'])

        observation['stats'] = np.array(self.last_stats_observations).transpose()
        observation['walls'] = np.array(self.last_walls_observations).transpose((1, 0, 2, 3))
        return observation, info

    def step(self, action):
        self.step_count += 1

        reward = 0
        is_running = True
        truncated = False

        act = self._action_space_to_action[action]
        current_player = self.game.get_current_player()
        if act[0] is Acts.swap_treasure and current_player.treasure is None and self.game.get_allowed_abilities(
                current_player).get(Acts.swap_treasure):
            # todo если нет клада в руке и можно его подобрать и действие == подобрать
            reward = self._reward()
        is_running = self._process_turn(*act)

        observation = self._get_obs()
        info = self._get_info()

        terminated = not is_running
        if not is_running:
            # todo если выиграл
            reward = self._reward()
            info["is_success"] = True

        if self.step_count >= self.max_steps:
            truncated = True
            info["TimeLimit.truncated"] = True

        self.last_walls_observations.append(observation['walls'])
        self.last_stats_observations.append(observation['stats'])
        observation['walls'] = np.array(self.last_walls_observations).transpose((1, 0, 2, 3))
        observation['stats'] = np.array(self.last_stats_observations).transpose()
        return observation, reward, terminated, truncated, info

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            act_pl_abilities = self.game.get_allowed_abilities(self.game.get_current_player())
            self.gui.draw(act_pl_abilities, self.game.get_current_player().name)
            self.gui.get_action({})

    def close(self):
        if self.gui is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_field(self):
        return one_hot_encode(self.game.field.game_map, self.game.field.treasures)

    def _get_agent_location(self):
        x, y = self.game.get_current_player().cell.position.get()
        return x, y

    def _get_stats(self):
        player = self.game.get_current_player()
        return np.array(
            [
                player.health / player.health_max,
                player.arrows / player.arrows_max,
                player.bombs / player.bombs_max,
                1 if player.treasure else 0,
                *self._get_agent_location()
            ],
            dtype=np.float32)
