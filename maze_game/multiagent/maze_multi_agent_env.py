import random
from collections import deque

import gymnasium
import numpy as np
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from maze_game.game_core import SpectatorGUI, Game, Actions as Acts, Directions
from maze_game.game_core import base_rules as ru
from maze_game.game_core.game_engine.field import wall as w, cell as c
from maze_game.game_core.game_engine.global_env.enums import TreasureTypes
from maze_game.game_map_encoder import one_hot_encode
from maze_game.multiagent.actions import Actions, action_space_to_action
from maze_game.multiagent.observation_wraper import LastObservationWrapper


def create_env(render_mode=None, **kwargs):
    env = MAMazeGameEnv(render_mode=render_mode, **kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    env = LastObservationWrapper(env)
    return env


class MAMazeGameEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "maze_game_v2",
        "is_parallelizable": False,
        "render_fps": 2,
    }
    _skip_agent_selection: str | None
    _max_reward = 1
    game: Game

    def __init__(self, render_mode=None, size=5, max_steps=100, seed=None, num_players=2):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]

        self.gui: SpectatorGUI | None = None
        self.rules = {}
        self.players = []
        self.step_count = 0
        self._remaining_reward = {}

        self.stacked_observations: dict[str, dict[str, deque]] = {
            agent: {
                "field": deque(maxlen=1),
                "walls": deque(maxlen=4),
                "stats": deque(maxlen=4),
                "other_stats": deque(maxlen=4),
            } for agent in self.agents
        }

        self.action_spaces = {i: spaces.Discrete(len(Actions)) for i in self.agents}

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
        stats_observation_space = spaces.Box(
            low=0,
            high=self.size + 2,
            shape=(6, 4),
            dtype=np.float32
        )
        other_stats_observation_space = spaces.Box(
            low=0,
            high=self.size + 2,
            shape=(6, num_players - 1, 4),
            dtype=np.float32
        )

        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Dict(
                        {
                            "field": field_observation_space,
                            "walls": walls_observation_space,
                            "treasures": treasures_observation_space,
                            "stats": stats_observation_space,
                            "other_stats": other_stats_observation_space,
                        }
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(len(Actions),), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.reset(seed)

    def _setup_game_local(self, seed=None):
        self.rules = ru
        self.rules['generator_rules']['rows'] = self.size
        self.rules['generator_rules']['cols'] = self.size
        self.rules['generator_rules']['is_separated_armory'] = True
        self.rules['generator_rules']['seed'] = random.random() if seed is None else seed
        self.rules['gameplay_rules']['diff_outer_concrete_walls'] = True

    def _create_players(self):
        return [(self._create_spawn(), agent) for agent in self.agents]

    def _create_spawn(self) -> dict[str, int]:
        return {'x': random.randint(1, self.size), 'y': random.randint(1, self.size)}

    def _make_init_turns(self):
        for _ in self.players:
            self._process_turn(Acts.info, None)

    def _reset_stacked_obs(self):
        for agent in self.agents:
            field, walls, treasures = self._get_obs(agent)
            observation: dict[str, np.ndarray] = {
                "field": field,
                "walls": walls,
                "treasures": treasures,
                "stats": self._get_stats(agent),
                "other_stats": self._get_other_stats(agent),
                "action_mask": self._action_masks(agent),
            }

            for key, obs in observation.items():
                if key in self.stacked_observations[agent].keys():
                    st_obs = self.stacked_observations[agent][key]
                    for _ in range(st_obs.maxlen):
                        st_obs.append(obs)

    # todo сделать так чтобы оно играло против всех прошлых противников
    def _process_turn(self, action: Acts, direction: Directions | None):
        # assert valid move
        current_player = self.game.get_current_player()
        assert self.game.get_allowed_abilities(current_player).get(action) is True, "played illegal move."
        assert (
                current_player.name == self.agent_selection
        ), f'wrong player: current selection is {self.agent_selection}, current player is {current_player.name}'

        response, next_player = self.game.make_turn(action.name, direction.name if direction else None)
        if response is not None:
            self.infos[self.agent_selection] = {
                'turn_info': response.get_turn_info(),
                'resp': response.get_info(),
            }
            raw_info = response.get_raw_info().get('response', {})

            if action is Acts.shoot_bow:
                # contains all agents which were damaged (even if they were killed)
                dmg_agents = set(raw_info.get('dmg_pls', []))
                dead_agents = set(raw_info.get('dead_pls', []))
                drop_agents = set(raw_info.get('drop_pls', []))
                for agent in dead_agents:
                    self.rewards[agent] = -self._max_reward
                    self.truncations[agent] = True
                self._receive_reward(
                    self.agent_selection,
                    self._reward_shoot(
                        len(dmg_agents.difference(dead_agents)),
                        len(dead_agents),
                        len(drop_agents)
                    )
                )
            out_treasure = raw_info.get('type_out_treasure')
            if out_treasure:
                self._receive_reward(self.agent_selection, self._reward_treasure(out_treasure))

        if action is not Acts.swap_treasure:
            self.agent_selection = self._get_next_agent(self.agent_selection)

    def _action_masks(self, agent):
        mask = np.zeros(len(Actions), np.int8)
        if agent != self.agent_selection:
            return mask

        current_player = self.game.get_current_player()
        act_pl_abilities = self.game.get_allowed_abilities(current_player)

        mask[0] = act_pl_abilities.get(Acts.swap_treasure)
        for i, direction in enumerate(Directions, 1):
            wall = current_player.cell.walls[direction]
            mask[i] = not wall.player_collision or type(current_player.cell) is c.CellRiver
            if act_pl_abilities.get(Acts.throw_bomb):
                mask[4 + i] = wall.breakable and type(wall) is not w.WallEmpty
            # todo shooting mask may be modified
            if act_pl_abilities.get(Acts.shoot_bow):
                mask[8 + i] = not wall.weapon_collision

        return mask

    def _get_obs(self, agent):
        # todo custom observe for agents
        return one_hot_encode(self.game.field.game_map, self.game.field.treasures)

    def _get_other_stats(self, current_agent):
        players_filter = filter(lambda pl: pl.name != current_agent, self.game.field.players)
        return np.array(
            [self._get_player_stats(player) for player in players_filter],
            dtype=np.float32)

    def _get_stats(self, agent):
        player_filter = filter(lambda pl: pl.name == agent, self.game.field.players)
        player = next(player_filter)
        return np.array(
            self._get_player_stats(player),
            dtype=np.float32)

    @staticmethod
    def _get_player_stats(player):
        return [
            player.health / player.health_max,
            player.arrows / player.arrows_max,
            player.bombs / player.bombs_max,
            1 if player.treasure else 0,
            player.cell.position.x,
            player.cell.position.y,
        ]

    def _get_agent_location(self):
        x, y = self.game.get_current_player().cell.position.get()
        return x, y

    def _receive_reward(self, agent: str, reward: float):
        self.rewards[agent] = reward
        self._remaining_reward[agent] -= reward

    def _reward_shoot(self, num_dmg: int, num_dead: int, num_drop: int):
        if num_dead == 0:
            return 0
        if num_dead + 1 == len(self.agents):
            return self._remaining_reward[self.agent_selection]
        return num_dead / (len(self.agents) - 1) * (self._remaining_reward[self.agent_selection] / 2)

    def _reward_treasure(self, treasure_type):
        if treasure_type is TreasureTypes.very:
            return self._remaining_reward[self.agent_selection]
        return self._remaining_reward[self.agent_selection] / 2 / (len(self.agents) - 1)

    def observe(self, agent):
        st_obs = self.stacked_observations[agent]
        field, walls, treasures = self._get_obs(agent)

        st_obs['field'].append(field)
        st_obs['walls'].append(walls)
        st_obs['stats'].append(self._get_stats(agent))
        st_obs['other_stats'].append(self._get_other_stats(agent))

        return {
            "observation": {
                "field": np.array(st_obs['field']).reshape((13, self.size + 2, self.size + 2)),
                "walls": np.array(st_obs['walls']).transpose((1, 0, 2, 3)),
                "treasures": treasures,
                "stats": np.array(st_obs['stats']).transpose((1, 0)),
                "other_stats": np.array(st_obs['other_stats']).transpose((2, 1, 0)),
            },
            "action_mask": self._action_masks(agent),
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _was_dead_step(self, action: None) -> None:
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert (
                self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        assert self._skip_agent_selection is not None
        self.agent_selection = self._skip_agent_selection
        self._skip_agent_selection = None

    def _get_next_agent(self, current_agent: str):
        return self.agents[(self.agents.index(current_agent) + 1) % len(self.agents)]

    def step(self, action):
        self.step_count += 1
        current_agent = self.agent_selection
        if (
                self.truncations[current_agent]
                or self.terminations[current_agent]
        ):
            self._skip_agent_selection = self._get_next_agent(current_agent)
            return self._was_dead_step(action)

        self.rewards[current_agent] = 0
        act = action_space_to_action(action)
        self._process_turn(*act)

        if self.step_count >= self.max_steps * len(self.possible_agents):
            self.truncations = {i: True for i in self.agents}

        # check if there is a winner
        if self.game.is_win_condition(self.rules):
            for agent in self.agents:
                if agent != current_agent:
                    self.rewards[agent] = -self._max_reward
            self.terminations = {i: True for i in self.agents}

        self._accumulate_rewards()

    def reset(self, seed=None, options=None):
        # reset environment
        self.step_count = 0

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self._remaining_reward = {i: self._max_reward for i in self.agents}

        self.agent_selection = self.agents[0]

        self._setup_game_local(seed=seed)
        self.players = self._create_players()
        self.game = Game(rules=self.rules)
        for i, player in enumerate(self.players, 1):
            self.game.field.spawn_player(*player, turn=i)

        self._make_init_turns()
        self._reset_stacked_obs()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.render_mode == "human":
            if self.gui is None:
                self.gui = SpectatorGUI(self.game.field, None, self.metadata["render_fps"])
            self.gui.field = self.game.field

        if self.render_mode == "human":
            act_pl_abilities = self.game.get_allowed_abilities(self.game.get_current_player())
            self.gui.draw(act_pl_abilities, self.game.get_current_player().name)
            self.gui.get_action(act_pl_abilities)

    def close(self):
        if self.gui is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.gui = None
