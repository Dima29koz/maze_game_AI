import argparse
import os

from maze_game.game_core.bots_ai.core import BotAI, BotAIDebug
from maze_game.game_core.game_engine import Position, LevelPosition
from maze_game.multiagent.actions import action_to_action_space
from maze_game.multiagent.agent.ppo_agent import PPOAgent
from maze_game.multiagent.maze_multi_agent_env import create_duel_env

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="checkpoint path")


class DuelRunner:
    game: None
    rules: dict
    bot: BotAI | None

    def __init__(self, num_players: int, checkpoint_path: str, render_mode=None):
        self.env = create_duel_env(render_mode=render_mode, num_players=num_players, callback_on_step=self.process_turn)
        self.env.metadata['render_fps'] = 1000
        self.reset()

        self.agent = PPOAgent(num_players, checkpoint_path)

    def _init_bot(self):
        players_: dict[str, Position] = {
            player_name: Position(pl_pos.get('x'), pl_pos.get('y'))
            for pl_pos, player_name in self.env.unwrapped.players
        }
        self.bot = BotAI(self.rules, players_)
        self.bot.real_field = self.game.field.game_map.get_level(LevelPosition(0, 0, 0)).field  # todo only for testing

    def run_test(self):
        print('seed:', self.rules['generator_rules']['seed'])
        winner = None
        rl_agents = ['player_0']
        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            # print(agent, info)

            winner = info.get('turn_info').get('player_name')
            if termination or truncation:
                action = None
            else:
                if agent in rl_agents:
                    action = self.agent.compute_action(observation)
                    act_pl_abilities = self.game.get_allowed_abilities(self.game.get_current_player())
                    self.bot.turn_prepare(self.game.get_current_player().name, act_pl_abilities)
                else:
                    act_pl_abilities = self.game.get_allowed_abilities(self.game.get_current_player())
                    act = self.bot.make_decision(self.game.get_current_player().name, act_pl_abilities)
                    action = action_to_action_space(act)
            self.env.step(action)
            self.env.render()
        self.reset()
        # print(winner)
        return winner

    def process_turn(self, response):
        if self.bot:
            self.bot.process_turn_resp(response.get_raw_info())

    def reset(self):
        self.env.reset()
        self.game = self.env.unwrapped.game
        self.rules = self.env.unwrapped.rules
        self._init_bot()


def run_dueling_test(runner_: DuelRunner, iters=100):
    winners = []
    for i in range(iters):
        winner = runner_.run_test()
        winners.append(winner)
        if i % 100 == 0:
            calc_win_rate(winners)
    return winners


def calc_win_rate(wins):
    print('games:', len(wins))
    win_rate = {}
    for e in wins:
        if e not in win_rate.keys():
            win_rate |= {e: 0}
        win_rate[e] += 1
    print(win_rate)


if __name__ == '__main__':
    args = parser.parse_args()
    path = os.path.expanduser(args.path)
    runner = DuelRunner(num_players=2, checkpoint_path=path, render_mode=None)
    res = run_dueling_test(runner, iters=1000)
    calc_win_rate(res)
