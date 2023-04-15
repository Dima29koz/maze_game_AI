from enum import IntEnum

from maze_game.game_core import Actions as Acts, Directions


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


def action_space_to_action(action: int):
    acts = {
        0: (Acts.move, Directions.top),
        1: (Acts.move, Directions.right),
        2: (Acts.move, Directions.bottom),
        3: (Acts.move, Directions.left),

        4: (Acts.throw_bomb, Directions.top),
        5: (Acts.throw_bomb, Directions.right),
        6: (Acts.throw_bomb, Directions.bottom),
        7: (Acts.throw_bomb, Directions.left),

        8: (Acts.swap_treasure, None),
    }
    return acts.get(action)
