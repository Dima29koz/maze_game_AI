from enum import IntEnum

from maze_game.game_core import Actions as Acts, Directions


class Actions(IntEnum):
    # Pick up an object
    swap_treasure = 0

    # move directions
    move_top = 1
    move_right = 2
    move_bottom = 3
    move_left = 4

    # bombing directions
    bomb_top = 5
    bomb_right = 6
    bomb_bottom = 7
    bomb_left = 8

    # shot directions
    shoot_top = 9
    shoot_right = 10
    shoot_bottom = 11
    shoot_left = 12


def action_space_to_action(action: int):
    acts = {
        0: (Acts.swap_treasure, None),

        1: (Acts.move, Directions.top),
        2: (Acts.move, Directions.right),
        3: (Acts.move, Directions.bottom),
        4: (Acts.move, Directions.left),

        5: (Acts.throw_bomb, Directions.top),
        6: (Acts.throw_bomb, Directions.right),
        7: (Acts.throw_bomb, Directions.bottom),
        8: (Acts.throw_bomb, Directions.left),

        9: (Acts.shoot_bow, Directions.top),
        10: (Acts.shoot_bow, Directions.right),
        11: (Acts.shoot_bow, Directions.bottom),
        12: (Acts.shoot_bow, Directions.left),
    }
    return acts.get(action)


def action_to_action_space(action: tuple[Acts, Directions | None]):
    acts = {
        (Acts.swap_treasure, None): 0,

        (Acts.move, Directions.top): 1,
        (Acts.move, Directions.right): 2,
        (Acts.move, Directions.bottom): 3,
        (Acts.move, Directions.left): 4,

        (Acts.throw_bomb, Directions.top): 5,
        (Acts.throw_bomb, Directions.right): 6,
        (Acts.throw_bomb, Directions.bottom): 7,
        (Acts.throw_bomb, Directions.left): 8,

        (Acts.shoot_bow, Directions.top): 9,
        (Acts.shoot_bow, Directions.right): 10,
        (Acts.shoot_bow, Directions.bottom): 11,
        (Acts.shoot_bow, Directions.left): 12,
    }
    return acts.get(action)
