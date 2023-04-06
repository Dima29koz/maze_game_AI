import numpy as np

from .game_core import Directions
from .game_core.game_engine.field import cell, wall
from .game_core.game_engine.field.game_map import GameMap
from .game_core.game_engine.global_env.types import LevelPosition

CELL_TO_IDX = {
    cell.NoneCell: 0,
    cell.Cell: 1,
    cell.CellExit: 2,
    cell.CellClinic: 3,
    cell.CellArmory: 4,
    cell.CellArmoryWeapon: 5,
    cell.CellArmoryExplosive: 6,
    cell.CellRiverBridge: 7,
    cell.CellRiverMouth: 8,
    cell.CellRiver: 9,
}

WALL_TO_IDX = {
    wall.WallEmpty: 0,
    wall.WallConcrete: 1,
    wall.WallOuter: 2,
    wall.WallExit: 3,
    wall.WallEntrance: 4,
    wall.WallRubber: 5,
}

CELL_DIR_TO_IDX = {
    Directions.top: 1,
    Directions.right: 2,
    Directions.bottom: 3,
    Directions.left: 4,
}


def encode_cell(obj: cell.CELL):
    res = CELL_TO_IDX[type(obj)]
    if type(obj) is cell.CellRiver:
        res += CELL_DIR_TO_IDX[obj.direction] - 1
    return res


def encode(game_map: GameMap, treasures: list):
    field = game_map.get_level(LevelPosition(0, 0, 0)).field
    rows = len(field[0])
    cols = len(field)

    num_layers = 6
    array = np.zeros((num_layers, rows, cols), dtype="uint8")

    for row in range(rows):
        for col in range(cols):
            obj = field[row][col]

            # array[0, row, col] = CELL_TO_IDX[type(obj)]
            array[0, row, col] = encode_cell(obj)
            for i, direction in enumerate(Directions, start=1):
                array[i, row, col] = WALL_TO_IDX[type(obj.walls.get(direction))]

    for treasure in treasures:
        array[num_layers - 1, treasure.position.y, treasure.position.x] += 1

    return array


def one_hot_encode(game_map: GameMap, treasures: list):
    field = game_map.get_level(LevelPosition(0, 0, 0)).field
    rows = len(field[0])
    cols = len(field)

    num_layers = 26
    array = np.zeros((num_layers, rows, cols), dtype="uint8")
    for row in range(rows):
        for col in range(cols):
            obj = field[row][col]

            # 9 types of cell 0-8 + 4 river directions 9-12
            if type(obj) is not cell.CellRiver:
                array[CELL_TO_IDX[type(obj)], row, col] = 1
            else:
                array[8 + CELL_DIR_TO_IDX[obj.direction], row, col] = 1

            # 4*3 atr of wall 13-24
            for i, direction in enumerate(Directions):
                array[13 + i*3, row, col] = int(obj.walls.get(direction).breakable)
                array[14 + i*3, row, col] = int(obj.walls.get(direction).weapon_collision)
                array[15 + i*3, row, col] = int(obj.walls.get(direction).player_collision)

    for treasure in treasures:
        array[num_layers - 1, treasure.position.y, treasure.position.x] += 1

    return array
