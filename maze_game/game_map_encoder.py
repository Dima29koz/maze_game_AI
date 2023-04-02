import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler, maxabs_scale, minmax_scale

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

    num_layers = 2
    array = np.zeros((num_layers, rows, cols), dtype="uint8")

    for row in range(rows):
        for col in range(cols):
            obj = field[row][col]

            # array[0, row, col] = CELL_TO_IDX[type(obj)]
            array[0, row, col] = encode_cell(obj)
            # try:
            #     array[1, row, col] = CELL_DIR_TO_IDX[obj.direction]
            # except (AttributeError, KeyError):
            #     array[1, row, col] = 0
            # for i, direction in enumerate(Directions, start=2):
            #     array[row, col, i] = WALL_TO_IDX[type(obj.walls.get(direction))]

    for treasure in treasures:
        array[num_layers - 1, treasure.position.y, treasure.position.x] += 1

    return array
