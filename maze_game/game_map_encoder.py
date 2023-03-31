import numpy as np

from .game_core import Directions
from .game_core.game_engine.field import cell, wall
from .game_core.game_engine.field.game_map import GameMap
from .game_core.game_engine.global_env.types import LevelPosition

CELL_TO_IDX = {
    cell.NoneCell: 0,
    cell.Cell: 1,
    cell.CellRiver: 2,
    cell.CellRiverBridge: 3,
    cell.CellRiverMouth: 4,
    cell.CellExit: 5,
    cell.CellClinic: 6,
    cell.CellArmory: 7,
    cell.CellArmoryWeapon: 8,
    cell.CellArmoryExplosive: 9,
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


def encode(game_map: GameMap, treasures: list):
    field = game_map.get_level(LevelPosition(0, 0, 0)).field
    rows = len(field[0])
    cols = len(field)

    num_layers = 3
    array = np.zeros((rows, cols, num_layers), dtype="uint8")

    for row in range(rows):
        for col in range(cols):
            obj = field[row][col]

            array[row, col, 0] = CELL_TO_IDX[type(obj)]
            try:
                array[row, col, 1] = CELL_DIR_TO_IDX[obj.direction]
            except (AttributeError, KeyError):
                array[row, col, 1] = 0
            # for i, direction in enumerate(Directions, start=2):
            #     array[row, col, i] = WALL_TO_IDX[type(obj.walls.get(direction))]

    for treasure in treasures:
        array[treasure.position.y, treasure.position.x, num_layers-1] += 1

    return array
