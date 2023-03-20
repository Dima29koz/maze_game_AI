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


def encode(game_map: GameMap, treasures: list, agent_x, agent_y):
    field = game_map.get_level(LevelPosition(0, 0, 0)).field
    rows = len(field[0])
    cols = len(field)

    array = np.zeros((rows, cols, 7), dtype="uint8")

    for row in range(rows):
        for col in range(cols):
            obj = field[row][col]

            array[row, col, 0] = CELL_TO_IDX[type(obj)]
            for i, direction in enumerate(Directions, start=1):
                array[row, col, i] = WALL_TO_IDX[type(obj.walls.get(direction))]

    for treasure in treasures:
        array[treasure.position.y, treasure.position.x, 5] += 1

    array[agent_y, agent_x, 6] = 1
    return array
