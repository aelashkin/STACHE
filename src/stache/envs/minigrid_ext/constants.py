"""
This file contains constants used by the minigrid environments
for color, object, and state indexing.
"""


#TODO: Add action constants for different environments

COLOR_TO_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5,
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}


# Hardcoded action values for "empty" and "fetch" environments
ACTION_MAPPING_EMPTY = [
    (0, "left", "Turn left"),
    (1, "right", "Turn right"),
    (2, "forward", "Move forward"),
    (3, "pickup", "Unused"),
    (4, "drop", "Unused"),
    (5, "toggle", "Unused"),
    (6, "done", "Unused"),
]

ACTION_MAPPING_FETCH = [
    (0, "left", "Turn left"),
    (1, "right", "Turn right"),
    (2, "forward", "Move forward"),
    (3, "pickup", "Pick up an object"),
    (4, "drop", "Unused"),
    (5, "toggle", "Unused"),
    (6, "done", "Unused"),
]