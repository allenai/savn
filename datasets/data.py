from .scene_util import get_scenes

from .constants import (
    KITCHEN_OBJECT_CLASS_LIST,
    LIVING_ROOM_OBJECT_CLASS_LIST,
    BEDROOM_OBJECT_CLASS_LIST,
    BATHROOM_OBJECT_CLASS_LIST,
    FULL_OBJECT_CLASS_LIST,
)


def name_to_num(name):
    return ["kitchen", "living_room", "bedroom", "bathroom"].index(name)


def num_to_name(num):
    return ["kitchen", "", "living_room", "bedroom", "bathroom"][int(num / 100)]


def get_data(scene_types, scenes):

    mapping = ["kitchen", "living_room", "bedroom", "bathroom"]
    idx = []
    for j in range(len(scene_types)):
        idx.append(mapping.index(scene_types[j]))

    scenes = [
        get_scenes("[{}]+{}".format(num + int(num > 0), scenes)) for num in [0, 1, 2, 3]
    ]

    possible_targets = FULL_OBJECT_CLASS_LIST

    targets = [
        KITCHEN_OBJECT_CLASS_LIST,
        LIVING_ROOM_OBJECT_CLASS_LIST,
        BEDROOM_OBJECT_CLASS_LIST,
        BATHROOM_OBJECT_CLASS_LIST,
    ]

    return [scenes[i] for i in idx], possible_targets, [targets[i] for i in idx]
