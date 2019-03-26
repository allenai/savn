from datasets.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_UP,
    LOOK_DOWN,
    DONE,
)


def get_actions(args):
    assert args.action_space == 6, "Expected 6 possible actions."
    return [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE]
