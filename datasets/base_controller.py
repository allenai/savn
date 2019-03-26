""" Base Controller for interacting with the Scene """


class BaseController:
    def __init__(self):
        super(BaseController, self).__init__()

    def start(self):
        raise NotImplementedError()

    def reset(self, scene_name=None):
        raise NotImplementedError()

    def step(self, action, raise_for_failure=False):
        raise NotImplementedError()
