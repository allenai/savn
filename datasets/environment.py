"""A wrapper for engaging with the THOR environment."""

import copy
import json
import os
import random
from .offline_controller_with_small_rotation import OfflineControllerWithSmallRotation


class Environment:
    """ Abstraction of the ai2thor enviroment. """

    def __init__(
        self,
        use_offline_controller,
        grid_size=0.25,
        fov=100.0,
        offline_data_dir="~/data/offline_data/",
        images_file_name="images.hdf5",
        local_executable_path=None,
    ):

        self.offline_data_dir = offline_data_dir
        self.use_offline_controller = use_offline_controller
        self.images_file_name = images_file_name
        self.controller = OfflineControllerWithSmallRotation(
            grid_size=grid_size,
            fov=fov,
            offline_data_dir=offline_data_dir,
            images_file_name=images_file_name,
            visualize=False,
            local_executable_path=local_executable_path,
        )
        self.grid_size = grid_size
        self._reachable_points = None
        self.start_state = None
        self.last_action = None
        self.fov = fov

    @property
    def scene_name(self):
        return self.controller.last_event.metadata["sceneName"]

    @property
    def current_frame(self):
        return self.controller.last_event.frame

    @property
    def last_event(self):
        return self.controller.last_event

    @property
    def last_action_success(self):
        if self.use_offline_controller:
            return self.controller.last_action_success
        return self.controller.last_event.metadata["lastActionSuccess"]

    def object_is_visible(self, objId):
        if not self.use_offline_controller:
            objects = self.last_event.metadata["objects"]
            visible_objects = [o["objectId"] for o in objects if o["visible"]]
            return objId in visible_objects

        return self.controller.object_is_visible(objId)

    def start(self, scene_name):
        """ Begin the scene. """
        self.controller.start()
        self.reset(scene_name=scene_name)

    def reset(self, scene_name):
        """ Reset the scene. """
        self.controller.reset(scene_name)
        self.controller.step(
            dict(action="Initialize", gridSize=self.grid_size, fieldOfView=self.fov)
        )

    def all_objects(self):
        if not self.use_offline_controller:
            objects = self.controller.last_event.metadata["objects"]
            return [o["objectId"] for o in objects]
        return self.controller.all_objects()

    def step(self, action_dict):
        return self.controller.step(action_dict)

    def teleport_agent_to(self, x, y, z, rotation, horizon):
        """ Teleport the agent to (x,y,z) with given rotation and horizon. """
        self.controller.step(dict(action="Teleport", x=x, y=y, z=z))
        self.controller.step(dict(action="Rotate", rotation=rotation))
        self.controller.step(dict(action="Look", horizon=horizon))

    def random_reachable_state(self, seed=None):
        """ Get a random reachable state. """
        if seed is not None:
            random.seed(seed)
        xyz = random.choice(self.reachable_points)
        rotation = random.choice([0, 90, 180, 270])
        horizon = random.choice([0, 30, 330])
        state = copy.copy(xyz)
        state["rotation"] = rotation
        state["horizon"] = horizon
        return state

    def randomize_agent_location(self, seed=None):
        """ Put agent in a random reachable state. """
        if not self.use_offline_controller:
            state = self.random_reachable_state(seed=seed)
            self.teleport_agent_to(**state)
            self.start_state = copy.deepcopy(state)
            return

        self.controller.randomize_state()
        self.start_state = copy.deepcopy(self.controller.state)

    def back_to_start(self):

        if self.start_state is None:
            self.reset(self.scene_name)
            return

        if not self.use_offline_controller:
            self.teleport_agent_to(**self.start_state)
        else:
            self.controller.back_to_start(self.start_state)

    @property
    def reachable_points(self):
        """ Use the JSON file to get the reachable points. """
        if self._reachable_points is not None:
            return self._reachable_points

        points_path = os.path.join(self.offline_data_dir, self.scene_name, "grid.json")
        if not os.path.exists(points_path):
            raise IOError("Path {0} does not exist".format(points_path))
        self._reachable_points = json.load(open(points_path))
        return self._reachable_points
