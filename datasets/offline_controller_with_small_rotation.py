""" Exhaustive BFS and Offline Controller. """

import importlib
from collections import deque
import json
import copy
import time
import random
import os
import platform

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from ai2thor.controller import Controller, distance
from .base_controller import BaseController

class ThorAgentState:
    """ Representation of a simple state of a Thor Agent which includes
        the position, horizon and rotation. """

    def __init__(self, x, y, z, rotation, horizon):
        self.x = round(x, 2)
        self.y = y
        self.z = round(z, 2)
        self.rotation = round(rotation)
        self.horizon = round(horizon)

    @classmethod
    def get_state_from_evenet(cls, event, forced_y=None):
        """ Extracts a state from an event. """
        state = cls(
            x=event.metadata["agent"]["position"]["x"],
            y=event.metadata["agent"]["position"]["y"],
            z=event.metadata["agent"]["position"]["z"],
            rotation=event.metadata["agent"]["rotation"]["y"],
            horizon=event.metadata["agent"]["cameraHorizon"],
        )
        if forced_y != None:
            state.y = forced_y
        return state

    def __eq__(self, other):
        """ If we check for exact equality then we get issues.
            For now we consider this 'close enough'. """
        if isinstance(other, ThorAgentState):
            return (
                self.x == other.x
                and
                # self.y == other.y and
                self.z == other.z
                and self.rotation == other.rotation
                and self.horizon == other.horizon
            )
        return NotImplemented

    def __str__(self):
        """ Get the string representation of a state. """
        """
        return '{:0.2f}|{:0.2f}|{:0.2f}|{:d}|{:d}'.format(
            self.x,
            self.y,
            self.z,
            round(self.rotation),
            round(self.horizon)
        )
        """
        return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
            self.x, self.z, round(self.rotation), round(self.horizon)
        )

    def position(self):
        """ Returns just the position. """
        return dict(x=self.x, y=self.y, z=self.z)


class ExhaustiveBFSController(Controller):
    """ A much slower and more exhaustive version of the BFSController.
        This may be helpful if you wish to find the shortest path to an object.
        The usual BFSController does not consider things like rotate or look down
        when you are navigating towards an object. Additionally, there is some
        rare occurances of positions which you can only get to in a certain way.
        This ExhaustiveBFSController introduces the safe_teleport method which
        ensures that all states will be covered. 
        Strongly recomend having a seperate directory for each scene. See 
        OfflineControllerWithSmallRotation for more information on how the generated data may be used. """

    def __init__(
        self,
        grid_size=0.25,
        fov=90.0,
        grid_file=None,
        graph_file=None,
        metadata_file=None,
        images_file=None,
        seg_file=None,
        class_file=None,
        depth_file=None,
        debug_mode=True,
        grid_assumption=False,
        local_executable_path=None,
        actions=["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"],
    ):

        super(ExhaustiveBFSController, self).__init__()
        # Allowed rotations.
        self.rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        # Allowed horizons.
        self.horizons = [0, 30]

        self.allow_enqueue = True
        self.queue = deque()
        self.seen_points = []
        self.grid_points = []
        self.seen_states = []
        self.bad_seen_states = []
        self.visited_seen_states = []
        self.grid_states = []
        self.grid_size = grid_size
        self._check_visited = False
        self.scene_name = None
        self.fov = fov
        self.y = None

        self.local_executable_path = local_executable_path

        # distance_threshold to be consistent with BFSController in generating grid.
        self.distance_threshold = self.grid_size / 5.0
        self.debug_mode = debug_mode
        self.actions = actions
        self.grid_assumption = grid_assumption

        self.grid_file = grid_file
        self.metadata_file = metadata_file
        self.graph_file = graph_file
        self.images_file = images_file
        self.seg_file = seg_file
        self.class_file = class_file
        self.depth_file = depth_file

        # Optionally make a gird (including x,y,z points that are reachable)
        self.make_grid = grid_file is not None

        # Optionally store the metadata of each state.
        self.make_metadata = metadata_file is not None

        # Optionally make a directed of (s,t) where exists a in self.actions
        # such that t is reachable via s via a.
        self.make_graph = graph_file is not None

        # Optionally store an hdf5 file which contains the frame for each state.
        self.make_images = images_file is not None

        self.make_seg = seg_file is not None
        self.make_class = class_file is not None

        self.make_depth = self.depth_file is not None

        self.metadata = {}
        self.classdata = {}

        self.graph = None
        if self.make_graph:
            import networkx as nx

            self.graph = nx.DiGraph()

        if self.make_images:
            import h5py

            self.images = h5py.File(self.images_file, "w")

        if self.make_seg:
            import h5py

            self.seg = h5py.File(self.seg_file, "w")

        if self.make_depth:
            import h5py

            self.depth = h5py.File(self.depth_file, "w")

    def safe_teleport(self, state):
        """ Approach a state from all possible directions if the usual teleport fails. """
        self.step(dict(action="Rotate", rotation=0))
        event = self.step(dict(action="Teleport", x=state.x, y=state.y, z=state.z))
        if event.metadata["lastActionSuccess"]:
            return event

        # Approach from the left.
        event = self.step(
            dict(action="Teleport", x=(state.x - self.grid_size), y=state.y, z=state.z)
        )
        if event.metadata["lastActionSuccess"]:
            self.step(dict(action="Rotate", rotation=90))
            event = self.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                return event

        # Approach from the right.
        event = self.step(
            dict(action="Teleport", x=(state.x + self.grid_size), y=state.y, z=state.z)
        )
        if event.metadata["lastActionSuccess"]:
            self.step(dict(action="Rotate", rotation=270))
            event = self.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                return event

        # Approach from the back.
        event = self.step(
            dict(action="Teleport", x=state.x, y=state.y, z=state.z - self.grid_size)
        )
        if event.metadata["lastActionSuccess"]:
            self.step(dict(action="Rotate", rotation=0))
            event = self.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                return event

        # Approach from the front.
        event = self.step(
            dict(action="Teleport", x=state.x, y=state.y, z=state.z + self.grid_size)
        )
        if event.metadata["lastActionSuccess"]:
            self.step(dict(action="Rotate", rotation=180))
            event = self.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                return event

        print(self.scene_name)
        print(str(state))
        raise Exception("Safe Teleport Failed")

    def teleport_to_state(self, state):
        """ Only use this method when we know the state is valid. """
        event = self.safe_teleport(state)
        assert event.metadata["lastActionSuccess"]
        event = self.step(dict(action="Rotate", rotation=state.rotation))
        assert event.metadata["lastActionSuccess"]
        event = self.step(dict(action="Look", horizon=state.horizon))
        assert event.metadata["lastActionSuccess"]

        if self.debug_mode:
            # Sanity check that we have teleported to the correct state.
            new_state = self.get_state_from_event(event)
            if state != new_state:
                print(state)
                print(new_state)
            assert state == new_state
        return event

    def get_state_from_event(self, event):
        return ThorAgentState.get_state_from_evenet(event, forced_y=self.y)

    def get_point_from_event(self, event):
        return event.metadata["agent"]["position"]

    def get_next_state(self, state, action, copy_state=False):
        """ Guess the next state when action is taken. Note that
            this will not predict the correct y value. """
        if copy_state:
            next_state = copy.deepcopy(state)
        else:
            next_state = state
        if action == "MoveAhead":
            if next_state.rotation == 0:
                next_state.z += self.grid_size
            elif next_state.rotation == 90:
                next_state.x += self.grid_size
            elif next_state.rotation == 180:
                next_state.z -= self.grid_size
            elif next_state.rotation == 270:
                next_state.x -= self.grid_size
            elif next_state.rotation == 45:
                next_state.z += self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 135:
                next_state.z -= self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 225:
                next_state.z -= self.grid_size
                next_state.x -= self.grid_size
            elif next_state.rotation == 315:
                next_state.z += self.grid_size
                next_state.x -= self.grid_size
            else:
                raise Exception("Unknown Rotation")
        elif action == "RotateRight":
            next_state.rotation = (next_state.rotation + 45) % 360
        elif action == "RotateLeft":
            next_state.rotation = (next_state.rotation - 45) % 360
        elif action == "LookUp":
            if abs(next_state.horizon) <= 1:
                return None
            next_state.horizon = next_state.horizon - 30
        elif action == "LookDown":
            if abs(next_state.horizon - 60) <= 1 or abs(next_state.horizon - 30) <= 1:
                return None
            next_state.horizon = next_state.horizon + 30
        return next_state

    def add_edge(self, curr_state, next_state):
        self.graph.add_edge(str(curr_state), str(next_state))

    def enqueue_state(self, state):
        """ Returns true if state is valid. """
        # ensure there are no dup states.
        if state in self.seen_states:
            return True

        if state in self.bad_seen_states:
            return False

        # ensure state is a legal rotation and horizon.
        if (
            round(state.horizon) not in self.horizons
            or round(state.rotation) not in self.rotations
        ):
            self.bad_seen_states.append(state)
            return False

        self.seen_states.append(state)
        self.queue.append(state)
        return True

    def enqueue_states(self, agent_state):

        if not self.allow_enqueue:
            return

        # Take all action in self.action and enqueue if they are valid.
        for action in self.actions:

            next_state_guess = self.get_next_state(agent_state, action, True)

            if next_state_guess is None:
                continue

            # # Bug.
            # if (
            #     self.scene_name == "FloorPlan208_physics"
            #     and next_state_guess.x == 0
            #     and next_state_guess.z == 1.75
            # ):
            #     self.teleport_to_state(agent_state)
            #     continue

            # Grid assumption is meant to make things faster and should not
            # be used in practice. In general it does not work when the y
            # values fluctuate in a scene. It circumvents using the actual controller.
            if self.grid_assumption:
                if next_state_guess in self.seen_states:
                    if self.make_graph:
                        self.add_edge(agent_state, next_state_guess)
                    continue

            event = self.step(
                dict(
                    action="Teleport",
                    x=next_state_guess.x,
                    y=next_state_guess.y,
                    z=next_state_guess.z,
                )
            )
            if not event.metadata["lastActionSuccess"]:
                self.teleport_to_state(agent_state)
                continue
            event = self.step(dict(action="Rotate", rotation=next_state_guess.rotation))
            if not event.metadata["lastActionSuccess"]:
                self.teleport_to_state(agent_state)
                continue
            event = self.step(dict(action="Look", horizon=next_state_guess.horizon))
            if not event.metadata["lastActionSuccess"]:
                self.teleport_to_state(agent_state)
                continue

            next_state = self.get_state_from_event(event)

            if next_state != next_state_guess:
                print(next_state)
                print(next_state_guess)
            assert next_state == next_state_guess

            if self.enqueue_state(next_state) and self.make_graph:
                self.add_edge(agent_state, next_state)

            # Return back to agents initial location.
            self.teleport_to_state(agent_state)

    def search_all_closed(self, scene_name):
        """ Runs the ExhaustiveBFSController on scene_name. """
        self.allow_enqueue = True
        self.queue = deque()
        self.seen_points = []
        self.visited_seen_points = []
        self.grid_points = []
        self.seen_states = []
        self.visited_seen_states = []
        self.scene_name = scene_name
        event = self.reset(scene_name)

        if self.make_seg or self.make_class:
            event = self.step(
                dict(
                    action="Initialize",
                    gridSize=self.grid_size,
                    fieldOfView=self.fov,
                    renderClassImage=True,
                    renderObjectImage=True,
                    renderDepthImage=True,
                )
            )
        else:
            event = self.step(
                dict(
                    action="Initialize",
                    renderDepthImage=True,
                    gridSize=self.grid_size,
                    fieldOfView=self.fov,
                )
            )
        self.y = event.metadata["agent"]["position"]["y"]

        self.enqueue_state(self.get_state_from_event(event))

        while self.queue:
            self.queue_step()

        if self.make_grid:
            with open(self.grid_file, "w") as outfile:
                json.dump(self.grid_points, outfile)
        if self.make_graph:
            from networkx.readwrite import json_graph

            with open(self.graph_file, "w") as outfile:
                data = json_graph.node_link_data(self.graph)
                json.dump(data, outfile)
        if self.make_metadata:
            with open(self.metadata_file, "w") as outfile:
                json.dump(self.metadata, outfile)
        if self.make_images:
            self.images.close()
        if self.make_seg:
            self.seg.close()
        if self.make_depth:
            self.depth.close()

        if self.make_class:
            with open(self.class_file, "w") as outfile:
                json.dump(self.classdata, outfile)

        print("Finished :", self.scene_name)

    def queue_step(self):
        search_state = self.queue.popleft()
        event = self.teleport_to_state(search_state)

        # if search_state.y > 1.3:
        #    raise Exception("**** got big point ")

        self.enqueue_states(search_state)
        self.visited_seen_states.append(search_state)

        if self.make_grid and not any(
            map(
                lambda p: distance(p, search_state.position())
                < self.distance_threshold,
                self.grid_points,
            )
        ):
            self.grid_points.append(search_state.position())

        if self.make_metadata:
            self.metadata[str(search_state)] = event.metadata

        if self.make_class:
            class_detections = event.class_detections2D
            for k, v in class_detections.items():
                class_detections[k] = str(v)
            self.classdata[str(search_state)] = class_detections

        if self.make_images and str(search_state) not in self.images:
            self.images.create_dataset(str(search_state), data=event.frame)

        if self.make_seg and str(search_state) not in self.seg:
            self.seg.create_dataset(
                str(search_state), data=event.class_segmentation_frame
            )

        if self.make_depth and str(search_state) not in self.depth:
            self.depth.create_dataset(str(search_state), data=event.depth_frame)

        elif str(search_state) in self.images:
            print(self.scene_name, str(search_state))


class OfflineControllerWithSmallRotationEvent:
    """ A stripped down version of an event. Only contains lastActionSuccess, sceneName,
        and optionally state and frame. Does not contain the rest of the metadata. """

    def __init__(self, last_action_success, scene_name, state=None, frame=None):
        self.metadata = {
            "lastActionSuccess": last_action_success,
            "sceneName": scene_name,
        }
        if state is not None:
            self.metadata["agent"] = {}
            self.metadata["agent"]["position"] = state.position()
            self.metadata["agent"]["rotation"] = {
                "x": 0.0,
                "y": state.rotation,
                "z": 0.0,
            }
            self.metadata["agent"]["cameraHorizon"] = state.horizon
        self.frame = frame


class OfflineControllerWithSmallRotation(BaseController):
    """ A stripped down version of the controller for non-interactive settings.
        Only allows for a few given actions. Note that you must use the
        ExhaustiveBFSController to first generate the data used by OfflineControllerWithSmallRotation.
        Data is stored in offline_data_dir/<scene_name>/.

        Can swap the metadata.json for a visible_object_map.json. A script for generating
        this is coming soon. If the swap is made then the OfflineControllerWithSmallRotation is faster and
        self.using_raw_metadata will be set to false.

        Additionally, images.hdf5 may be swapped out with ResNet features or anything
        that you want to be returned for event.frame. """

    def __init__(
        self,
        grid_size=0.25,
        fov=100,
        offline_data_dir="/mnt/6tb/mitchellw/data/living_room_offline_data",
        grid_file_name="grid.json",
        graph_file_name="graph.json",
        metadata_file_name="visible_object_map.json",
        # metadata_file_name='metadata.json',
        images_file_name="images.hdf5",
        debug_mode=True,
        actions=["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"],
        visualize=True,
        local_executable_path=None,
    ):

        super(OfflineControllerWithSmallRotation, self).__init__()
        self.grid_size = grid_size
        self.offline_data_dir = offline_data_dir
        self.grid_file_name = grid_file_name
        self.graph_file_name = graph_file_name
        self.metadata_file_name = metadata_file_name
        self.images_file_name = images_file_name
        self.grid = None
        self.graph = None
        self.metadata = None
        self.images = None
        self.controller = None
        self.using_raw_metadata = True
        self.actions = actions
        # Allowed rotations.
        self.rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        # Allowed horizons.
        self.horizons = [0, 30]
        self.debug_mode = debug_mode
        self.fov = fov

        self.local_executable_path = local_executable_path

        self.y = None

        self.last_event = None

        self.controller = ExhaustiveBFSController()
        if self.local_executable_path is not None:
            self.controller.local_executable_path = self.local_executable_path

        self.visualize = visualize

        self.scene_name = None
        self.state = None
        self.last_action_success = True

        self.h5py = importlib.import_module("h5py")
        self.nx = importlib.import_module("networkx")
        self.json_graph_loader = importlib.import_module("networkx.readwrite")

    def start(self):
        if self.visualize:
            self.controller.start()
            self.controller.step(
                dict(action="Initialize", gridSize=self.grid_size, fieldOfView=self.fov)
            )

    def get_full_state(self, x, y, z, rotation=0.0, horizon=0.0):
        return ThorAgentState(x, y, z, rotation, horizon)

    def get_state_from_str(self, x, z, rotation=0.0, horizon=0.0):
        return ThorAgentState(x, self.y, z, rotation, horizon)

    def reset(self, scene_name=None):

        if scene_name is None:
            scene_name = "FloorPlan28"

        if scene_name != self.scene_name:
            self.scene_name = scene_name
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.grid_file_name
                ),
                "r",
            ) as f:
                self.grid = json.load(f)
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.graph_file_name
                ),
                "r",
            ) as f:
                graph_json = json.load(f)
            self.graph = self.json_graph_loader.node_link_graph(
                graph_json
            ).to_directed()
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.metadata_file_name
                ),
                "r",
            ) as f:
                self.metadata = json.load(f)
                # Determine if using the raw metadata, which is structured as a dictionary of
                # state -> metatdata. The alternative is a map of obj -> states where object is visible.
                key = next(iter(self.metadata.keys()))
                try:
                    float(key.split("|")[0])
                    self.using_raw_metadata = True
                except ValueError:
                    self.using_raw_metadata = False

            if self.images is not None:
                self.images.close()
            self.images = self.h5py.File(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.images_file_name
                ),
                "r",
            )

        self.state = self.get_full_state(
            **self.grid[0], rotation=random.choice(self.rotations)
        )
        self.y = self.state.y
        self.last_action_success = True
        self.last_event = self._successful_event()

        if self.visualize:
            self.controller.reset(scene_name)
            self.controller.teleport_to_state(self.state)

    def randomize_state(self):
        self.state = self.get_state_from_str(
            *[float(x) for x in random.choice(list(self.images.keys())).split("|")]
        )
        self.state.horizon = 0
        self.last_action_success = True
        self.last_event = self._successful_event()

        if self.visualize:
            self.controller.teleport_to_state(self.state)

    def back_to_start(self, start):
        self.state = start
        if self.visualize:
            self.controller.teleport_to_state(self.state)

    def step(self, action, raise_for_failure=False):

        if "action" not in action or action["action"] not in self.actions:
            if action["action"] == "Initialize":
                if self.visualize:
                    self.controller.step(action, raise_for_failure)
                return
            raise Exception("Unsupported action.")

        action = action["action"]

        next_state = self.controller.get_next_state(self.state, action, True)

        if self.visualize and next_state is not None:
            viz_event = self.controller.step(
                dict(action="Teleport", x=next_state.x, y=next_state.y, z=next_state.z)
            )
            viz_event = self.controller.step(
                dict(action="Rotate", rotation=next_state.rotation)
            )
            viz_event = self.controller.step(
                dict(action="Look", horizon=next_state.horizon)
            )
            viz_next_state = self.controller.get_state_from_event(viz_event)
            if (
                round(viz_next_state.horizon) not in self.horizons
                or round(viz_next_state.rotation) not in self.rotations
            ):
                # return back to original state.
                self.controller.teleport_to_state(self.state)

        if next_state is not None:
            next_state_key = str(next_state)
            neighbors = self.graph.neighbors(str(self.state))

            if next_state_key in neighbors:
                self.state = self.get_state_from_str(
                    *[float(x) for x in next_state_key.split("|")]
                )
                self.last_action_success = True
                event = self._successful_event()
                if self.debug_mode and self.visualize:
                    if self.controller.get_state_from_event(
                        viz_event
                    ) != self.controller.get_state_from_event(event):
                        print(action)
                        print(str(self.controller.get_state_from_event(viz_event)))
                        print(str(self.controller.get_state_from_event(event)))

                    assert self.controller.get_state_from_event(
                        viz_event
                    ) == self.controller.get_state_from_event(event)
                    assert viz_event.metadata["lastActionSuccess"]

                    # Uncomment if you want to view the frames side by side to
                    # ensure that they are duplicated.
                    # from matplotlib import pyplot as plt
                    # fig = plt.figure()
                    # fig.add_subplot(2,1,1)
                    # plt.imshow(self.get_image())
                    # fig.add_subplot(2,1,2)
                    # plt.imshow(viz_event.frame)
                    # plt.show()

                self.last_event = event
                return event

        self.last_action_success = False
        self.last_event.metadata["lastActionSuccess"] = False
        return self.last_event

    def shortest_path(self, source_state, target_state):
        return self.nx.shortest_path(self.graph, str(source_state), str(target_state))

    def optimal_plan(self, source_state, path):
        """ This is for debugging. It modifies the state. """
        self.state = source_state
        actions = []
        i = 1
        while i < len(path):
            for a in self.actions:
                next_state = self.controller.get_next_state(self.state, a, True)
                if str(next_state) == path[i]:
                    actions.append(a)
                    i += 1
                    self.state = next_state
                    break

        return actions

    def shortest_path_to_target(self, source_state, objId, get_plan=False):
        """ Many ways to reach objId, which one is best? """
        states_where_visible = []
        if self.using_raw_metadata:
            for s in self.metadata:
                objects = self.metadata[s]["objects"]
                visible_objects = [o["objectId"] for o in objects if o["visible"]]
                if objId in visible_objects:
                    states_where_visible.append(s)
        else:
            states_where_visible = self.metadata[objId]

        # transform from strings into states
        states_where_visible = [
            self.get_state_from_str(*[float(x) for x in str_.split("|")])
            for str_ in states_where_visible
        ]

        best_path = None
        best_path_len = 0
        for t in states_where_visible:
            path = self.shortest_path(source_state, t)
            if len(path) < best_path_len or best_path is None:
                best_path = path
                best_path_len = len(path)
        best_plan = []

        if get_plan:
            best_plan = self.optimal_plan(source_state, best_path)

        return best_path, best_path_len, best_plan

    def visualize_plan(self, source, plan):
        """ Visualize the best path from source to plan. """
        assert self.visualize
        self.controller.teleport_to_state(source)
        time.sleep(0.5)
        for a in plan:
            print(a)
            self.controller.step(dict(action=a))
            time.sleep(0.5)

    def object_is_visible(self, objId):
        if self.using_raw_metadata:
            objects = self.metadata[str(self.state)]["objects"]
            visible_objects = [o["objectId"] for o in objects if o["visible"]]
            return objId in visible_objects
        else:
            return str(self.state) in self.metadata[objId]

    def _successful_event(self):
        return OfflineControllerWithSmallRotationEvent(
            self.last_action_success, self.scene_name, self.state, self.get_image()
        )

    def get_image(self):
        return self.images[str(self.state)][:]

    def all_objects(self):
        if self.using_raw_metadata:
            return [o["objectId"] for o in self.metadata[str(self.state)]["objects"]]
        else:
            return self.metadata.keys()
