"Defines the tasks that an agent should complete in a given environment."

class Episode(object):
    """Manages an episode in the THOR env."""

    def __init__(self, **kwargs):
        pass

    @property
    def environment(self):
        raise NotImplementedError()

    def state_for_agent(self):
        raise NotImplementedError()

    def step(self, action_as_int):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    @property
    def actions_list(self):
        raise NotImplementedError()

    @property
    def total_actions(self):
        return len(self.actions_list)

    def index_to_action(self, index):
        """Given an action index, referring to possible_actions of ActionUtil(),
        converts to an usable action. """
        assert 0 <= index < self.total_actions
        return self.actions_list[index]
