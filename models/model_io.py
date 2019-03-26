class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    def __init__(
        self, state=None, hidden=None, target_class_embedding=None, action_probs=None
    ):
        self.state = state
        self.hidden = hidden
        self.target_class_embedding = target_class_embedding
        self.action_probs = action_probs


class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, embedding=None):

        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.embedding = embedding
