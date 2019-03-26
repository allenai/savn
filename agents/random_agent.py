import torch

from .agent import ThorAgent
from utils.net_util import gpuify

from episodes.basic_episode import BasicEpisode
from models.model_io import ModelInput, ModelOutput


class RandomNavigationAgent(ThorAgent):
    def __init__(self, create_model, args, rank, gpu_id):
        max_episode_length = args.max_episode_length
        episode = BasicEpisode(args, gpu_id, args.strict_done)
        super(RandomNavigationAgent, self).__init__(
            create_model(args), args, rank, episode, max_episode_length, gpu_id
        )
        self.action_space = args.action_space

    """ A random navigation agent. """

    def eval_at_state(self, params=None):
        critic = torch.ones(1, 1)
        actor = torch.ones(1, self.action_space)
        critic = gpuify(critic, self.gpu_id)
        actor = gpuify(actor, self.gpu_id)
        return ModelInput(), ModelOutput(value=critic, logit=actor)

    def reset_hidden(self, volatile=False):
        pass

    def repackage_hidden(self, volatile=False):
        pass

    def preprocess_frame(self, frame):
        return None

    def state(self):
        return None

    def sync_with_shared(self, shared_model):
        return
