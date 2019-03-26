""" Base class for all Agents. """
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.constants import DONE_ACTION_INT


class ThorAgent:
    """ Base class for all actor-critic agents. """

    def __init__(
        self, model, args, rank, episode=None, max_episode_length=1e3, gpu_id=-1
    ):
        self.gpu_id = gpu_id

        self._model = None
        self.model = model
        self._episode = episode
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.max_length = False
        self.hidden = None
        self.actions = []
        self.probs = []
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        self.max_episode_length = max_episode_length
        self.success = False
        self.backprop_t = 0
        torch.manual_seed(args.seed + rank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)

        self.verbose = args.verbose
        self.learned_loss = args.learned_loss
        self.learned_input = None
        self.learned_t = 0
        self.num_steps = args.num_steps
        self.hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())

    def eval_at_state(self, model_options):
        """ Eval at state. """
        raise NotImplementedError()

    @property
    def episode(self):
        """ Return the current episode. """
        return self._episode

    @property
    def environment(self):
        """ Return the current environmnet. """
        return self.episode.environment

    @property
    def state(self):
        """ Return the state of the agent. """
        raise NotImplementedError()

    @state.setter
    def state(self, value):
        raise NotImplementedError()

    @property
    def model(self):
        """ Returns the model. """
        return self._model

    def print_info(self):
        """ Print the actions. """
        for action in self.actions:
            print(action)

    @model.setter
    def model(self, model_to_set):
        self._model = model_to_set
        if self.gpu_id >= 0 and self._model is not None:
            with torch.cuda.device(self.gpu_id):
                self._model = self.model.cuda()

    def _increment_episode_length(self):
        self.eps_len += 1
        if self.eps_len >= self.max_episode_length:
            if not self.done:
                self.max_length = True
                self.done = True
            else:
                self.max_length = False
        else:
            self.max_length = False

    def action(self, model_options, training, demo=False):
        """ Train the agent. """
        if training:
            self.model.train()
        else:
            self.model.eval()

        model_input, out = self.eval_at_state(model_options)
        self.hidden = out.hidden
        prob = F.softmax(out.logit, dim=1)
        action = prob.multinomial(1).data
        log_prob = F.log_softmax(out.logit, dim=1)
        self.last_action_probs = prob
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, Variable(action))
        self.reward, self.done, self.info = self.episode.step(action[0, 0])

        if self.verbose:
            print(self.episode.actions_list[action])
        self.probs.append(prob)
        self.entropies.append(entropy)
        self.values.append(out.value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.actions.append(action)
        self.episode.prev_frame = model_input.state
        self.episode.current_frame = self.state()

        if self.learned_loss:
            res = torch.cat((self.hidden[0], self.last_action_probs), dim=1)
            if self.learned_input is None:
                self.learned_input = res
            else:
                self.learned_input = torch.cat((self.learned_input, res), dim=0)

        self._increment_episode_length()

        if self.episode.strict_done and action == DONE_ACTION_INT:
            self.success = self.info
            self.done = True
        elif self.done:
            self.success = not self.max_length
        return out.value, prob, action

    def reset_hidden(self, volatile=False):
        """ Reset the hidden state of the LSTM. """
        raise NotImplementedError()

    def repackage_hidden(self, volatile=False):
        """ Repackage the hidden state of the LSTM. """
        raise NotImplementedError()

    def clear_actions(self):
        """ Clear the information stored by the agent. """
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.probs = []
        self.reward = 0
        self.backprop_t = 0
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        self.learned_input = None
        self.learned_t = 0
        return self

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        raise NotImplementedError()

    def exit(self):
        """ Called on exit. """
        pass

    def reset_episode(self):
        """ Reset the episode so that it is identical. """
        return self._episode.reset()
