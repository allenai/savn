from __future__ import division

import torch.nn as nn
from .basemodel import BaseModel
from .tcn import TemporalConvNet


class SAVN(BaseModel):
    def __init__(self, args):
        super(SAVN, self).__init__(args)
        self.args = args

        self.feature_size = args.hidden_state_sz + args.action_space
        self.learned_input_sz = args.hidden_state_sz + args.action_space

        self.num_steps = args.num_steps
        self.ll_key = nn.Linear(self.feature_size, self.feature_size)
        self.ll_linear = nn.Linear(self.feature_size, self.feature_size)
        self.ll_tc = TemporalConvNet(
            self.num_steps, [10, 1], kernel_size=2, dropout=0.0
        )

    def learned_loss(self, hx, H, params=None):
        H_input = H.unsqueeze(0)
        x = self.ll_tc(H_input, params).squeeze(0)
        return x.pow(2).sum(1).pow(0.5)
