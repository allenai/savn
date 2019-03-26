""" Borrowed from https://github.com/dgriff777/rl_a3c_pytorch. """

from __future__ import division

import math
from collections import defaultdict

import torch
import torch.optim as optim


class SharedAdam(optim.Optimizer):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self, params, args):
        # TODO: remove constants
        lr = args.lr
        betas = (0.9, 0.999)
        eps = 1e-3
        weight_decay = 0
        amsgrad = args.amsgrad
        defaults = defaultdict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(SharedAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["step"] = torch.zeros(1)
                    state["exp_avg"] = p.data.new().resize_as_(p.data).zero_()
                    state["exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()
                    state["max_exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()

        print("initialized optimizer.")

    def share_memory(self):
        print("attempting to share memory.")
        try:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"].share_memory_()
                    state["exp_avg"].share_memory_()
                    state["exp_avg_sq"].share_memory_()
                    state["max_exp_avg_sq"].share_memory_()
        except Exception as e:
            print(e)
        print("sharing memory.")

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till
                    # now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"].item()
                bias_correction2 = 1 - beta2 ** state["step"].item()
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
