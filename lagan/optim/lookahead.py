from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings
import random


class Lookahead(Optimizer):

    def __init__(self, optimizer, k=5, super_slow_k=5000, alpha=0.5, k_min=3, k_max=1000):
        self.optimizer = optimizer
        self.k_min = k_min
        self.k_max = k_max
        self.resample_k = k < 0
        self.k = k if k > 0 else random.randint(k_min, k_max)  # endpoints included
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.super_slow_state = defaultdict(dict)
        self.fast_state = self.optimizer.state

    def update_super_slow(self, group):
        print("In update_super_slow")
        for fast in group["params"]:
            param_state = self.state[fast]
            if "super_slow_param" not in param_state:
                param_state["super_slow_param"] = torch.zeros_like(fast.data)
                param_state["super_slow_param"].copy_(fast.data)
            sslow = param_state["super_slow_param"]
            sslow += (fast.data - sslow) * self.alpha
            slow = param_state["slow_param"]
            slow.data.copy_(sslow)
            fast.data.copy_(sslow)
            
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
        if self.resample_k:
            self.k = random.randint(self.k_min, self.k_max)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)
            
    def update_lookahead_super_slow(self):
        for group in self.param_groups:
            self.update_super_slow(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state
