import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb
import random
import copy
import sys

from utils import Bunch, load_rb, update_args
from helpers import get_activation

# for easy rng manipulation
class TorchSeed:
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.rng_pt = torch.get_rng_state()
        torch.manual_seed(self.seed)
    def __exit__(self, type, value, traceback):
        torch.set_rng_state(self.rng_pt)


DEFAULT_ARGS = {
    'L': 4,
    'Z': 4,
    
    'D1': 5,
    'D2': 0,
    'N': 100,
    'M': 50

    'rnn_init_g': 1.5,
    'planner_noise': 0,
    'actor_noise': 0,

    'ff_bias': True,
    'res_bias': False,

    'm1_act': 'none',
    'm2_act': 'none',
    'out_act': 'none',
    'model_path': None,
    'res_path': None,

    'network_seed': None,
    'planner_seed': None,
    'planner_x_seed': None,

    'actor_seed': None,
    'actor_x_seed': None
}

class TaskNet(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = update_args(DEFAULT_ARGS, args)
       
        if self.args.network_seed is None:
            self.args.network_seed = random.randrange(1e6)

        self.out_act = get_activation(self.args.out_act)
        self.m1_act = get_activation(self.args.m1_act)
        self.m2_act = get_activation(self.args.m2_act)

        self._init_vars()
        self.reset()

    def _init_vars(self):
        with TorchSeed(self.args.network_seed):
            D1 = self.args.D1 if self.args.D1 != 0 else self.args.N

            if self.args.D1_T == 0:
                self.M_u = nn.Linear(self.args.L + self.args.T, D1, bias=self.args.ff_bias)
            else:
                self.M_u_T = nn.Linear(self.args.T, self.args.D1_T, bias=self.args.ff_bias)
                self.M_u_L = nn.Linear(self.args.L, D1, bias=self.args.ff_bias)


            self.M_ro = nn.Linear(N, self.args.Z, bias=self.args.ff_bias)
        self.reservoir = M2Reservoir(self.args)

        self.actor = Actor(self.args)
        self.planner = Planner(self.args)

        if self.args.model_path is not None:
            self.load_state_dict(torch.load(self.args.model_path))

    def add_task(self):
        M = self.M_u.weight.data
        self.M_u.weight.data = torch.cat((M, torch.zeros((M.shape[0],1))), dim=1)
        self.args.T += 1

    def forward(self, o, extras=False):
        # pass through the forward part
        # o should have shape [batch size, self.args.T + self.args.L]
        if self.args.D1_T == 0:
            u = self.m1_act(self.M_u(o))
        else:
            T_lim = self.args.D1 - self.args.D1_T
            # pdb.set_trace()
            u_L = self.M_u_L(o[:,:self.args.L]) # the actual input stuff
            u_T = self.M_u_T(o[:,self.args.L:]) # just the task info
            if self.args.c_noise > 0:
                u_T = u_T + torch.normal(torch.zeros_like(u_T), self.args.c_noise)
            # pdb.set_trace()
            u_T = torch.tanh(u_T)

            u = u_L
            u[:,T_lim:] += u_T # edit just the first D1_T entries
            # u = self.m1_act(torch.stack([self.M_u_L(o[:,:self.args.D1_T]), self.M_u_T(o[:,self.args.D1_T:])]), dim=1)
        # if hasattr(self.args, 'net_fb') and self.args.net_fb:
        #     self.z = self.z.expand(o.shape[0], self.z.shape[1])
        #     oz = torch.cat((o, self.z), dim=1)
        #     u = self.m1_act(self.M_u(oz))
        # else:
        #     u = self.m1_act(self.M_u(o))
        if extras:
            v, etc = self.reservoir(u, extras=True)
        else:
            v = self.reservoir(u, extras=False)
        z = self.M_ro(self.m2_act(v))
        self.z = self.out_act(z)

        if not extras:
            return self.z
        elif self.args.use_reservoir:
            return self.z, {'u': u, 'x': etc['x'], 'v': v}
        else:
            return self.z, {'u': u, 'v': v}

    def reset(self, res_state=None, device=None):
        self.z = torch.zeros((1, self.args.Z))
        if self.args.use_reservoir:
            self.reservoir.reset(res_state=res_state, device=device)

class Actor(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = update_args(DEFAULT_ARGS, args)

        if self.args.actor_seed is None:
            self.args.actor_seed = random.randrange(1e6)
        if self.args.actor_x_seed is None:
            self.args.actor_x_seed = np.random.randint(1e6)

        self.tau_x = 10
        self.activation = torch.tanh

        # use second set of dynamics equations as in jazayeri papers
        self.dynamics_mode = 0

        self._init_vars()
        self.reset()

    def _init_vars(self):
        if self.args.res_path is not None:
            self.load_state_dict(torch.load(self.args.res_path))
        else:
            with TorchSeed(self.args.actor_seed):
                # input layer
                self.W_u = nn.Linear(self.args.D, self.args.N, bias=False)
                torch.nn.init.normal_(self.W_u.weight.data, std=self.args.rnn_init_g / np.sqrt(self.args.D))

                # recurrent weights
                self.J = nn.Linear(self.args.N, self.args.N, bias=self.args.res_bias)
                torch.nn.init.normal_(self.J.weight.data, std=self.args.rnn_init_g / np.sqrt(self.args.N))

                # output layer
                self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=self.args.ff_bias)


    def burn_in(self, steps):
        for i in range(steps):
            g = torch.tanh(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    # extras currently doesn't do anything. maybe add x val, etc.
    def forward(self, u=None, extras=False):
        if self.dynamics_mode == 0:
            if u is None:
                g = self.activation(self.J(self.x))
            else:
                g = self.activation(self.J(self.x) + self.W_u(u))
            # adding any inherent rnn noise
            if self.args.actor_noise > 0:
                nn = torch.normal(torch.zeros_like(g), self.args.actor_noise)
                # pdb.set_trace()
                g = g + nn
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x

            v = self.W_ro(self.x)

        elif self.dynamics_mode == 1:
            if u is None:
                g = self.J(self.r)
            else:
                g = self.J(self.r) + self.W_u(u)
            if self.args.actor_noise > 0:
                gn = g + torch.normal(torch.zeros_like(g), self.args.actor_noise)
            else:
                gn = g
            delta_x = (-self.x + gn) / self.tau_x
            self.x = self.x + delta_x
            self.r = self.activation(self.x)

            v = self.W_ro(self.r)

        if extras:
            etc = {'x': self.x.detach()}
            return v, etc
        return v

    def reset(self, res_state=None, burn_in=True, device=None):
        if res_state is None:
            # load specified hidden state from seed
            res_state = self.args.actor_x_seed

        if type(res_state) is np.ndarray:
            # load an actual particular hidden state
            # if there's an error here then highly possible that res_state has wrong form
            self.x = torch.as_tensor(res_state).float()
        elif type(res_state) is torch.Tensor:
            self.x = res_state
        elif res_state == 'zero' or res_state == -1:
            # reset to 0
            self.x = torch.zeros((1, self.args.N))
        elif res_state == 'random' or res_state == -2:
            # reset to totally random value without using reservoir seed
            self.x = torch.normal(0, 1, (1, self.args.N))
        elif type(res_state) is int and res_state >= 0:
            # if any seed set, set the net to that seed and burn in
            with TorchSeed(res_state):
                self.x = torch.normal(0, 1, (1, self.args.N))
        else:
            print('not any of these types, something went wrong')
            pdb.set_trace()

        if device is not None:
            self.x = self.x.to(device)

        if self.dynamics_mode == 1:
            self.r = self.activation(self.x)

        if burn_in:
            self.burn_in(100)


class Planner(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = update_args(DEFAULT_ARGS, args)

        if self.args.planner_seed is None:
            self.args.planner_seed = random.randrange(1e6)
        if self.args.planner_x_seed is None:
            self.args.planner_x_seed = np.random.randint(1e6)

        self.tau_x = 10
        self.activation = torch.tanh

        self._init_vars()
        self.reset()

    def _init_vars(self):
        if self.args.res_path is not None:
            self.load_state_dict(torch.load(self.args.res_path))
        else:
            with TorchSeed(self.args.planner_seed):
                # input layer
                self.W_u = nn.Linear(self.args.C + self.args.S, self.args.M, bias=False)
                torch.nn.init.normal_(self.W_u.weight.data, std=self.args.rnn_init_g / np.sqrt(self.args.D))

                # recurrent weights
                self.J = nn.Linear(self.args.M, self.args.M, bias=self.args.res_bias)
                torch.nn.init.normal_(self.J.weight.data, std=self.args.rnn_init_g / np.sqrt(self.args.M))

                # output layer
                self.W_ro = nn.Linear(self.args.M, self.args.D, bias=self.args.ff_bias)


    def burn_in(self, steps):
        for i in range(steps):
            g = torch.tanh(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    # extras currently doesn't do anything. maybe add x val, etc.
    def forward(self, u=None, extras=False):
        if u is None:
            g = self.activation(self.J(self.x))
        else:
            g = self.activation(self.J(self.x) + self.W_u(u))
        # adding any inherent rnn noise
        if self.args.planner_noise > 0:
            nn = torch.normal(torch.zeros_like(g), self.args.planner_noise)
            # pdb.set_trace()
            g = g + nn
        delta_x = (-self.x + g) / self.tau_x
        self.x = self.x + delta_x

        v = self.W_ro(self.x)

        if extras:
            etc = {'x': self.x.detach()}
            return v, etc
        return v

    def reset(self, res_state=None, burn_in=True, device=None):
        if res_state is None:
            # load specified hidden state from seed
            res_state = self.args.planner_x_seed

        if type(res_state) is np.ndarray:
            # load an actual particular hidden state
            # if there's an error here then highly possible that res_state has wrong form
            self.x = torch.as_tensor(res_state).float()
        elif type(res_state) is torch.Tensor:
            self.x = res_state
        elif res_state == 'zero' or res_state == -1:
            # reset to 0
            self.x = torch.zeros((1, self.args.M))
        elif res_state == 'random' or res_state == -2:
            # reset to totally random value without using reservoir seed
            self.x = torch.normal(0, 1, (1, self.args.M))
        elif type(res_state) is int and res_state >= 0:
            # if any seed set, set the net to that seed and burn in
            with TorchSeed(res_state):
                self.x = torch.normal(0, 1, (1, self.args.M))
        else:
            print('not any of these types, something went wrong')
            pdb.set_trace()

        if device is not None:
            self.x = self.x.to(device)

        if burn_in:
            self.burn_in(100)

