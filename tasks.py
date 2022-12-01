import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import pickle
import os
import sys
import json
import pdb
import random
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections as matcoll

import argparse

# from motifs import gen_fn_motifs
from utils import update_args, load_args, load_rb, Bunch

eps = 1e-6

mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = .5

cols = ['coral', 'cornflowerblue', 'magenta', 'orchid']

# dset_id is the name of the dataset (as saved)
# n is the index of the trial in the dataset
class Task:
    def __init__(self, t_len, dset_id=None, n=None):
        self.t_len = t_len
        self.dset_id = dset_id
        self.n = n

        self.L = 4
        self.Z = 6

    def get_x(self):
        pass

    def get_y(self):
        pass


# modalities in: 0
# modalities out: 0
class RSG(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        if args.intervals is None:
            t_o = np.random.randint(args.min_t, args.max_t)
        else:
            t_o = random.choice(args.intervals)
        t_p = int(t_o * args.gain)
        ready_time = np.random.randint(args.p_len * 2, args.max_ready)
        set_time = ready_time + t_o
        go_time = set_time + t_p

        self.t_type = args.t_type
        self.p_len = args.p_len
        self.rsg = (ready_time, set_time, go_time)
        self.t_o = t_o
        self.t_p = t_p
        self.gain = args.gain

    def get_x(self, args=None):
        rt, st, gt = self.rsg
        # ready pulse
        x_ready = np.zeros(self.t_len)
        x_ready[rt:rt+self.p_len] = 1
        # set pulse
        x_set = np.zeros(self.t_len)
        x_set[st:st+self.p_len] = 1
        # insert set pulse
        x = np.zeros((4, self.t_len))
        x[0] = x_set
        # perceptual shift. only to the ready signal
        if args is not None and args.m_noise != 0:
            x_ready = shift_x(x_ready, args.m_noise, self.t_o)
        x[0] += x_ready
        # noisy up/down corruption
        if args is not None and args.x_noise != 0:
            x = corrupt_x(args, x)
        return x

    def get_y(self, args=None):
        y = np.zeros((4, self.t_len))
        y0 = np.arange(self.t_len)
        slope = 1 / self.t_p
        y0 = y0 * slope - self.rsg[1] * slope
        # so the output value is not too large
        y0 = np.clip(y0, 0, 1.5)
        # RSG output is only 1D
        y[0] = y0
        return y

# modalities in: 0, 1, 2
# modalities out: 0, 1, 2
class Memory(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        if args.angles is None:
            theta = np.random.random() * 2 * np.pi
        else:
            theta = np.random.choice(args.angles) * np.pi / 180
        stimulus = [np.cos(theta), np.sin(theta)]

        self.t_type = args.t_type
        assert 'memory' in args.t_type
        self.stimulus = stimulus
        self.fix = args.fix_t
        self.stim = self.fix + args.stim_t
        self.memory = self.stim + np.random.randint(args.memory_t_min, args.memory_t_max)

        self.y_channel = args.y_channel
        self.fix_forever = args.fix_forever

    def get_x(self, args=None):
        x = np.zeros((4, self.t_len))
        x[0,:self.memory] = 1
        x[1,self.fix:self.stim] = self.stimulus[0]
        x[2,self.fix:self.stim] = self.stimulus[1]
        # noisy up/down corruption
        if args is not None and args.x_noise != 0:
            x = corrupt_x(args, x)
        return x

    def get_y(self, args=None):
        y = np.zeros((5, self.t_len))
        if self.fix_forever == 0:
            y[0,:self.memory] = 1
        else:
            y[0] = 1
        if self.y_channel == 1:
            y[1,self.memory:] = self.stimulus[0]
            y[2,self.memory:] = self.stimulus[1]
        elif self.y_channel == 2:
            y[3,self.memory:] = self.stimulus[0]
            y[4,self.memory:] = self.stimulus[1]
        # reversing output stimulus for anti condition
        if self.t_type.endswith('anti'):
            y[1:5,] = -y[1:5,]
        return y


# ways to add noise to x
def corrupt_x(args, x):
    x += np.random.normal(scale=args.x_noise, size=x.shape)
    return x

def shift_x(x, m_noise, t_p):
    if m_noise == 0:
        return x
    disp = int(np.random.normal(0, m_noise*t_p/50))
    x = np.roll(x, disp)
    return x

def create_dataset(args):
    t_type = args.t_type
    n_trials = args.n_trials

    if t_type.startswith('rsg'):
        assert args.max_ready + args.max_t + int(args.max_t * args.gain) < args.t_len
        TaskObj = RSG
    elif t_type == 'flip-flop':
        TaskObj = FlipFlop
    elif 'memory' in t_type:
        TaskObj = Memory
    else:
        raise NotImplementedError

    trials = []
    for n in range(n_trials):
        trial = TaskObj(args, dset_id=args.name, n=n)
        args.L = trial.L
        args.Z = trial.Z
        trials.append(trial)

    return trials, args

# turn task_args argument into usable argument variables
# lots of defaults are written down here
def get_task_args(args):
    tarr = args.task_args
    targs = Bunch()
    if args.t_type.startswith('rsg'):
        targs.t_len = get_tval(tarr, 'l', 600, int)
        targs.p_len = get_tval(tarr, 'pl', 5, int)
        targs.gain = get_tval(tarr, 'gain', 1, float)
        targs.max_ready = get_tval(tarr, 'max_ready', 80, int)
        if args.intervals is None:
            targs.min_t = get_tval(tarr, 'gt', targs.p_len * 4, int)
            targs.max_t = get_tval(tarr, 'lt', targs.t_len // 2 - targs.p_len * 4 - targs.max_ready, int)
        else:
            targs.max_t = max(args.intervals)
            targs.min_t = min(args.intervals)

    elif args.t_type == 'delay-copy':
        targs.t_len = get_tval(tarr, 'l', 500, int)
        targs.dim = get_tval(tarr, 'dim', 2, int)
        targs.n_freqs = get_tval(tarr, 'n_freqs', 20, int)
        targs.f_range = get_tval(tarr, 'f_range', [10, 40], float, n_vals=2)
        targs.amp = get_tval(tarr, 'amp', 1, float)

    elif args.t_type == 'flip-flop':
        targs.t_len = get_tval(tarr, 'l', 500, int)
        targs.dim = get_tval(tarr, 'dim', 3, int)
        targs.p_len = get_tval(tarr, 'pl', 5, int)
        targs.geop = get_tval(tarr, 'p', .02, float)

    elif 'memory' in args.t_type:
        targs.t_len = get_tval(tarr, 'l', 300, int)
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        targs.memory_t_min = get_tval(tarr, 'memory_min', -50, int)
        targs.memory_t_max = get_tval(tarr, 'memory_max', 100, int)
        targs.y_channel = get_tval(tarr, 'y', 1, int)
        targs.fix_forever = get_tval(tarr, 'ff', False, bool)

    return targs

# get particular value(s) given name and casting type
def get_tval(targs, name, default, dtype, n_vals=1):
    if name in targs:
        # set parameter(s) if set in command line
        idx = targs.index(name)
        if n_vals == 1: # one value to set
            val = dtype(targs[idx + 1])
        else: # multiple values to set
            vals = []
            for i in range(1, n_vals+1):
                vals.append(dtype(targs[idx + i]))
    else:
        # if parameter is not set in command line, set it to default
        val = default
    return val


def save_dataset(dset, name, config=None):
    fname = os.path.join('datasets', name + '.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(dset, f)
    gname = os.path.join('datasets', 'configs', name + '.json')
    if config is not None:
        with open(gname, 'w') as f:
            json.dump(config.to_json(), f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load', choices=['create', 'load'])
    parser.add_argument('name')
    parser.add_argument('-c', '--config', default=None, help='create from a config file')

    # general dataset arguments
    parser.add_argument('-t', '--t_type', default='memory', help='type of trial to create')
    parser.add_argument('-n', '--n_trials', type=int, default=2000)

    # task-specific arguments
    parser.add_argument('-a', '--task_args', nargs='*', default=[], help='terms to specify parameters of trial type')
    # rsg intervals
    parser.add_argument('-i', '--intervals', nargs='*', type=int, default=None, help='select from rsg intervals')
    # delay memory pro anti preset angles
    parser.add_argument('--angles', nargs='*', type=float, default=None, help='angles in degrees for dmpa tasks')
    

    args = parser.parse_args()
    if args.config is not None:
        # if using config file, load args from config, ignore everything else
        config_args = load_args(args.config)
        del config_args.name
        del config_args.config
        args = update_args(args, config_args)
    else:
        # add task-specific arguments. shouldn't need to do this if loading from config file
        task_args = get_task_args(args)
        args = update_args(args, task_args)

    args.argv = ' '.join(sys.argv)

    if args.mode == 'create':
        # create and save a dataset
        dset, config = create_dataset(args)
        save_dataset(dset, args.name, config=config)
    elif args.mode == 'load':
        # visualize a dataset
        dset = load_rb(args.name)
        t_type = type(dset[0])
        xr = np.arange(dset[0].t_len)

        samples = random.sample(dset, 12)
        fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(10,6))
        for i, ax in enumerate(fig.axes):
            ax.axvline(x=0, color='dimgray', alpha = 1)
            ax.axhline(y=0, color='dimgray', alpha = 1)
            ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            ax.tick_params(axis='both', color='white')
            #ax.set_title(sample[i][2])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            trial = samples[i]
            trial_x = trial.get_x()
            trial_y = trial.get_y()

            # if t_type in [RSG]:
            #     trial_x = np.sum(trial_x, axis=0)
            #     trial_y = trial_y[0]
            #     ml, sl, bl = ax.stem(xr, trial_x, use_line_collection=True, linefmt='coral', label='ready/set')
            #     ml.set_markerfacecolor('coral')
            #     ml.set_markeredgecolor('coral')
            #     if t_type == 'rsg-bin':
            #         ml, sl, bl = ax.stem(xr, [1], use_line_collection=True, linefmt='dodgerblue', label='go')
            #         ml.set_markerfacecolor('dodgerblue')
            #         ml.set_markeredgecolor('dodgerblue')
            #     else:
            #         ax.plot(xr, trial_y, color='dodgerblue', label='go', lw=2)
            #         if t_type is RSG:
            #             ax.set_title(f'{trial.rsg}: [{trial.t_o}, {trial.t_p}] ', fontsize=9)

            # elif t_type in [DelayProAnti, MemoryProAnti]:
            #     ax.plot(xr, trial_x[1], color='grey', lw=1, ls='--', alpha=.6)
            #     ax.plot(xr, trial_x[2], color='salmon', lw=1, ls='--', alpha=.6)
            #     ax.plot(xr, trial_x[3], color='dodgerblue', lw=1, ls='--', alpha=.6)
            #     ax.plot(xr, trial_y[1], color='grey', lw=1.5)
            #     ax.plot(xr, trial_y[2], color='salmon', lw=1.5)
            #     ax.plot(xr, trial_y[3], color='dodgerblue', lw=1.5)

            if t_type is Memory:
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                ax.plot(xr, trial_x[1], color='salmon', lw=1, ls='--', alpha=.6)
                ax.plot(xr, trial_x[2], color='dodgerblue', lw=1, ls='--', alpha=.6)
                ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)
                ax.plot(xr, trial_y[3], color='coral', lw=1.5)
                ax.plot(xr, trial_y[4], color='skyblue', lw=1.5)

        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()
