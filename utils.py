import os
import numpy as np

#import tensorflow as tf

import yaml
import logging
import time
import json
import csv
import pickle
import copy
import pdb
import re
# import pandas as pd

class LogObject(object):
    pass

# turn arbitrary file into args to be used
def load_args(path=None, to_bunch=True):
    if path:
        try:
            # maybe it's yaml
            config = yaml.safe_load(open(path))
        except:
            # maybe it's json
            config = json.load(open(path, 'r'))
    else:
        config = {}
    if to_bunch:
        try:
            return Bunch(config)
        except:
            print('Bunchify failed!')
    else:
        return config

# combine two args, overwriting with the second
# overwrite can take three possible values: True, False, or None (in which case overwrite Nones only)
def update_args(args, new_args, overwrite=True, to_bunch=True):
    dic = args if type(args) is dict else vars(args)
    new_dic = new_args if type(new_args) is dict else vars(new_args)
    for k in new_dic.keys():
        if overwrite is True or k not in dic or (dic[k] is None and overwrite is None) :
            dic[k] = new_dic[k]
    if to_bunch:
        return Bunch(dic)
    return dic


# produce run id and create log directory
def log_this(config, log_dir, log_name=None, checkpoints=False, use_id=True):
    run_id = str(int(time.time() * 100))[-7:]
    config.run_id = run_id
    print('\n=== Logging ===', flush=True)
    
    if log_name is None or len(log_name) == 0:
        log_name = run_id
    print(f'Run id: {run_id} with name {log_name}', flush=True)

    run_dir = os.path.join(log_dir, log_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f'Log folder: {run_dir}', flush=True)

    log_path = os.path.join(run_dir, f'log_{run_id}.log')
    print(f'Log file: {log_path}', flush=True)

    if checkpoints:
        checkpoint_dir = os.path.join(run_dir, f'checkpoints_{run_id}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f'Logging checkpoints to: {checkpoint_dir}', flush=True)
    else:
        checkpoint_dir = None

    # might want to send stdout here later too
    path_config = os.path.join(run_dir, f'config_{run_id}.json')
    with open(path_config, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, indent=4)
        print(f'Config file saved to: {path_config}', flush=True)

    log = LogObject()
    log.checkpoint_dir = checkpoint_dir
    log.run_dir = run_dir
    log.run_log = log_path
    log.run_id = run_id

    print('===============\n', flush=True)
    return log


# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991
class Bunch:
    def __init__(self, *args, **kwds):
        if len(args) > 0:
            if type(args[0]) is dict:
                self.__dict__ = copy.deepcopy(args[0])
            else:
                for k,v in args[0].__dict__.items():
                    self.__dict__[k] = copy.deepcopy(v)
                # self.__dict__.update(args[0].__dict__)
        self.__dict__.update(kwds)

    def __repr__(self):
        return 'Bunch(' + str(self.__dict__) + ')'

    def to_json(self):
        return copy.deepcopy(self.__dict__)

def load_rb(path):
    with open(path, 'rb') as f:
        qs = pickle.load(f)
    return qs

def lrange(l, p=0.1):
    return np.linspace(0, (l-1) * p, l)


# get config dictionary from the model path
def get_config(path, ctype='model', to_bunch=False):
    head, tail = os.path.split(path)
    if ctype == 'dset':
        fname = '.'.join(tail.split('.')[:-1]) + '.json'
        c_folder = os.path.join(head, 'configs')
        if os.path.isfile(os.path.join(c_folder, fname)):
            c_path = os.path.join(head, 'configs', fname)
        else:
            raise NotImplementedError

    elif ctype == 'model':
        if tail == 'model_best.pth' or 'test' in tail:
            for i in os.listdir(head):
                if i.startswith('config'):
                    c_path = os.path.join(head, i)
                    break
        else:
            folders = head.split('/')
            if folders[-1].startswith('checkpoints_'):
                run_id = folders[-1].split('_')[-1]
                c_path = os.path.join(*folders[:-1], 'config_'+run_id+'.json')
            else:
                run_id = re.split('_|\.', tail)[1]
                c_path = os.path.join(head, 'config_'+run_id+'.json')
        if not os.path.isfile(c_path):
            raise NotImplementedError
    else:
        raise NotImplementedError
    with open(c_path, 'r') as f:
        config = json.load(f)
    if to_bunch:
        return Bunch(**config)
    else:
        return config

