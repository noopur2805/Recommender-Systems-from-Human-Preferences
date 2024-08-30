from collections import defaultdict
import logging
import numpy as np
import os
import json
import random 
import torch

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | Overall Steps: {:>8}\n".format(
            self.stats["steps"][-1][0])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "steps":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(
                np.mean([x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        if self.console_logger:
            self.console_logger.info(log_str)
        else:
            print(log_str)

import time

def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    return time_str(time_left)


def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string


def process_reward(h_response, reward_type):
    #assert reward_type in ['return_time', 'session_length','mix']
    if reward_type == 0: #'return_time':
        reward=h_response[:,0] # return time reward, integer range from -1 to 4
    elif reward_type == 1: #'session_length':
        reward=h_response[:,1]
    elif reward_type == 2: #'mix':
        reward=0.7*(h_response[:,0])+0.3*(h_response[:,1])
    return reward.unsqueeze(-1)

def process_reward_eva(h_response_t, reward_type):
    
    if reward_type == 0: #'return_time':
        reward=h_response_t[0] # return time reward, integer range from -1 to 4
    elif reward_type == 1: #'session_length':
        reward=h_response_t[1]
    elif reward_type == 2: #'mix':
        reward=0.7*(h_response_t[0])+0.3*(h_response_t[1])
    return reward

def response_to_lable(h_response, reward_type):
    if reward_type == 1: #'session_length':
        return h_response[:,1].long()
    else:
        return h_response[:,0].long()

mean=torch.load('./PrefRec/state_stats/state_mean.pt').to(torch.float32)
std=torch.load('./PrefRec/state_stats/state_std.pt').to(torch.float32)
h_mean=torch.load('./PrefRec/state_stats/h_state_mean.pt').to(torch.float32)
h_std=torch.load('./PrefRec/state_stats/h_state_std.pt').to(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean=torch.cat([mean,h_mean]).to(device)
std=torch.cat([std,h_std]).to(device)
h_mean=None
h_std=None

def norm_state(state):
    return (state - mean)/std
