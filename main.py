import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch
import logging
import random
import yaml
from types import SimpleNamespace as SN
import pprint
from algos import *
from utils import *
from buffer import *
from data_to_buffer import *
from reward_model import *
from evaluation import w_offline_ab

root_path = "./logs_prefrec" # log path
root_path = "./logs"

def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')
    return logger


# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment()
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(root_path, "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    random.seed(_config["seed"])
    np.random.seed(_config["seed"])
    torch.manual_seed(_config["seed"])
    # run the framework
    run(_run, _config, _log)


def parse_config_file(params):
    config_file = "ddpg"
    for i, v in enumerate(params):
        if v.split("=")[0] == "--config":
            config_file = v.split("=")[1]
            del params[i]
            break
    return config_file

def Create_Policy(args):
    if args.algo=="ddpg":
        return DDPG.DDPG(args)
    if args.algo=="td3":
        return TD3.TD3(args)
    if args.algo=="td3_bc":
        return TD3_BC.TD3_BC(args)
    if args.algo=="il":
        return IL.IL(args)
    if args.algo=="iql":
        return IQL.IQL(args)
    if args.algo=="bcq":
        return BCQ.BCQ(args)
    if args.algo=="sac":
        return SAC.SAC(args)
    if args.algo=="prefrec":
        return PrefRec.PrefRec(args)

def Create_Reward(args):
    return RewardModel(args)

def run(_run, _config, _log):
    args = SN(**_config)
    args.ex_results_path = os.path.join(args.ex_results_path, str(_run._id))
    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                    indent=4,
                                    width=1)
    _log.info("\n\n" + experiment_params + "\n")

    if args.use_tensorboard:
        logger.setup_tb(args.ex_results_path)

    # sacred is on by default
    logger.setup_sacred(_run)

    start_time = time.time()
    last_time = start_time
    
    #last_train_step=0
    last_test_step=-args.test_every-1
    last_save_step=0
    last_log_step=0

    policy=Create_Policy(args)
    reward_model=Create_Reward(args)
    policy.reward_model = reward_model

    if args.save_model:
        test_folder="" # the path to the test files
        best_return=-1000000.
    n_buffer=16
    buffer_size=1500000
    buffer_train_path="" # the path to the replay buffers 
    pref_buffer_size=args.pref_buffer_size
    pref_buffer_path="" # the path to the preference buffers 
    pref_steps_till_now=0
    steps_till_now=0
    
    logger.console_logger.info("Beginning reward pretraining for {} epochs.".format(args.pretrain_n_epoch))
    for n in range(args.pretrain_n_epoch):
        shuffled_buffer_idx=list(range(n_buffer))
        random.shuffle(shuffled_buffer_idx)
        for i in shuffled_buffer_idx:
            pref_buffer=PrefBuffer(args)
            pref_buffer.load(os.path.join(pref_buffer_path,f"{i}-fold.npz"))
            for e in range(int(pref_buffer_size//args.pref_batch_size)):
                pref_overall_step=pref_steps_till_now+e
                loss, acc=reward_model.train(pref_buffer,args.pref_batch_size) # loss: float; acc: float
                logger.log_stat('reward_model_loss', loss, pref_overall_step)
                logger.log_stat('reward_model_acc', acc, pref_overall_step)
            pref_steps_till_now+=int(pref_buffer_size//args.pref_batch_size)
            logger.console_logger.info("Pretrain loss: {}, acc: {}.".format(loss,acc))

    logger.console_logger.info("Beginning RL training for {} epochs.".format(args.n_epoch))     
    for n in range(args.n_epoch):
        shuffled_buffer_idx=list(range(n_buffer))
        random.shuffle(shuffled_buffer_idx)
        for i in shuffled_buffer_idx:
            replay_buffer=ReplayBuffer(args)
            replay_buffer.load(os.path.join(buffer_train_path,f"{i}-fold.npz"))

            if args.fine_tune_reward:
                pref_buffer=PrefBuffer(args)
                pref_buffer.load(os.path.join(pref_buffer_path,f"{i}-fold.npz"))

            for e in range(int(buffer_size//args.batch_size)):
                assert hasattr(policy, 'reward_model') # soft bind
                loss=policy.train(replay_buffer, args.batch_size)
                overall_step=steps_till_now+e
                for key in loss.keys():
                    logger.log_stat(key, loss[key].item(),overall_step)

                if args.fine_tune_reward:
                    pref_overall_step=pref_steps_till_now+e
                    loss, acc=reward_model.train(pref_buffer,args.pref_batch_size)
                    logger.log_stat('reward_model_loss', loss, pref_overall_step)
                    logger.log_stat('reward_model_acc', acc, pref_overall_step)

                if args.save_model and (overall_step-last_test_step) / args.test_every >= 1.0:
                    returns=[]
                    for test_uers in tqdm(os.listdir(test_folder)[:args.test_n_user]):
                        file_path=os.path.join(test_folder, test_uers)
                        R=w_offline_ab(policy, file_path, args.return_type)
                        if R: # filter out invalid user
                            returns.append(R)
                    returns=np.array(returns).mean().item()
                    logger.console_logger.info(f"Perform evaluation at {overall_step} steps: {returns}.")
                    logger.log_stat("test_return", returns, overall_step)
                    if returns > best_return:
                        best_return=returns
                        save_path = os.path.join(
                            args.ex_results_path, "models/")
                        os.makedirs(save_path, exist_ok=True)
                        policy.save(save_path)
                    last_test_step=overall_step
                if (overall_step-last_log_step) / args.log_every >= 1.0:
                    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_log_step, overall_step, args.n_epoch*n_buffer*int(buffer_size//args.batch_size)), time_str(time.time() - start_time)))
                    last_time = time.time()
                    logger.log_stat("steps", overall_step, overall_step)
                    logger.print_recent_stats()
                    last_log_step=overall_step
            steps_till_now+=int(buffer_size//args.batch_size)
            if args.fine_tune_reward:
                pref_steps_till_now+=int(buffer_size//args.batch_size)  # steps depend on # of RL updatings
            
if __name__ == '__main__':
    params = deepcopy(sys.argv)
    config_file = parse_config_file(params)
    ex.add_config(f'./PrefRec/config/{config_file}.yaml') # path to config file
    logger.info(
        f"Saving to FileStorageObserver in {root_path}/results/{config_file}.")
    file_obs_path = os.path.join(results_path, config_file)
    ex.add_config(name=config_file)
    ex.add_config(ex_results_path=file_obs_path)
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)
