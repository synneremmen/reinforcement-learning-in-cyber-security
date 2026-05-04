# CleanRL script for training Cage Challenge 2 agents. CleanRL documentation can be found at https://docs.cleanrl.dev/,
import os
import time
import torch

from cyberwheel.utils import parse_override_args, YAMLConfig
from cyberwheel.runners.rl_trainer import RLTrainer

"""
This script will train cyberwheel. Using the args from the config file passed, it will run an Actor-Critic RL algorithm and run training
with intermittent evaluations/saves. If tracking to W&B, this will be logged in your W&B project for each training run.
"""
# Allows using command line to override args in the YAML config
def train_cyberwheel(args: YAMLConfig):
    if args.debug_mode:
        args.num_envs = 1
        args.track = False
        args.device = 'cpu'
        args.async_env = False
        args.experiment_name = 'DEBUG_' + args.experiment_name
    args.batch_size = int(args.num_envs * args.num_steps)   # Number of environment steps to performa backprop with
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # Number of environments steps to perform backprop with in each epoch
    args.num_updates = args.total_timesteps // args.batch_size  # Total number of policy update phases

    if args.num_saves > 0:
        args.save_frequency = int(args.num_updates / args.num_saves)    # Number of policy updates between each model save and evaluation
    else:
        args.save_frequency = 1

    max_net = args.network_size_compatibility
    if hasattr(args, "num_hosts"):
        args.max_num_hosts = getattr(args, "num_hosts")
    else:
        args.max_num_hosts = 100 if max_net == 'small' else 1000 if max_net == 'medium' else 10000 # if max_net == 'large'
        
    if hasattr(args, "num_subnets"):
        args.max_num_subnets = getattr(args, "num_subnets")
    else:
        args.max_num_subnets = 10 if max_net == 'small' else 100 if max_net == 'medium' else 1000 #if max_net == 'large'

    # Unique experiment name if empty
    if not args.experiment_name:
        # args.experiment_name = f"{os.path.basename(__file__).rstrip('.py')}_{args.seed}_{int(time.time())}"
        agent = '-' + args.agents['red'].split('.')[0] if args.agents['red'] else ''
        args.experiment_name = f"train-{args.policy_type}{agent}-{args.max_num_hosts}-{args.seed}"

    args.evaluation = False

    # Initialize the Trainer object
    trainer = RLTrainer(args)

    # Setup W&B if tracking
    if args.track:
        trainer.wandb_setup()

    # Configure training parameters and train
    trainer.configure_training()

    for update in range(1, args.num_updates + 1):
        print(f"----- Update {update} -----")
        # update envs each training step if leader and entry host are random (initial method to test)
        red_agent = trainer.handler.envs.envs[0].red_agent
        if red_agent.leader == "random" and red_agent.entry_host == "random":
            seed = trainer.args.seed + update  # change seed each update to get different networks
            trainer.handler.envs = trainer.get_envs(seed=seed)  # reinitialize envs (entry host and leader will be reselected)
        trainer.train(update)

    trainer.close()
