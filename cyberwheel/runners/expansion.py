from collections import defaultdict

from cyberwheel.runners.rl_table_handler import RLTableHandler
from importlib.resources import files
from cyberwheel.utils import parse_override_args, YAMLConfig
import os
import numpy as np
import time
from cyberwheel.runners.rl_trainer import RLTrainer

def train_expanded_agents(args: YAMLConfig):
    if args.debug_mode:
        args.num_envs = 1
        args.track = False
        args.device = 'cpu'
        args.async_env = False
        args.experiment_name = 'DEBUG_' + args.experiment_name
    args.batch_size = int(args.num_envs * args.num_steps)   # Number of environment steps to performa backprop with
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # Number of environments steps to perform backprop with in each epoch
    args.num_updates = args.total_timesteps // args.batch_size  # Total number of policy update phases
    args.save_frequency = int(args.num_updates / args.num_saves)    # Number of policy updates between each model save and evaluation
    if args.save_frequency == 0:
        args.save_frequency = 1

    # retrieve abstract agent for policy type
    # parameterized-RedAgentvsRLBlueAgent
    # tabular-RedAgentvsRLBlueAgent
    if args.policy_type not in ["parameterized", "tabular"]:
        raise ValueError(f"Invalid policy type {args.policy_type}.")

    max_net = args.network_size_compatibility
    if args.policy_type == "tabular" and hasattr(args, "num_hosts"):
        args.max_num_hosts = getattr(args, "num_hosts")
    else:
        args.max_num_hosts = 100 if max_net == 'small' else 1000 if max_net == 'medium' else 10000 # if max_net == 'large'
        
    if args.policy_type == "tabular" and hasattr(args, "num_subnets"):
        args.max_num_subnets = getattr(args, "num_subnets")
    else:
        args.max_num_subnets = 10 if max_net == 'small' else 100 if max_net == 'medium' else 1000 #if max_net == 'large'

    # Unique experiment name if empty
    if not args.experiment_name:
        # args.experiment_name = f"{os.path.basename(__file__).rstrip('.py')}_{args.seed}_{int(time.time())}"
        args.experiment_name = f"train-{args.policy_type}-{args.max_num_hosts}-{args.seed}"

    args.agents["red"] = "rl_red_agent.yaml"
    abstract_trainer = RLTrainer(args)
    abstract_trainer.configure_training()

    print()
    print("*** Abstract agent ***")
    abstract_trainer.handler.load_models()

    abstract_policy = abstract_trainer.handler.agents["red"]["policy"]

    print()
    print("*** Expanding agent ****")
    args.seed = args.seed + 1  # change seed to get different envs for expanded agent training
    args.experiment_name = f"train_expansion-{args.policy_type}-{args.max_num_hosts}-{args.seed}-{args.method}{ '-reuse' if getattr(args, 'reuse_model', True) else '' }-ExpandedRedAgentvsRLBlueAgent"
    args.agents["red"] = "rl_red_complex.yaml"
    expanded_trainer = RLTrainer(args)
    expanded_trainer.configure_training()  
    expanded_trainer.handler.get_action_mapping(files("cyberwheel.data.configs.red_agent").joinpath("rl_red_complex.yaml"))
    if args.method not in ["copy_params", "increase_depth", "kl_divergence", "softmax", "copy_values"]:
        raise ValueError(f"Invalid expansion method {args.method}.")
    expanded_trainer.handler.expand_model(abstract_policy, args, writer=expanded_trainer.writer) 
    
    print()
    print("*** Training expanded agent ****")

    for update in range(1, args.num_updates + 1):
        # update envs each training step if leader and entry host are random (initial method to test)
        red_agent = expanded_trainer.handler.envs.envs[0].red_agent
        if red_agent.leader == "random" and red_agent.entry_host == "random":
            expanded_trainer.handler.envs = expanded_trainer.get_envs()  # reinitialize envs (entry host and leader will be reselected)
        expanded_trainer.train(update)

    abstract_trainer.close()
    # expanded_probs_trainer.close()
    expanded_trainer.close()