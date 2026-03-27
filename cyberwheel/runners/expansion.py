from collections import defaultdict

from cyberwheel.runners.rl_table_handler import RLTableHandler
from importlib.resources import files
from cyberwheel.utils import parse_override_args, YAMLConfig
import os
import numpy as np
import time
from cyberwheel.runners.rl_trainer import RLTrainer

def train_table_agents(args: YAMLConfig):
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


    args.experiment_name = "TableRLRedAgentvsRLBlueAgent"
    args.agents["red"] = "rl_red_agent.yaml"
    abstract_trainer = RLTrainer(args)
    abstract_trainer.configure_training()

    print("*** Abstract agent ***")

    if args.load:
        # can use model to train
        print("Loading abstract agent...")
        abstract_trainer.handler.load_models()
    else:
        # need to train a model
        print("Training abstract agent...")
        for update in range(1, args.num_updates + 1):
            # update envs each training step if leader and entry host are random (initial method to test)
            red_agent = abstract_trainer.handler.envs.envs[0].red_agent
            if red_agent.leader == "random" and red_agent.entry_host == "random":
                abstract_trainer.handler.envs = abstract_trainer.get_envs()  # reinitialize envs (entry host and leader will be reselected)
            abstract_trainer.train(update)

    abstract_policy = abstract_trainer.handler.agents["red"]["policy"]
    abstract_num_states = len(abstract_policy.q_table)
    print(f"Abstract policy states before expansion: {abstract_num_states}")
    if abstract_num_states == 0:
        raise ValueError(
            "Abstract policy has 0 states before expansion. "
            "Check that the correct experiment_name is loaded and that red_agent.pt contains a populated q_table."
        )

    # args.experiment_name = "TableRLExpandedProbsRedAgentvsRLBlueAgent"
    # args.agents["red"] = "rl_red_complex.yaml"
    # expanded_probs_trainer = RLTrainer(args)
    # expanded_probs_trainer.configure_training()  
    # expanded_probs_trainer.handler.get_action_mapping(files("cyberwheel.data.configs.red_agent").joinpath("rl_red_complex.yaml"))
    # expanded_probs_trainer.handler.agents["red"]["policy"].q_table = expanded_probs_trainer.handler.expand_model(abstract_policy, "probabilities")
    # print(f"Expanded-probabilities policy states: {len(expanded_probs_trainer.handler.agents['red']['policy'].q_table)}")
    # expanded_probs_trainer.handler.initial_epsilon = 1.0
    # print("Model expanded. Starting training of complex agent...")
    
    # print("*** Expanded Probabilities agent ****")

    # for update in range(1, args.num_updates + 1):
    #     # update envs each training step if leader and entry host are random (initial method to test)
    #     red_agent = expanded_probs_trainer.handler.envs.envs[0].red_agent
    #     if red_agent.leader == "random" and red_agent.entry_host == "random":
    #         expanded_probs_trainer.handler.envs = expanded_probs_trainer.get_envs()  # reinitialize envs (entry host and leader will be reselected)
    #     expanded_probs_trainer.train(update)

    args.experiment_name = "TableRLExpandedQvaluesRedAgentvsRLBlueAgent"
    args.agents["red"] = "rl_red_complex.yaml"
    expanded_copy_trainer = RLTrainer(args)
    expanded_copy_trainer.configure_training()  
    expanded_copy_trainer.handler.get_action_mapping(files("cyberwheel.data.configs.red_agent").joinpath("rl_red_complex.yaml"))
    expanded_copy_trainer.handler.agents["red"]["policy"].q_table = expanded_copy_trainer.handler.expand_model(abstract_policy, "q_values")
    print(f"Expanded-q-values policy states: {len(expanded_copy_trainer.handler.agents['red']['policy'].q_table)}")
    # expanded_copy_trainer.handler.initial_epsilon = 1.0
    print("Model expanded. Starting training of complex agent...")

    print("*** Expanded Q-Values agent ****")

    for update in range(1, args.num_updates + 1):
        # update envs each training step if leader and entry host are random (initial method to test)
        red_agent = expanded_copy_trainer.handler.envs.envs[0].red_agent
        if red_agent.leader == "random" and red_agent.entry_host == "random":
            expanded_copy_trainer.handler.envs = expanded_copy_trainer.get_envs()  # reinitialize envs (entry host and leader will be reselected)
        expanded_copy_trainer.train(update)

    abstract_trainer.close()
    # expanded_probs_trainer.close()
    expanded_copy_trainer.close()
