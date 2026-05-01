
from collections import defaultdict
from cyberwheel.network.network_generation.random_network_generator import generate_random_networks
from cyberwheel.utils.rl_policy import RLPolicyTabular, RLPolicyParameterized, RLPolicyActorCritic
import gymnasium as gym
import time
import importlib
import pandas as pd
import torch
# import wandb
import os
import random
import yaml

from importlib.resources import files
from pathlib import Path
from tqdm import tqdm

from cyberwheel.network.network_base import Network
from cyberwheel.utils import RLPolicyActorCritic
from cyberwheel.utils.get_service_map import get_service_map
from cyberwheel.runners.visualizer import Visualizer
from cyberwheel.runners.rl_trainer import RLTrainer
from cyberwheel.utils.set_seed import set_seed


class RLEvaluator(RLTrainer):
    def __init__(self, args):
        super().__init__(args)
        # if self.args.download_model:
        #     self.api = wandb.Api()
        #     self.run = self.api.run(
        #         f"{self.args.wandb_entity}/{self.args.wandb_project_name}/runs/{self.args.run}"
        #     )

    def configure_evaluation(self):
        if self.args.deterministic:
            set_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        else:
            seed = random.randint(0, 999999999)
            set_seed(seed)
            torch.backends.cudnn.deterministic = False

        self.device = torch.device(self.args.device)
        print(f"Using device {self.device}")

        # Load networks from yaml here
        network_configs = []
        if self.args.network_config is None:
            if self.args.policy_type == "tabular":
                t = "table"
            else: 
                t = ""

            network_configs = generate_random_networks(n_networks=1, output_path="cyberwheel/data/configs/network", num_subnets=self.args.num_subnets, num_hosts=self.args.num_hosts, seed=self.args.seed if self.args.deterministic else None, t=t)
        elif isinstance(self.args.network_config, str):
            network_configs.append(self.args.network_config)
        else:
            for config in self.args.network_config:
                network_configs.append(config)
        
        self.networks = {}
        print("Network configs to evaluate:", network_configs)
        self.args.service_mapping = {}
        for config in network_configs:
            if config.startswith("/") or config.startswith("cyberwheel"):
                network_config = config
            else:
                network_config = files("cyberwheel.data.configs.network").joinpath(config)

            print(f"Building network: {config} ...")

            network = Network.create_network_from_yaml(network_config)
            network_name = network.name
            self.networks[network_name] = [network]

            print("Mapping attack validity to hosts...", end=" ")
            self.args.service_mapping[network_name] = get_service_map(network)
            print("done")
        
        self.args.agent_config = {}
        print("Network:",self.networks)

        for agent_type in self.args.agents:
            self.args.agent_config[agent_type] = {}
            agent_yaml = self.args.agents[agent_type]
            print(f"Loading {agent_type} agent config from {agent_yaml}...")
            agent_config = files(f"cyberwheel.data.configs.{agent_type}_agent").joinpath(agent_yaml) # get agent yaml path (dir + yaml filename)
            with open(agent_config, "r") as yaml_file: # load the yaml file
                self.args.agent_config[agent_type] = yaml.safe_load(yaml_file)
            if self.args.agent_config[agent_type]["rl"]: # if config says this is an RL agent, include for evalutation
                self.agents[agent_type] = None
        print("Agents to evaluate:", self.agents.keys())
        self.env = self.make_env(0, evaluation=True, net_name=list(self.networks.keys())[0])()
        self.policy = {}

    def load_models(self):
        for agent in self.agents:
            agent_filename = f"{agent}_{self.args.checkpoint}.pt"
            # If download from W&B, use API to get run data.
            if self.args.download_model:
                model = self.run.file(agent_filename)
                model.download(
                    files("cyberwheel.data.models").joinpath(self.args.experiment_name), exist_ok=True
                )
            if self.args.nrec:
                load_path = Path("/persistent01/cyberwheel/models") / self.args.experiment_name
            elif self.args.drive:
                load_path = files("content.drive.MyDrive.RLCS.models").joinpath(self.args.experiment_name)
            else:
                load_path = files("cyberwheel.data.models").joinpath(self.args.experiment_name)
            if self.args.policy_type == "tabular":
                self.policy[agent] = RLPolicyTabular(self.agents[agent]["max_action_space_size"], self.agents[agent]["obs"].shape, self.args)
                save_dict = torch.load(
                                load_path.joinpath(agent_filename),
                                map_location=self.device,
                                weights_only=False,
                            )
                # Restore Q-table
                q_table_dict = save_dict['q_table']
                self.policy[agent].q_table = defaultdict(
                    lambda: torch.zeros(self.agents[agent]["policy"].action_space_shape)
                )
                self.policy[agent].q_table.update(q_table_dict)
                
            elif self.args.policy_type == "parameterized":
                self.policy[agent] = RLPolicyParameterized(self.agents[agent]["max_action_space_size"], self.agents[agent]["obs"].shape, eval=True).to(self.device)
                state_dict = torch.load(
                    load_path.joinpath(agent_filename),
                    map_location=self.device,
                )
                # Eval mode policies may not instantiate target_model; drop those keys if present.
                if any(k.startswith("target_model.") for k in state_dict.keys()):
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("target_model.")}
                self.policy[agent].load_state_dict(state_dict, strict=False)
                self.policy[agent].eval()
            else:
                self.policy[agent] = RLPolicyActorCritic(self.agents[agent]["max_action_space_size"], self.agents[agent]["obs"].shape).to(self.device)            
                self.policy[agent].load_state_dict(
                    torch.load(
                        load_path.joinpath(agent_filename),
                        map_location=self.device,
                    )
                )
                self.policy[agent].eval()
        print("Models loaded for agents:", self.policy.keys())

    def _initialize_environment(self):
        print("Resetting the environment...")

        self.episode_rewards = []
        self.total_reward = 0
        self.steps = 0
        self.obs = self.env.reset()

        print("Playing environment...")

        # Set up dirpath to store action logs CSV
        if self.args.graph_name:
            self.now_str = self.args.graph_name
        else:
            self.now_str = f"{self.args.experiment_name}_evaluate_{self.args.network_config.split('.')[0]}_{self.args.red_agent}_{self.args.reward_function}reward"
        self.log_file = files("cyberwheel.data.action_logs").joinpath(f"{self.now_str}.csv")

        self.actions_df = pd.DataFrame()
        # data = {
        #         "episode": [],
        #         "step": [],
        # }
        self.action_mask = {}
        self.rewards = {}
        
        # for agent in self.agents:
        #     data[agent] = {}
            # data[agent]["action_name"] = []
            # data[agent]["action_src"] = []
            # data[agent]["action_dest"] = []
            # data[agent]["action_success"] = []
            # data[agent]["reward"] = []

        with open(self.log_file, 'w'): # Create an empty CSV for new action logs, overwrite previous
            pass
    
    def mask_actions(self, new_action_mask, action_mask):
        new_mask = torch.tensor(
            new_action_mask,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return new_mask

    def evaluate(self):
        print("Starting evaluation for agents:", self.agents.keys())
        for agent in self.agents:
            self.action_mask[agent] = torch.zeros(self.agents[agent]["max_action_space_size"], dtype=torch.bool).to(self.device)
            self.rewards[agent] = [0] * self.args.num_episodes

        # Get valid targets that will give rewards when impacted
        valid_targets = self.env.reward_calculator.get_valid_targets()
        print("\n=== Valid Target Hosts (will give reward when impacted) ===")
        print(f"Total valid targets: {len(valid_targets)}")
        network = self.env.network
        for host_name in sorted(valid_targets):
            if host_name in network.hosts:
                host = network.hosts[host_name]
                host_type = host.host_type.name if hasattr(host, 'host_type') and hasattr(host.host_type, 'name') else "Unknown"
                is_decoy = " [DECOY]" if host.decoy else ""
                print(f"  - {host_name} ({host_type}){is_decoy}")
            else:
                # Might be a decoy or other target not in regular hosts
                print(f"  - {host_name} (not in network.hosts)")
        print()
        
        # Track impacted hosts across all episodes
        self.impacted_hosts = set()
        
        self.start_time = time.time()
        for episode in range(self.args.num_episodes):
            obs, _ = self.env.reset()
            for step in range(self.args.num_steps):
                action = None
                actions = {}
                action_masks = self.env.action_mask

                for agent in self.agents:
                    agent_obs = torch.Tensor(obs[agent]).to(self.device)
                    tmp_mask = action_masks[agent]
                    self.action_mask[agent] = self.mask_actions(tmp_mask, self.action_mask[agent])
                    action, _, _, _ = self.policy[agent].get_action_and_value(agent_obs, action_mask=self.action_mask[agent])
                    actions[agent] = action

                obs, rew, done, _, info = self.env.step(actions)

                # Track which hosts are impacted
                if "host_info" in info:
                    for host_name, host_view in info["host_info"].items():
                        if host_view.get("impacted", False):
                            self.impacted_hosts.add(host_name)

                actions_df = {
                    "episode": episode,
                    "step": step,
                    "reward": rew,
                }
                self.total_reward += rew
                for agent in self.agents:
                    actions_df[f"{agent}_action_name"] = [info[f"{agent}_action"]]
                    actions_df[f"{agent}_action_success"] = [info[f"{agent}_action_success"]]
                    actions_df[f"{agent}_action_src"] = [info[f"{agent}_action_src"]]
                    actions_df[f"{agent}_action_dest"] = [info[f"{agent}_action_dst"]]
                    actions_df[f"{agent}_reward"] = [info[f"{agent}_reward"]]

                actions_df = pd.DataFrame(actions_df)
                actions_df.to_csv(self.log_file, mode='a', header = os.path.getsize(self.log_file) == 0, index=False)
                # save graph
                if self.args.visualize:
                    self.visualizer = Visualizer(
                        network=list(self.networks.values())[0][0],
                        experiment_name=self.args.experiment_name,
                    )
                    self.visualizer.visualize(episode, step, info)
                    # visualize(net, episode, step, now_str, history, killchain)
                #return # TODO
        
        print("\n=== Evaluation Complete ===")
        print(f"Total valid target hosts impacted: {len(self.impacted_hosts & valid_targets)}/{len(valid_targets)}")
        print(f"Total other hosts impacted: {len(self.impacted_hosts - valid_targets)}")
        
        network = self.env.network
        if self.impacted_hosts & valid_targets:
            print("\nValid target hosts that were impacted:")
            for host_name in sorted(self.impacted_hosts & valid_targets):
                if host_name in network.hosts:
                    host = network.hosts[host_name]
                    host_type = host.host_type.name if hasattr(host, 'host_type') and hasattr(host.host_type, 'name') else "Unknown"
                    is_decoy = " [DECOY]" if host.decoy else ""
                    print(f"  - {host_name} ({host_type}){is_decoy}")
                else:
                    print(f"  - {host_name} (not in network.hosts)")
        
        not_impacted = valid_targets - self.impacted_hosts
        if not_impacted:
            print("\nValid target hosts NOT impacted:")
            for host_name in sorted(not_impacted):
                if host_name in network.hosts:
                    host = network.hosts[host_name]
                    host_type = host.host_type.name if hasattr(host, 'host_type') and hasattr(host.host_type, 'name') else "Unknown"
                    is_decoy = " [DECOY]" if host.decoy else ""
                    print(f"  - {host_name} ({host_type}){is_decoy}")
                else:
                    print(f"  - {host_name} (not in network.hosts)")
        
        other_impacted = self.impacted_hosts - valid_targets
        if other_impacted:
            print("\nOther hosts impacted (no reward):")
            for host_name in sorted(other_impacted):
                if host_name in network.hosts:
                    host = network.hosts[host_name]
                    host_type = host.host_type.name if hasattr(host, 'host_type') and hasattr(host.host_type, 'name') else "Unknown"
                    print(f"  - {host_name} ({host_type})")
                else:
                    print(f"  - {host_name} (not in network.hosts)")

"""
    def evaluate(self):
        
        for episode in tqdm(range(self.args.num_episodes)):
            obs, _ = self.env.reset()
            for step in range(self.args.num_steps):
                self.blue_obs = torch.Tensor(obs["blue"]).to(self.device)
                self.red_obs = torch.Tensor(obs["red"]).to(self.device)

                tmp_blue_mask = self.env.blue_action_mask
                tmp_red_mask = self.env.red_action_mask

                self.blue_action_mask = self.mask_actions(tmp_blue_mask, self.blue_action_mask)
                self.red_action_mask = self.mask_actions(tmp_red_mask, self.red_action_mask)

                blue_action, _, _, _ = self.blue_agent.get_action_and_value(
                    self.blue_obs, action_mask=self.blue_action_mask
                )
                red_action, _, _, _ = self.red_agent.get_action_and_value(
                    self.red_obs, action_mask=self.red_action_mask
                )

                action = {"blue": blue_action, "red": red_action}

                obs, rew, done, _, info = self.env.step(action)

                blue_reward = info["blue_reward"]
                red_reward = info["red_reward"]

                blue_action = info["blue_action"]
                red_action_type = info["red_action"]
                red_action_src = info["red_action_src"]
                red_action_dest = info["red_action_dst"]
                red_action_success = info["red_action_success"]
                blue_action_success = info["blue_action_success"]
                self.red_obs = obs["red"]
                self.blue_obs = obs["blue"]

                actions_df = pd.DataFrame(
                {
                    "episode": [episode],
                    "step": [step],
                    "red_action_success": [red_action_success],
                    "red_action_type": [red_action_type],
                    "red_action_src": [red_action_src],
                    "red_action_dest": [red_action_dest],
                    "blue_action": [blue_action],
                    "blue_success": [blue_action_success],
                    "blue_reward": [blue_reward],
                    "red_reward": [red_reward],
                    "reward": [rew],
                })
                actions_df.to_csv(self.log_file, mode='a', header = os.path.getsize(self.log_file) == 0, index=False)

                # If generating graphs for dash server view
                if self.args.visualize:
                    # visualize(net, episode, step, now_str, history, killchain)
                    pass

                self.total_reward += rew
                self.steps += 1

            self.steps = 0
            self.episode_rewards.append(self.total_reward)
            self.total_reward = 0

        # Save action metadata to CSV in action_logs/

        self.total_time = time.time() - self.start_time
        print("charts/SPS", int(2000 / self.total_time))
        #self.total_reward = sum(self.episode_rewards)
        #self.episodes = len(self.episode_rewards)
        print(f"Total Time Elapsed: {self.total_time}")
"""