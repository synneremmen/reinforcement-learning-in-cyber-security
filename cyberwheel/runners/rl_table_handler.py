from collections import defaultdict
from torch import nn, optim

from cyberwheel.utils import RLPolicyTableBased
from gymnasium.vector import VectorEnv, AsyncVectorEnv

from importlib.resources import files

import numpy as np
import torch
import os

class RLTableHandler:

    def __init__(self, envs: VectorEnv, args, agents: dict, static_agents=[], load=False):
        self.envs = envs
        self.args = args
        self.device = self.args.device
        print(f"Using device '{self.device}'")

        self.agents = {}
        self.static_agents = static_agents
        self.episode = 1
        
        for agent in agents:
            self.agents[agent] = agents[agent]
            self.agents[agent]["shape"] = self.agents[agent]["obs"].shape
            self.agents[agent]["policy"] = RLPolicyTableBased(self.agents[agent]["max_action_space_size"], self.agents[agent]["shape"]).to(self.device)

        if load:
            self.load_models(files("cyberwheel.data.models").joinpath(self.args.experiment_name))

    def define_multiagent_variables(self):
        """Initialize episode tracking variables"""
        reset = self.envs.reset(seed=[i for i in range(self.args.num_envs)])[0]

        for agent in self.agents:
            agent_dict = self.agents[agent]
            self.agents[agent]["obs"] = torch.zeros((self.args.num_steps, self.args.num_envs) + agent_dict["obs"].shape).to(self.device)
            self.agents[agent]["actions"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["action_masks"] = torch.zeros((self.args.num_steps, self.args.num_envs, agent_dict["max_action_space_size"]), dtype=torch.bool).to(self.device)
            self.agents[agent]["resets"] = np.array(reset[agent])
            self.agents[agent]["next_obs"] = torch.Tensor(self.agents[agent]["resets"]).to(self.device)
            self.agents[agent]["rewards"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["episode_rewards"] = torch.zeros(self.args.num_envs).to(self.device)
            self.agents[agent]["episode_lengths"] = torch.zeros(self.args.num_envs).to(self.device)
            self.agents[agent]["dones"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["values"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)
        self.global_step = 0

    def mask_actions(self, new_action_mask, action_mask):
        """Convert action mask to tensor"""
        new_mask = torch.tensor(
            new_action_mask,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return new_mask
    
    def update_action_masks(self, step: int):
        """Get current action masks from all environments"""
        masks = self.envs.call("action_mask") if self.args.async_env else [env.unwrapped.action_mask for env in self.envs.envs]
        for agent in self.agents:
            for i in range(self.args.num_envs):
                self.agents[agent]["action_masks"][step][i] = self.mask_actions(masks[i][agent], self.agents[agent]["action_masks"][step][i])


    def step_multiagent(self, step: int):
        """Execute one step in all environments"""
        self.global_step += self.args.num_envs
        self.dones[step] = self.next_done
        policy_action = {}

        for agent in self.agents:
            self.agents[agent]["obs"][step] = self.agents[agent]["next_obs"]
            for env_idx in range(self.args.num_envs):
                obs = self.agents[agent]["obs"][step][env_idx].cpu().numpy()
                action_mask = self.agents[agent]["action_masks"][step][env_idx]
                # self.agents[agent]["values"][step][env_idx] = self.agents[agent]["policy"].get_value(obs, action_mask=action_mask).detach()
                action = self.agents[agent]["policy"].select_action(
                    obs, action_mask
                )
                # value = self.agents[agent]["policy"].get_value(obs, action=action)
                # self.agents[agent]["values"][step][env_idx] = value
                self.agents[agent]["actions"][step][env_idx] = action
            
            policy_action[agent] = self.agents[agent]["actions"][step].cpu().numpy()
            # self.[step] = self.next_done 

        obs, reward, done, _, info = self.envs.step(policy_action)

        for agent in self.agents:
            for env_idx in range(self.args.num_envs):
                reward_val = self.agents[agent]["rewards"][step][env_idx].item()
                self.agents[agent]["episode_rewards"][env_idx] += reward_val
                self.agents[agent]["episode_lengths"][env_idx] += 1
                self.agents[agent]["next_obs"][env_idx] = torch.Tensor(obs[agent][env_idx]).to(self.device)
                if f"{agent}_reward" in info:
                    self.agents[agent]["rewards"][step][env_idx] = torch.tensor(info[f"{agent}_reward"][env_idx]).to(self.device)

        self.next_done = torch.Tensor(done).to(self.device)

    def log_stuff(self, writer, episodic_runtime, episodic_processing_time):
        """Log training metrics"""
        output_str = f"global_step={self.global_step}"
        
        for agent in self.agents:
            mean_rew = self.agents[agent]["rewards"].sum(axis=0).mean()
            output_str += f", {agent}_episodic_return={mean_rew}"
            writer.add_scalar(f"charts/{agent}_episodic_return", mean_rew, self.global_step)
                
            q_table = self.agents[agent]["policy"].q_table
            num_states = len(q_table)
            if num_states > 0:
                all_q_values = torch.stack([v for v in q_table.values()])
                mean_q = all_q_values.mean().item()
                max_q = all_q_values.max().item()
                writer.add_scalar(f"charts/{agent}_mean_q_value", mean_q, self.global_step)
                writer.add_scalar(f"charts/{agent}_max_q_value", max_q, self.global_step)
                writer.add_scalar(f"charts/{agent}_num_states_visited", num_states, self.global_step)
                
        print(output_str)
        writer.add_scalar(f"charts/episodic_runtime", episodic_runtime, self.global_step)
        writer.add_scalar(f"charts/episodic_process_time", episodic_processing_time, self.global_step)

    def flatten_batch(self):
        pass

    def update_policy(self, mb_inds = None):
        """Update Q-tables using collected transitions (offline learning)"""
    
        for agent in self.agents:
            for step in range(self.args.num_steps):
                for env_idx in range(self.args.num_envs):

                    # Update Q-table
                    self.agents[agent]["policy"].update_q_table(
                        obs=self.agents[agent]["obs"][step][env_idx].cpu().numpy(),
                        action=self.agents[agent]["actions"][step][env_idx].cpu().numpy(),
                        reward=self.agents[agent]["rewards"][step][env_idx].cpu().numpy(),
                        next_obs=self.agents[agent]["next_obs"][env_idx].cpu().numpy(),
                        done=self.dones[step][env_idx].cpu().numpy(),
                        next_action_mask=self.agents[agent]["action_masks"][step + 1][env_idx].cpu().numpy() if step < self.args.num_steps - 1 else None,
                        alpha=self.args.learning_rate,
                        gamma=self.args.gamma
                    )
        self.episode += 1
        self.decay_epsilon(self.episode)

    def calculate_explained_variance(self):
        pass

    def save_models(self):
        run_path = files("cyberwheel.data.models").joinpath(self.args.experiment_name)
        agent_paths = {}
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        for agent in self.agents:
            agent_path = run_path.joinpath(f"{agent}_agent.pt")
            globalstep_path = run_path.joinpath(f"{agent}_{self.global_step}.pt")
            
            q_table_dict = dict(self.agents[agent]["policy"].q_table)
            save_dict = {
                'q_table': q_table_dict,
                'epsilon': self.agents[agent]["policy"].epsilon,
                'global_step': self.global_step,
                'num_states': len(q_table_dict),
            }
            
            torch.save(save_dict, agent_path)
            torch.save(save_dict, globalstep_path)
            print(f"Saved {agent} to {agent_path}")
            
            if self.args.track:
                import wandb
                wandb.save(str(agent_path), base_path=str(run_path), policy="now")
                wandb.save(str(globalstep_path), base_path=str(run_path), policy="now")
                
            agent_paths[agent] = agent_path
        return agent_paths

    def load_models(self, path):
        """Load Q-tables from disk"""
        for agent in self.agents:
            agent_path = path.joinpath(f"{agent}_agent.pt")
            if os.path.exists(agent_path):
                save_dict = torch.load(agent_path)
                
                # Restore Q-table
                q_table_dict = save_dict['q_table']
                self.agents[agent]["policy"].q_table = defaultdict(
                    lambda: torch.zeros(self.agents[agent]["policy"].action_space_shape)
                )
                self.agents[agent]["policy"].q_table.update(q_table_dict)
                
                # Restore epsilon if saved
                if 'epsilon' in save_dict:
                    self.agents[agent]["policy"].epsilon = save_dict['epsilon']
                    
    def log_training_metrics(self, writer):
        for agent in self.agents:
            q_table = self.agents[agent]["policy"].q_table
            num_states = len(q_table)
            if num_states > 0:
                all_q_values = torch.stack([v for v in q_table.values()])
                mean_q = all_q_values.mean().item()
                max_q = all_q_values.max().item()
                writer.add_scalar(f"charts/{agent}_mean_q_value", mean_q, self.global_step)
                writer.add_scalar(f"charts/{agent}_max_q_value", max_q, self.global_step)
                writer.add_scalar(f"charts/{agent}_num_states_visited", num_states, self.global_step)
            

    def reset(self):
        """Reset environment and agent states"""
        for agent in self.agents:
            reset = self.envs.reset()[0]
            self.agents[agent]["resets"] = np.array(reset[agent])
            self.agents[agent]["next_obs"] = torch.Tensor(self.agents[agent]["resets"]).to(self.device)
            self.agents[agent]["episode_rewards"] = torch.zeros(self.args.num_envs).to(self.device)
            self.agents[agent]["episode_lengths"] = torch.zeros(self.args.num_envs).to(self.device)
            self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["obs"] = torch.zeros((self.args.num_steps, self.args.num_envs) + self.agents[agent]["next_obs"].shape[1:]).to(self.device)
            self.agents[agent]["rewards"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["action_masks"] = torch.zeros((self.args.num_steps, self.args.num_envs, self.agents[agent]["max_action_space_size"]), dtype=torch.bool).to(self.device)

    def decay_epsilon(self, episode: int):
        """Decay epsilon for epsilon-greedy exploration"""
        for agent in self.agents:
            if hasattr(self.agents[agent]["policy"], 'epsilon'):
                # Linear decay
                initial_epsilon = getattr(self.args, 'epsilon', 1.0)
                final_epsilon = getattr(self.args, 'final_epsilon', 0.01)
                decay_episodes = getattr(self.args, 'epsilon_decay_episodes', self.args.total_timesteps / 10) 
                epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * min(episode / decay_episodes, 1.0)
                self.agents[agent]["policy"].epsilon = epsilon



# path = "/Users/synneandreassen/Documents/MasterMaskinlæringCode/INF399/Environments/cyberwheel/cyberwheel/data/models/TableRLRedAgentvsRLBlueAgent/red_agent.pt"

# # Load
# ckpt = torch.load(path, map_location="cpu")

# # Modify epsilon
# ckpt['epsilon'] = 0.5  # Set desired value

# # Save back
# torch.save(ckpt, path)
# print(f"Updated epsilon to {ckpt['epsilon']}")

# python3 /Users/synneandreassen/Documents/MasterMaskinlæringCode/INF399/Environments/cyberwheel/cyberwheel/runners/rl_table_handler.py