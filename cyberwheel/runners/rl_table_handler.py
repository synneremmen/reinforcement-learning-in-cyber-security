from collections import defaultdict
from torch import nn, optim

from cyberwheel.utils import RLPolicyTableBased
from gymnasium.vector import VectorEnv, AsyncVectorEnv

from importlib.resources import files

import numpy as np
import torch
import os

class RLTableHandler:

    def __init__(self, envs: VectorEnv, args, agents: dict, static_agents=[]):
        self.envs = envs
        self.args = args
        self.device = self.args.device
        print(f"Using device '{self.device}'")

        self.agents = {}
        self.static_agents = static_agents
        self.episode = 1
        self.load = getattr(self.args, 'load', False)
        self.initial_epsilon = getattr(self.args, 'epsilon', 0.2)
        self.do_decay_epsilon = getattr(self.args, 'decay_epsilon', False)

        for agent in agents:
            self.agents[agent] = agents[agent]
            self.agents[agent]["shape"] = self.agents[agent]["obs"].shape
            self.agents[agent]["policy"] = RLPolicyTableBased(self.agents[agent]["max_action_space_size"], self.agents[agent]["shape"], self.initial_epsilon).to(self.device)

        if self.load:
            print(self.args.experiment_name)
            self.load_models()

    def _reset_red_diagnostics(self):
        self.red_action_attempts = defaultdict(int)
        self.red_action_successes = defaultdict(int)
        self.red_reward_valid_targets = 0.0
        self.red_reward_invalid_targets = 0.0
        self.red_valid_target_attempts = 0
        self.red_invalid_target_attempts = 0

    def _phase_bucket(self, action_name, phase_list=None):
        action = str(action_name).lower()
        if phase_list is not None:
            for phase in phase_list[0]:
                if phase in ["discovery", "impact"]:
                    return phase
        return action if action in ["discovery", "impact"] else "other"

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
        self._reset_red_diagnostics()

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
                    self.agents[agent]["rewards"][step][env_idx] = torch.tensor(info[f"{agent}_reward"][env_idx], dtype=torch.float32).to(self.device)

                if agent == "red" and "red_action" in info and "red_action_success" in info and "red_target_valid" in info:
                    action_name = info["red_action"][env_idx]
                    # phase = self._phase_bucket(action_name)
                    kill_chain_phases = info.get("red_kill_chain_phases", [])
                    phase = self._phase_bucket(action_name, kill_chain_phases)
                    
                    success = bool(info["red_action_success"][env_idx])
                    valid_target = bool(info["red_target_valid"][env_idx])
                    red_reward = float(info["red_reward"][env_idx]) if "red_reward" in info else 0.0

                    self.red_action_attempts[phase] += 1
                    if success:
                        self.red_action_successes[phase] += 1

                    if valid_target:
                        self.red_valid_target_attempts += 1
                        self.red_reward_valid_targets += red_reward
                    else:
                        self.red_invalid_target_attempts += 1
                        self.red_reward_invalid_targets += red_reward

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
                
        # print(output_str)
        writer.add_scalar("charts/episodic_runtime", episodic_runtime, self.global_step)
        writer.add_scalar("charts/episodic_process_time", episodic_processing_time, self.global_step)

        total_attempts = self.red_valid_target_attempts + self.red_invalid_target_attempts
        valid_ratio = self.red_valid_target_attempts / total_attempts if total_attempts > 0 else 0.0
        writer.add_scalar("charts/red_valid_target_attempt_ratio", valid_ratio, self.global_step)
        writer.add_scalar("charts/red_reward_valid_targets", self.red_reward_valid_targets, self.global_step)
        writer.add_scalar("charts/red_reward_invalid_targets", self.red_reward_invalid_targets, self.global_step)

        for phase in ["discovery", "impact"]:
            attempts = self.red_action_attempts[phase]
            successes = self.red_action_successes[phase]
            success_rate = successes / attempts if attempts > 0 else 0.0
            writer.add_scalar(f"charts/red_{phase}_attempts", attempts, self.global_step)
            writer.add_scalar(f"charts/red_{phase}_successes", successes, self.global_step)
            writer.add_scalar(f"charts/red_{phase}_success_rate", success_rate, self.global_step)

        self._reset_red_diagnostics()

    def flatten_batch(self):
        pass

    def update_policy(self, step = None):
        """Update Q-tables using collected transitions (offline learning)"""
    
        for agent in self.agents:
            td_updates = []
            for step in range(self.args.num_steps):
                for env_idx in range(self.args.num_envs):

                    # Update Q-table
                    update_value = self.agents[agent]["policy"].update_q_table(
                        obs=self.agents[agent]["obs"][step][env_idx].cpu().numpy(),
                        action=self.agents[agent]["actions"][step][env_idx].cpu().numpy(),
                        reward=self.agents[agent]["rewards"][step][env_idx].cpu().numpy(),
                        next_obs=self.agents[agent]["obs"][step + 1][env_idx].cpu().numpy() if step < self.args.num_steps - 1 else self.agents[agent]["next_obs"][env_idx].cpu().numpy(),
                        done=self.dones[step][env_idx].cpu().numpy(),
                        next_action_mask=self.agents[agent]["action_masks"][step + 1][env_idx].cpu().numpy() if step < self.args.num_steps - 1 else None,
                        alpha=self.args.alpha,
                        gamma=self.args.gamma
                    )
                    td_updates.append(float(update_value))

            if len(td_updates) > 0:
                td_updates_np = np.array(td_updates, dtype=np.float32)
                self.agents[agent]["td_update_mean"] = float(td_updates_np.mean())
                self.agents[agent]["td_update_abs_mean"] = float(np.abs(td_updates_np).mean())
                self.agents[agent]["td_update_max_abs"] = float(np.abs(td_updates_np).max())
            else:
                self.agents[agent]["td_update_mean"] = 0.0
                self.agents[agent]["td_update_abs_mean"] = 0.0
                self.agents[agent]["td_update_max_abs"] = 0.0
        self.episode += 1
        if self.do_decay_epsilon:
            self.decay_epsilon(self.episode)

    def calculate_explained_variance(self):
        pass

    def save_models(self):
        if self.args.drive:
            from pathlib import Path
            drive_dir = Path("/content/drive/MyDrive/RLCSModels")
            drive_dir.mkdir(parents=True, exist_ok=True)
            torch.save(save_dict, drive_dir / "red_agent.pt")
            print("Models saved to Google Drive and download initiated.")
        
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

    def load_models(self, name="red_agent"):
        """Load Q-tables from disk"""
        for agent in self.agents:
            agent_path = files("cyberwheel.data.models").joinpath(self.args.experiment_name).joinpath(f"{name}.pt")
            if os.path.exists(agent_path):
                save_dict = torch.load(agent_path)
                
                # Restore Q-table
                q_table_dict = save_dict['q_table']
                self.agents[agent]["policy"].q_table = defaultdict(
                    lambda: torch.zeros(self.agents[agent]["policy"].action_space_shape)
                )
                self.agents[agent]["policy"].q_table.update(q_table_dict)

                self.agents[agent]["policy"].epsilon = self.initial_epsilon
                
                self.global_step = 0
                self.episode = 1
                self._reset_red_diagnostics()
            
                print(f"Loaded {agent} with fresh history (epsilon={self.initial_epsilon})")
                
                # Restore epsilon if saved
                # if 'epsilon' in save_dict:
                    # self.initial_epsilon = save_dict['epsilon']
                    # self.agents[agent]["policy"].epsilon = save_dict['epsilon']
                    
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
            writer.add_scalar(f"charts/{agent}_epsilon", self.agents[agent]["policy"].epsilon, self.global_step)
            writer.add_scalar(f"charts/{agent}_td_update_mean", self.agents[agent].get("td_update_mean", 0.0), self.global_step)
            writer.add_scalar(f"charts/{agent}_td_update_abs_mean", self.agents[agent].get("td_update_abs_mean", 0.0), self.global_step)
            writer.add_scalar(f"charts/{agent}_td_update_max_abs", self.agents[agent].get("td_update_max_abs", 0.0), self.global_step)
            

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
        self._reset_red_diagnostics()

    def decay_epsilon(self, episode: int):
        """Decay epsilon for epsilon-greedy exploration"""
        for agent in self.agents:
            if hasattr(self.agents[agent]["policy"], 'epsilon'):
                initial_epsilon = getattr(self.args, 'epsilon', 0.2) # self.initial_epsilon if self.initial_epsilon is not None else getattr(self.args, 'initial_epsilon', 0.2)
                final_epsilon = getattr(self.args, 'final_epsilon', 0.01)
                decay_episodes = getattr(self.args, 'epsilon_decay_episodes', self.args.num_updates)  # Decay over all training updates
                epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * min(episode / decay_episodes, 1.0)
                self.agents[agent]["policy"].epsilon = epsilon

    def expand_model(self, old_policy, expansion_type="q_values", num_hosts=6):
        old_action_shape = old_policy.action_space_shape
        old_q_table = old_policy.q_table
        new_action_shape = self.agents["red"]["policy"].action_space_shape
        print(f"Expanding model from action shape {old_action_shape} to {new_action_shape} with {num_hosts} hosts...")
        new_q_table = defaultdict(lambda: torch.zeros(new_action_shape))
        # each action in the old action space corresponds to one or multiple actions in the new action space
        """
        How to choose values of new actions???
        Q(aij) = Q(bi) + Q(aij|bi)
        pi(aij) = pi(bi)pi(aij|bi)
        """
        for state, action_values in old_q_table.items():
            new_q_table[state] = self.get_new_action_values(action_values, expansion_type)

        print("New Q-table initialized with", len(new_q_table), "states and", new_q_table[next(iter(new_q_table))].shape, "actions per state.")
        return new_q_table

    def get_action_mapping(self, path="/cyberwheel/data/configs/red_agent/rl_red_complex.yaml"):
        # mapping of new actions given indicies of old found in yaml file
        with open(path, "r") as f:
            data=f.read()
        phases = {"portscan": 0, "pingsweep": 0, "discovery": 0, "lateral-movement": 0, "privilege-escalation": 0, "impact": 0}
        action_data = data.split("actions:")[1].strip()
        for action in action_data.split("\n\n"):
            phase = action.split("phase:")[1].strip()
            phases[phase] += 1
        self.mapping = phases

    def get_new_action_values(self, action_values, expansion_type):
        """
        Get new action values for expanded action space while keeping probabilities of old actions the same.
        Args:
        - probabilities: tensor of action probabilities for old actions
        - repeats: list of how many new actions each old action corresponds to
        - action_values: tensor of action values for old actions (before expansion)
        Returns:
        - new_action_values: tensor of action values for new actions (after expansion)
        """
        if not self.mapping:
            raise ValueError("Action mapping is empty. Run get_action_mapping() before expanding the model.")
        repeats = list(self.mapping.values())

        if expansion_type == "probabilities":
            probabilities = torch.softmax(action_values, dim=0)    
            new_values = torch.tensor([], dtype=probabilities.dtype, device=probabilities.device)
            nothing_val = probabilities[-1].unsqueeze(0)
            mean = torch.mean(action_values)
            std = torch.std(action_values)
            target_min = torch.min(action_values)
            target_max = torch.max(action_values)
            values = probabilities[:-1]
        elif expansion_type == "q_values":
            new_values = torch.tensor([], dtype=action_values.dtype, device=action_values.device)
            nothing_val = action_values[-1].unsqueeze(0)
            values = action_values[:-1]
        else:
            raise ValueError(f"Invalid expansion_type '{expansion_type}'. Must be 'probabilities' or 'q_values'.")

        for i, val in enumerate(values):
            num_repeat = repeats[i % len(repeats)]
            if expansion_type == "probabilities":
                new_val = val / num_repeat
            else:
                new_val = val
            repeated_vals = new_val.repeat(num_repeat)
            new_values = torch.cat([new_values, repeated_vals])

        new_values = torch.cat([new_values, nothing_val]) # nothing (last action)

        if new_values.numel() != self.agents["red"]["policy"].action_space_shape:
            raise ValueError(
                f"Expanded action vector has length {new_values.numel()}, but expected {self.agents['red']['policy'].action_space_shape}. "
                "Check phase-to-repeat mapping and inclusion of the 'nothing' action."
            )

        if expansion_type == "probabilities":
            new_action_values = self.invert_softmax(new_values, mean, std, target_min, target_max)
            # print(
            #     f"[expand] old_range=({target_min.item():.4f}, {target_max.item():.4f}) -> "
            #     f"new_range=({new_action_values.min().item():.4f}, {new_action_values.max().item():.4f})"
            # )
        else:
            new_action_values = new_values
        # print("New action values:", new_action_values)
        return new_action_values
    
    def invert_softmax(self, probs, mean, std, target_min, target_max):
        log_values = torch.log(probs + 1e-9) 
        log_min = torch.min(log_values)
        log_max = torch.max(log_values)
        log_span = log_max - log_min
        target_span = target_max - target_min

        if log_span > 1e-12 and target_span > 1e-12:
            values = (log_values - log_min) * (target_span / log_span) + target_min
        elif std > 1e-12:
            values = (log_values - log_values.mean()) * (std / (log_values.std() + 1e-12)) + mean
        else:
            values = torch.full_like(log_values, mean)
        return values


# path = "/Users/synneandreassen/Documents/MasterMaskinlæringCode/INF399/Environments/cyberwheel/cyberwheel/data/models/TableRLRedAgentvsRLBlueAgent/red_agent.pt"

# # Load
# ckpt = torch.load(path, map_location="cpu")

# # Modify epsilon
# ckpt['epsilon'] = 0.9  # Set desired value

# # Save back
# torch.save(ckpt, path)
# print(f"Updated epsilon to {ckpt['epsilon']}")

# python3 /Users/synneandreassen/Documents/MasterMaskinlæringCode/INF399/Environments/cyberwheel/cyberwheel/runners/rl_table_handler.py