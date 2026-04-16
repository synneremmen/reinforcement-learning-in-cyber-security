from collections import defaultdict
from pathlib import Path

from torch import optim

from cyberwheel.utils import RLPolicyQlearning
from gymnasium.vector import VectorEnv

from importlib.resources import files

import numpy as np
import torch
import os

class RLQHandler:

    def __init__(self, envs: VectorEnv, args, agents: dict, static_agents=[]):
        self.envs = envs
        self.args = args
        # Use a GPU if available. You can choose a specific GPU with CUDA, for example by setting 'device' to "cuda:0"
        self.device = self.args.device
        print(f"Using device '{self.device}'")

        #self.agents = agents.keys()
        #print(agents)
        self.agents = {}
        self.static_agents = static_agents
        self.episode = 1
        self.load = getattr(self.args, 'load', False)
        if self.args.policy_type not in ["qlearning"]:
            raise ValueError(f"Invalid policy type '{self.args.policy_type}'. Must be either 'actor_critic', 'table_based', or 'qlearning'.")

        for agent in agents:
            self.agents[agent] = agents[agent]
            self.agents[agent]["shape"] = self.agents[agent]["obs"].shape
            self.agents[agent]["policy"] = RLPolicyQlearning(self.agents[agent]["max_action_space_size"], self.agents[agent]["shape"]).to(self.device)
            self.agents[agent]["optimizer"] = optim.Adam([
                { 'params': list(self.agents[agent]["policy"].model.parameters()),  'lr': float(self.args.learning_rate),  'eps': 1e-3 },
            ])

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
        reset = self.envs.reset(seed=[i for i in range(self.args.num_envs)])[0]

        for agent in self.agents:
            agent_dict = self.agents[agent]
            self.agents[agent]["obs"] = torch.zeros((self.args.num_steps, self.args.num_envs) + agent_dict["obs"].shape).to(self.device)
            self.agents[agent]["actions"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["action_masks"] = torch.zeros((self.args.num_steps, self.args.num_envs, agent_dict["max_action_space_size"]), dtype=torch.bool).to(self.device)
            self.agents[agent]["resets"] = np.array(reset[agent]) # TODO: Need to update with determinism
            self.agents[agent]["next_obs"] = torch.Tensor(self.agents[agent]["resets"]).to(self.device)
            self.agents[agent]["rewards"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["episode_rewards"] = torch.zeros(self.args.num_envs).to(self.device)
            self.agents[agent]["episode_lengths"] = torch.zeros(self.args.num_envs).to(self.device)
            self.agents[agent]["dones"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["q_values"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["next_q_values"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)
        self.global_step = 0
        self._reset_red_diagnostics()

    def mask_actions(self, new_action_mask, action_mask):
        new_mask = torch.tensor(
            new_action_mask,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return new_mask
    
    def update_action_masks(self, step: int):
        masks = self.envs.call("action_mask") if self.args.async_env else [env.unwrapped.action_mask for env in self.envs.envs]
        for agent in self.agents:
            for i in range(self.args.num_envs):
                self.agents[agent]["action_masks"][step][i] = self.mask_actions(masks[i][agent], self.agents[agent]["action_masks"][step][i])


    def step_multiagent(self, step: int):
        self.global_step += self.args.num_envs
        self.dones[step] = self.next_done
        policy_action = {}

        for agent in self.agents:
            self.agents[agent]["obs"][step] = self.agents[agent]["next_obs"]
            obs = self.agents[agent]["obs"][step]
            action_mask = self.agents[agent]["action_masks"][step]
            with torch.no_grad():
                action = self.agents[agent]["policy"].select_action(obs, action_mask)
                self.agents[agent]["q_values"][step] = self.agents[agent]["policy"].get_value(obs, action, action_mask=action_mask)
                self.agents[agent]["next_q_values"][step] = self.agents[agent]["policy"].get_value(self.agents[agent]["next_obs"], action_mask=self.agents[agent]["action_masks"][step+1])
            self.agents[agent]["actions"][step] = action
            action = action.cpu().numpy()

            # Execute the selected action in the environment to collect experience for training.
            policy_action[agent] = action
        #print(policy_action)

        print(f"Step {step}: Executing actions {policy_action} in the environment.")
        obs, reward, done, _, info = self.envs.step(policy_action)

        for agent in self.agents:
            self.agents[agent]["dones"][step] = self.next_done
            self.agents[agent]["episode_lengths"] += 1
            next_obs_tensor = torch.tensor(obs[agent], dtype=torch.float32, device=self.device)
            self.agents[agent]["next_obs"] = next_obs_tensor
            if f"{agent}_reward" in info:
                self.agents[agent]["rewards"][step] = torch.tensor(info[f"{agent}_reward"], dtype=torch.float32, device=self.device).view(-1)
            self.agents[agent]["episode_rewards"] += self.agents[agent]["rewards"][step]
            
            if agent == "red" and "red_action" in info and "red_action_success" in info and "red_target_valid" in info:
                action_name = info["red_action"][0] if isinstance(info["red_action"], (list, tuple, np.ndarray)) else info["red_action"]
                kill_chain_phases = info.get("red_kill_chain_phases", [])
                phase = self._phase_bucket(action_name, kill_chain_phases)
                
                success_raw = info["red_action_success"]
                valid_raw = info["red_target_valid"]
                reward_raw = info["red_reward"] if "red_reward" in info else 0.0
                success = bool(success_raw[0] if isinstance(success_raw, (list, tuple, np.ndarray)) else success_raw)
                valid_target = bool(valid_raw[0] if isinstance(valid_raw, (list, tuple, np.ndarray)) else valid_raw)
                red_reward = float(reward_raw[0] if isinstance(reward_raw, (list, tuple, np.ndarray)) else reward_raw)

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
        output_str = f"global_step={self.global_step}"
        
        for agent in self.agents:
            mean_rew = self.agents[agent]["rewards"].sum(axis=0).mean()
            output_str += f", {agent}_episodic_return={mean_rew}"
            writer.add_scalar(f"charts/{agent}_episodic_return", mean_rew, self.global_step)
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
        for agent in self.agents:
            self.agents[agent]["batched"] = {}
            self.agents[agent]["batched"]["obs"] = self.agents[agent]["obs"].reshape((-1,) + self.agents[agent]["shape"])
            self.agents[agent]["batched"]["actions"] = self.agents[agent]["actions"].reshape(-1)
            self.agents[agent]["batched"]["rewards"] = self.agents[agent]["rewards"].reshape(-1)
            self.agents[agent]["batched"]["q_values"] = self.agents[agent]["q_values"].reshape(-1)
            self.agents[agent]["batched"]["action_masks"] = self.agents[agent]["action_masks"].reshape(-1, self.agents[agent]["action_masks"].shape[-1])
            self.agents[agent]["batched"]["dones"] = self.agents[agent]["dones"].reshape(-1)
    
    def update_policy(self, mb_inds):
        # Convert mb_inds to CPU tensor for proper indexing
        # mb_inds = torch.from_numpy(mb_inds).long()
        
        for agent in self.agents:
            if getattr(self.args, "drive", False):
                with open(f'/content/drive/MyDrive/RLCS/{self.args.experiment_name}_step.txt', 'w') as f:
                    f.write("Global step: " + str(self.global_step) + ", Episode: " + str(self.episode))
                print("Global step: " + str(self.global_step) + ", Episode: " + str(self.episode))

            self.agents[agent]["q_value"] = self.agents[agent]["policy"].get_value(
                self.agents[agent]["batched"]["obs"][mb_inds],
                self.agents[agent]["batched"]["actions"].long()[mb_inds],
                action_mask=self.agents[agent]["batched"]["action_masks"][mb_inds],
            )

    def calculate_loss(self, mb_inds):
        # Convert mb_inds to CPU tensor for proper indexing
        # mb_inds = torch.from_numpy(mb_inds).long()
        
        for agent in self.agents:
            q_values = self.agents[agent]["q_value"]
            next_q_values = self.agents[agent]["next_q_values"].reshape(-1)[mb_inds]
            rewards = self.agents[agent]["batched"]["rewards"][mb_inds].view(-1)
            dones = self.agents[agent]["batched"]["dones"][mb_inds].view(-1)
            target = rewards + (1 - dones) * self.args.gamma * next_q_values
            self.agents[agent]["loss"] = self.agents[agent]["policy"].lossfn(q_values, target)
        

    def backpropagate(self, update):
        for agent in self.agents:
            # Backpropagation for Actor-Critic policy
            self.agents[agent]["optimizer"].zero_grad()
            self.agents[agent]["loss"].backward()
            self.agents[agent]["optimizer"].step()

    def save_models(self):
        if self.args.nrec:
            run_path = files("persistent01.cyberwheel.data.models").joinpath(self.args.experiment_name)
        elif self.args.drive:
            run_path = files("content.drive.MyDrive.RLCS.models").joinpath(self.args.experiment_name)
        else:
            run_path = files("cyberwheel.data.models").joinpath(self.args.experiment_name)
        agent_paths = {}
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        for agent in self.agents:
            agent_path = run_path.joinpath(f"{agent}_agent.pt")
            globalstep_path = run_path.joinpath(f"{agent}_{self.global_step}.pt")

            if getattr(self.args, "drive", False):
                drive_model_dir = Path("/content/drive/MyDrive/RLCS/model/" + self.args.experiment_name)
                drive_model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.agents[agent]["policy"].state_dict(), drive_model_dir / f"{agent}_agent.pt")
                torch.save(self.agents[agent]["policy"].state_dict(), drive_model_dir / f"{agent}_{self.global_step}.pt")
                print("Models saved to Google Drive.")

            torch.save(self.agents[agent]["policy"].state_dict(), agent_path)
            torch.save(self.agents[agent]["policy"].state_dict(), globalstep_path)
            print(f"Saved {agent} to {agent_path}")

            if self.args.track:
                import wandb
                wandb.save(agent_path, base_path=run_path, policy="now")
                wandb.save(globalstep_path, base_path=run_path, policy="now")
            agent_paths[agent] = agent_path

        return agent_paths
    
    def load_models(self, name="red_agent"):
        """Load Q-tables from disk"""
        for agent in self.agents:
            if self.args.nrec:
                load_path = files("persistent01.cyberwheel.data.models").joinpath(self.args.experiment_name)
            elif self.args.drive:
                load_path = Path(str(files("content.drive.MyDrive.RLCS.models").joinpath(self.args.experiment_name)))
            else:
                load_path = files("cyberwheel.data.models").joinpath(self.args.experiment_name)
            
            agent_path = load_path.joinpath(f"{name}.pt")
            print(f"Loading {agent} agent from: {agent_path}")
            if os.path.exists(agent_path):
                self.agents[agent]["policy"].load_state_dict(torch.load(agent_path, map_location=torch.device(self.device)))                
                self.global_step = 0
                self.episode = 1
                self._reset_red_diagnostics()
            
                print(f"Loaded {agent} with fresh history (epsilon={self.initial_epsilon})")
                
    
    def log_training_metrics(self, writer):
        for agent in self.agents:
            writer.add_scalar(f"losses/{agent}_q_loss", self.agents[agent]["loss"].item(), self.global_step)

    def reset(self):
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

    def expand_model(self, abstract_policy, method="probabilities"):
        pass