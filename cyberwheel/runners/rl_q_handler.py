from collections import defaultdict
from pathlib import Path

from torch import optim

from cyberwheel.utils import RLPolicyQLearning
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

        for agent in agents:
            self.agents[agent] = agents[agent]
            self.agents[agent]["shape"] = self.agents[agent]["obs"].shape
            self.agents[agent]["policy"] = RLPolicyQLearning(self.agents[agent]["max_action_space_size"], self.agents[agent]["shape"]).to(self.device)
            self.agents[agent]["optimizer"] = optim.Adam([
                { 'params': list(self.agents[agent]["policy"].model.parameters()),  'lr': float(self.args.learning_rate),  'eps': 1e-3 },
            ])
            self.agents[agent]["lossfn"] = torch.nn.MSELoss()

        if self.load:
            print(self.args.experiment_name)
            self.load_models()

    def _reset_red_diagnostics(self):
        self.red_action_attempts = defaultdict(int)
        self.red_action_successes = defaultdict(int)
        self.steps_before_first_valid_target = 0 # how to indicate no flag found?
        self.reached_valid_target = False
        self.number_of_impacted_valid_targets = 0
        self.red_reward_valid_targets = 0.0
        self.red_reward_invalid_targets = 0.0
        self.red_valid_target_attempts = 0
        self.red_invalid_target_attempts = 0
        self.num_valid_targets = torch.zeros(self.args.num_envs)

    def _phase_bucket(self, action_name, phase_list=None):
        action = str(action_name).lower().strip("[]'")
        if phase_list is not None and action not in ["discovery", "impact"]:
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
            self.agents[agent]["next_action_masks"] = torch.zeros((self.args.num_steps, self.args.num_envs, agent_dict["max_action_space_size"]), dtype=torch.bool).to(self.device)
            self.agents[agent]["resets"] = np.array(reset[agent]) # TODO: Need to update with determinism
            self.agents[agent]["next_obs"] = torch.Tensor(self.agents[agent]["resets"]).to(self.device)
            self.agents[agent]["next_obses"] = torch.zeros((self.args.num_steps, self.args.num_envs) + agent_dict["shape"]).to(self.device)
            self.agents[agent]["rewards"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["episode_rewards"] = torch.zeros(self.args.num_envs).to(self.device)
            self.agents[agent]["episode_lengths"] = torch.zeros(self.args.num_envs).to(self.device)
            self.agents[agent]["dones"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            # self.agents[agent]["q_values"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            # self.agents[agent]["next_q_values"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
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
                action_mask = self.mask_actions(masks[i][agent], self.agents[agent]["action_masks"][step][i])
                self.agents[agent]["action_masks"][step][i] = action_mask
                self.agents[agent]["next_action_masks"][step-1][i] = action_mask # set next action mask for previous step


    def step_multiagent(self, step: int):
        self.global_step += self.args.num_envs
        self.dones[step] = self.next_done
        policy_action = {}

        for agent in self.agents:
            self.agents[agent]["obs"][step] = self.agents[agent]["next_obs"]
            for env_idx in range(self.args.num_envs):

                obs = self.agents[agent]["obs"][step][env_idx]
                action_mask = self.agents[agent]["action_masks"][step][env_idx]
                with torch.no_grad():
                    action = self.agents[agent]["policy"].select_action(obs, action_mask=action_mask)
                    # self.agents[agent]["q_values"][step] = self.agents[agent]["policy"].get_value(obs, action)
                self.agents[agent]["actions"][step][env_idx] = action
            # action = action.cpu().numpy()

            # Execute the selected action in the environment to collect experience for training.
            policy_action[agent] = self.agents[agent]["actions"][step].cpu().numpy()
        #print(policy_action)

        # print(f"Step {step}: Executing actions {policy_action} in the environment.")
        obs, reward, done, _, info = self.envs.step(policy_action)

        for agent in self.agents:
            for env_idx in range(self.args.num_envs):
                self.agents[agent]["episode_lengths"][env_idx] += 1
                # self.agents[agent]["next_obs"] = next_obs_tensor
                # print(f"Step {step}: Received next_obs for {agent} with shape {obs[agent].shape}")
                self.agents[agent]["next_obses"][step][env_idx] = torch.tensor(obs[agent][env_idx]).to(self.device)
                # with torch.no_grad():
                # next_q = self.agents[agent]["policy"].get_value(next_obs_tensor)
                # self.agents[agent]["next_q_values"][step] = next_q.view(-1)
                if f"{agent}_reward" in info:
                    self.agents[agent]["rewards"][step][env_idx] = torch.tensor(info[f"{agent}_reward"][env_idx]).to(self.device).view(-1)
                reward_val = self.agents[agent]["rewards"][step][env_idx].item()
                self.agents[agent]["episode_rewards"][env_idx] += reward_val
                self.agents[agent]["episode_lengths"][env_idx] += 1

            if not self.reached_valid_target and agent == "red":
                self.steps_before_first_valid_target += 1
            
            if agent == "red" and "red_action" in info and "red_action_success" in info and "red_target_valid" in info:
                self.num_valid_targets = len(info["valid_targets"])
                action_name = info["red_action"]
                kill_chain_phases = info.get("red_kill_chain_phases", [])
                phase = self._phase_bucket(action_name, kill_chain_phases)
                success = bool(info["red_action_success"])
                if phase in ["discovery", "privilege-escalation", "lateral-movement", "impact"] and success:
                    print(f"Phase: {phase} - Received reward {reward_val}")
                valid_target = bool(info["red_target_valid"])
                successful_impact = success and valid_target and phase == "impact"

                if not self.reached_valid_target and successful_impact:
                    # reached first valid target
                    self.reached_valid_target = True
                    print(f"Reached first valid target on step {self.steps_before_first_valid_target}")

                if successful_impact:
                    # log number of valid impacted targets
                    self.number_of_impacted_valid_targets += 1

                self.red_action_attempts[phase] += 1
                if success:
                    self.red_action_successes[phase] += 1

                if valid_target:
                    self.red_valid_target_attempts += 1
                    self.red_reward_valid_targets += reward_val
                else:
                    self.red_invalid_target_attempts += 1
                    self.red_reward_invalid_targets += reward_val

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
            self.agents[agent]["batched"]["next_obses"] = self.agents[agent]["next_obses"].reshape((-1,) + self.agents[agent]["shape"])
            self.agents[agent]["batched"]["actions"] = self.agents[agent]["actions"]
            self.agents[agent]["batched"]["rewards"] = self.agents[agent]["rewards"]
            self.agents[agent]["batched"]["action_masks"] = self.agents[agent]["action_masks"].reshape(-1, self.agents[agent]["action_masks"].shape[-1])
            self.agents[agent]["batched"]["next_action_masks"] = self.agents[agent]["next_action_masks"].reshape(-1, self.agents[agent]["next_action_masks"].shape[-1])
            self.agents[agent]["batched"]["dones"] = self.agents[agent]["dones"]
            
            
    def calculate_explained_variance(self):
        pass
    
    def update_policy(self, mb_inds):
        for agent in self.agents:
            if getattr(self.args, "drive", False):
                with open(f'/content/drive/MyDrive/RLCS/{self.args.experiment_name}_step.txt', 'w') as f:
                    f.write("Global step: " + str(self.global_step) + ", Episode: " + str(self.episode))
                print("Global step: " + str(self.global_step) + ", Episode: " + str(self.episode))


    def calculate_loss(self, mb_inds):
        # Convert mb_inds to CPU tensor for proper indexing
        # mb_inds = torch.from_numpy(mb_inds).long()
        for agent in self.agents:
            obs = self.agents[agent]["batched"]["obs"][mb_inds]
            next_obs = self.agents[agent]["batched"]["next_obses"][mb_inds]
            actions = self.agents[agent]["batched"]["actions"][mb_inds].long()
            next_action_masks = self.agents[agent]["batched"]["next_action_masks"][mb_inds]
            q_values = self.agents[agent]["policy"].get_value(obs, actions)
            
            with torch.no_grad():
                next_q_values = self.agents[agent]["policy"].get_value(next_obs, action_mask=next_action_masks)
            
            rewards = self.agents[agent]["batched"]["rewards"][mb_inds]
            dones = self.agents[agent]["batched"]["dones"][mb_inds]
            
            if not obs.dim == 2 and not actions.dim() == 2 and not rewards.dim() == 2 and not dones.dim() == 2 and not q_values.dim() == 2 and not next_q_values.dim() == 2 and not next_action_masks.dim() == 2:
                raise ValueError(f"Unexpected dimensions - obs: {obs.shape}, actions: {actions.shape}, rewards: {rewards.shape}, dones: {dones.shape}, q_values: {q_values.shape}, next_q_values: {next_q_values.shape}, next_action_masks: {next_action_masks.shape}")
            
            target = rewards + (1 - dones) * self.args.gamma * next_q_values
            # moving target... instable learning
            self.agents[agent]["loss"] = self.agents[agent]["lossfn"](q_values, target)

    def backpropagate(self, update):
        for agent in self.agents:
            # Backpropagation for Actor-Critic policy
            self.agents[agent]["optimizer"].zero_grad()
            self.agents[agent]["loss"].backward()
            self.agents[agent]["optimizer"].step()
            if self.agents[agent]["policy"].use_target:
                self.agents[agent]["policy"].soft_update()

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
            self.agents[agent]["resets"] = np.array(reset[agent]) # 1, 401
            self.agents[agent]["next_obs"] = torch.Tensor(self.agents[agent]["resets"]).to(self.device)
            self.agents[agent]["next_obses"] = torch.zeros((self.args.num_steps, self.args.num_envs) + self.agents[agent]["shape"]).to(self.device)
            self.agents[agent]["episode_rewards"] = torch.zeros(self.args.num_envs).to(self.device)
            self.agents[agent]["episode_lengths"] = torch.zeros(self.args.num_envs).to(self.device)
            self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["obs"] = torch.zeros((self.args.num_steps, self.args.num_envs) + self.agents[agent]["next_obs"].shape[1:]).to(self.device)
            self.agents[agent]["rewards"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["action_masks"] = torch.zeros((self.args.num_steps, self.args.num_envs, self.agents[agent]["max_action_space_size"]), dtype=torch.bool).to(self.device)
            self.agents[agent]["next_action_masks"] = torch.zeros((self.args.num_steps, self.args.num_envs, self.agents[agent]["max_action_space_size"]), dtype=torch.bool).to(self.device)
        self._reset_red_diagnostics()

    def get_action_mapping(self, path="/cyberwheel/data/configs/red_agent/rl_red_complex.yaml"):
        # mapping of new actions given indicies of old found in yaml file
        with open(path, "r") as f:
            data=f.read()
        phases = {"portscan": 0, "pingsweep": 0, "discovery": 0, "lateral-movement": 0, "privilege-escalation": 0, "impact": 0}
        action_data = data.split("actions:")[1].strip()
        for action in action_data.split("\n\n"):
            phase = action.split("phase:")[1].strip()
            phases[phase] += 1

    def expand_model(self, abstract_policy, method=None):    
        self.agents["red"]["policy"].expand_model(abstract_policy["agents"]["red"]["obs_shape"], abstract_policy["agents"]["red"]["action_space_size"], method=method, mapping=self.get_action_mapping())
