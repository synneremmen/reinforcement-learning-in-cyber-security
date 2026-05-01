from collections import defaultdict
from pathlib import Path

from torch import optim
from torch.optim.lr_scheduler import PolynomialLR

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
        self.visited_states = set()

        for agent in agents:
            self.agents[agent] = agents[agent]
            self.agents[agent]["shape"] = self.agents[agent]["obs"].shape
            self.agents[agent]["policy"] = RLPolicyQLearning(self.agents[agent]["max_action_space_size"], self.agents[agent]["shape"], use_target=self.args.use_target).to(self.device)
            self.agents[agent]["optimizer"] = optim.Adam([
                { 'params': list(self.agents[agent]["policy"].model.parameters()),  'lr': float(self.args.learning_rate),  'eps': 1e-3 },
            ])
            self.agents[agent]["scheduler"] = PolynomialLR(self.agents[agent]["optimizer"], total_iters=self.args.total_timesteps, power=0.8)
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
        self.visited_states.update(tuple(obs[agent][i].tolist()) for agent in self.agents for i in range(self.args.num_envs))

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
                # if phase in ["discovery", "privilege-escalation", "lateral-movement", "impact"] and success:
                    # print(f"Phase: {phase} - Received reward {reward_val}")
                valid_target = bool(info["red_target_valid"])
                successful_impact = success and valid_target and phase == "impact"

                if not self.reached_valid_target and successful_impact:
                    # reached first valid target
                    self.reached_valid_target = True
                    # print(f"Reached first valid target on step {self.steps_before_first_valid_target}")

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
        writer.add_scalar("charts/red_steps_before_first_valid_target", self.steps_before_first_valid_target, self.global_step)
        writer.add_scalar("charts/red_number_of_impacted_valid_targets", self.number_of_impacted_valid_targets, self.global_step)
        writer.add_scalar("charts/number_valid_targets", self.num_valid_targets, self.global_step)
        impact_ratio = self.number_of_impacted_valid_targets / self.num_valid_targets
        writer.add_scalar("charts/red_impacted_valid_targets_ratio", impact_ratio, self.global_step)
        writer.add_scalar("charts/red_learning_rate", self.agents[agent]["optimizer"].param_groups[0]["lr"], self.global_step)

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
            # self.agents[agent]["scheduler"].step()
            if self.agents[agent]["policy"].use_target:
                self.agents[agent]["policy"].soft_update()

    def save_models(self):
        if self.args.nrec:
            run_path = Path("/persistent01/cyberwheel/models") / self.args.experiment_name
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

            # no need to save target model
            policy_state = {k: v for k, v in self.agents[agent]["policy"].state_dict().items() if not k.startswith("target_model.")}

            checkpoint = {
                "state_dict": policy_state,
                "architecture": self.agents[agent]["policy"].architecture_metadata(),
            }
            torch.save(checkpoint, agent_path)
            torch.save(checkpoint, globalstep_path)
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
                load_path = Path("/persistent01/cyberwheel/models") / self.args.experiment_name
            elif self.args.drive:
                load_path = Path(str(files("content.drive.MyDrive.RLCS.models").joinpath(self.args.experiment_name)))
            else:
                load_path = files("cyberwheel.data.models").joinpath(self.args.experiment_name)
            
            agent_path = load_path.joinpath(f"{name}.pt")
            print(f"Loading {agent} agent from: {agent_path}")
            if os.path.exists(agent_path):
                checkpoint = torch.load(agent_path, map_location=torch.device(self.device))
                state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
                inferred_hidden_layers = RLPolicyQLearning.hidden_layers_from_state_dict(state_dict)
                print(f"Inferred hidden layers from state_dict: {inferred_hidden_layers}")
                hidden_layers = None
                if isinstance(checkpoint, dict):
                    hidden_layers = checkpoint.get("architecture", {}).get("hidden_layers")
                if not hidden_layers or list(hidden_layers) != list(inferred_hidden_layers):
                    if hidden_layers not in (None, []):
                        print(f"Checkpoint architecture metadata {hidden_layers} does not match state_dict layers {inferred_hidden_layers}; using state_dict layers.")
                    hidden_layers = inferred_hidden_layers
                print(f"Loaded hidden layers: {hidden_layers}")

                policy = self.agents[agent]["policy"]
                if list(policy.hidden_layers) != list(hidden_layers):
                    self.agents[agent]["policy"] = RLPolicyQLearning(
                        action_space_shape=policy.action_space_shape,
                        obs_space_shape=policy.obs_space_shape,
                        epsilon=policy.epsilon,
                        use_target=policy.use_target,
                        hidden_layers=hidden_layers,
                    ).to(self.device)
                    policy = self.agents[agent]["policy"]
                    self.agents[agent]["optimizer"] = optim.Adam([
                        {"params": list(policy.model.parameters()), "lr": float(self.args.learning_rate), "eps": 1e-3},
                    ])
                    self.agents[agent]["scheduler"] = PolynomialLR(self.agents[agent]["optimizer"], total_iters=self.args.total_timesteps, power=0.8)

                model_state_dict = {
                    key[6:] if key.startswith("model.") else key: value
                    for key, value in state_dict.items()
                    if not key.startswith("target_model.")
                }
                policy.model.load_state_dict(model_state_dict)
                if policy.use_target:
                    policy.set_target_model()
                self.global_step = 0
                self.episode = 1
                self._reset_red_diagnostics()
            
                # print(f"Loaded {agent} with fresh history (epsilon={self.initial_epsilon})")
                
    
    def log_training_metrics(self, writer):
        for agent in self.agents:
            writer.add_scalar(f"losses/{agent}_q_loss", self.agents[agent]["loss"].item(), self.global_step)
            writer.add_scalar(f"charts/{agent}_num_states_visited", len(self.visited_states), self.global_step)

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
        self.mapping = phases

    def expand_model(self, abstract_policy, args, writer=None):
        print(f"Expanding model from action space {abstract_policy.action_space_shape} to {self.agents['red']['policy'].action_space_shape} using method '{args.method}'")
        # print("-- Hidden layers before expansion: ", self.agents["red"]["policy"].hidden_layers)
        if args.method == "copy_params":
            new_model = self.agents["red"]["policy"].copy_params(old_policy=abstract_policy, mapping=self.mapping)
        elif args.method == "increase_depth":
            new_model = self.agents["red"]["policy"].increase_depth(old_policy=abstract_policy, reuse_model=args.reuse_model)
        elif args.method == "kl_divergence":
            new_model = self.policy_distallation(abstract_policy, args, writer=writer)
        else:
            raise ValueError("Invalid expansion method specified. Use 'copy_params', 'increase_depth' or 'kl_divergence'.")
        
        # print("-- Hidden layers after expansion: ", self.agents["red"]["policy"].hidden_layers)
        self.agents["red"]["policy"].model = new_model
        if self.agents["red"]["policy"].use_target:
            self.agents["red"]["policy"].set_target_model()


    def policy_distallation(self, abstract_policy, args, writer=None):
        if args.reuse_model:
            print("Reusing old model weights for layers with matching shapes.")
            new_model = self.agents["red"]["policy"].increase_depth(old_policy=abstract_policy)
        else:
            new_model = self.agents["red"]["policy"]._build_model(self.agents["red"]["policy"].hidden_layers)
        optimizer = optim.Adam(new_model.parameters())
        global_step = 0
        args.kl_divergence_steps
        num_updates = args.kl_divergence_steps // self.args.num_steps
        abstract_policy.model.eval()
        new_model.model.train()
        print(f"Running for {args.kl_divergence_steps} steps with {num_updates} updates of KL divergence loss.")
        for _ in range(num_updates):
            self.reset()
            obs = self.agents["red"]["next_obs"]
            for step in range(self.args.num_steps):
                self.update_action_masks(step)
                if args.expand_teacher_probs:
                    with torch.no_grad():
                        action = new_model.select_action(obs, action_mask=self.agents["red"]["action_masks"][step][0])
                        probs_teacher = abstract_policy.get_probabilities(obs, mode="teacher")
                        # teacher is smaller than student, so we must expand the teacher probabilities to match the student's action space before calculating KL divergence
                        expanded_probs_teacher = torch.zeros((1, new_model.action_space_shape), device=self.device)
                        idx = 0
                        teacher_idx = 0
                        for host in range(self.args.num_hosts):
                            for count in self.mapping.values():
                                teacher_value = probs_teacher[:, teacher_idx] / count # need to sum to 1, so divide teacher probability by number of new actions corresponding to old action
                                expanded_probs_teacher[:, idx:idx+count] = teacher_value.repeat(1, count)
                                idx += count
                                teacher_idx += 1
                        expanded_probs_teacher[:,-1] = probs_teacher[:,-1]
                    log_probs_student = new_model.get_probabilities(obs, mode="student", expand_teacher_probs=args.expand_teacher_probs)
                else:
                    with torch.no_grad():
                        action = new_model.select_action(obs, action_mask=self.agents["red"]["action_masks"][step][0])
                        probs_teacher = abstract_policy.get_probabilities(obs, mode="teacher")
                        expanded_probs_teacher = probs_teacher
                    # print("Teacher vs student probabilities:")
                    # print(f"Teacher: {expanded_probs_teacher}")
                    log_probs_student = new_model.get_probabilities(obs, mode="student", mapping=self.mapping, num_hosts=self.args.num_hosts, expand_teacher_probs=args.expand_teacher_probs)
                action = action.cpu().numpy()
                policy_action = {"red": action}
                # print(f"Step {step}: Executing action {policy_action} in the environment.")

                obs, reward, done, _, info = self.envs.step(policy_action)
                obs = torch.tensor(obs["red"], dtype=torch.float32, device=self.device)
                kl_loss = new_model.kl_divergence(log_probs_student, expanded_probs_teacher)
                writer.add_scalar("charts/kl_divergence_loss", kl_loss.item(), global_step)
                # print(f"KL divergence loss: {kl_loss.item():.6f}\n")
                optimizer.zero_grad()
                kl_loss.backward()
                optimizer.step()
                global_step += 1
        return new_model
