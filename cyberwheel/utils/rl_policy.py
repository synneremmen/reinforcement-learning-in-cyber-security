from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            """Initialise neural network weights using orthogonal initialization. Works well in practice."""
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer


class RLPolicyActorCritic(nn.Module):
    """
    The agent class that contains the code for defining the actor and critic networks used by PPO.
    Also includes functions for getting values from the critic and actions from the actor.
    """

    def __init__(self, action_space_shape=0, obs_space_shape=0):
        super().__init__()
        # Actor network has an input layer, 2 hidden layers with 64 nodes, and an output layer.
        # Input layer is the size of the observation space and output layer is the size of the action space.
        # Predicts the best action to take at the current state.
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(int(np.array(obs_space_shape).prod()), 64)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_space_shape), std=0.01),
        )

        # Critic network has an input layer, 2 hidden layers with 64 nodes, and an output layer.
        # Input layer is the size of the observation space and output layer has 1 node for the predicted value.
        # Predicts the "value" - the expected cumulative reward from using the actor policy from the current state onward.
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(int(np.array(obs_space_shape).prod()), 64)
            ),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, obs):
        """Gets the value for a given state x by running x through the critic network"""
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None, action_mask=None):
        """
        Gets the action and value for the current state by running x through the actor and critic respectively.
        Also calculates the log probabilities of the action and the policy's entropy which are used to calculate PPO's training loss.
        """
        logits = self.actor(obs)
        if action_mask != None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)
    

class RLPolicyTableBased(nn.Module): 
    """
    TODO: Add header
    """
    def __init__(self, action_space_shape=0, obs_space_shape=0, epsilon=0.1, num_hosts=6, device="cpu"):
        super().__init__()
        obs_space_shape = int(np.array(obs_space_shape).prod())
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.num_hosts = num_hosts # TODO: hardcoded, bad coding...
        self.device = device
        # self.state_space = (1 + 3 * 2**6)**self.num_hosts
        # self.compressed_state_space = (1 + 3 * 2*4)**self.num_hosts # given that we compress the 7 host-level attributes to 3, and keep the global attribute, we get 1 + 3*6 possible combinations per host
        print(f"Using {self.num_hosts} hosts:")
        #print(f"Size of q_table: {self.compressed_state_space} x {self.action_space_shape} = {self.compressed_state_space * self.action_space_shape} entries")
        self.q_table = defaultdict(self._new_q_values)
        # default to random values [0,1] to encourage exploration of unseen actions?
        print(f"Initialized Q-table with shape {len(self.q_table)} x {self.action_space_shape} = {len(self.q_table) * self.action_space_shape} entries")
        self.epsilon = epsilon

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device_arg = kwargs.get("device", None)
        if device_arg is None and len(args) > 0 and isinstance(args[0], (str, torch.device)):
            device_arg = args[0]
        if device_arg is not None:
            self.device = torch.device(device_arg)
            for state, q_values in list(self.q_table.items()):
                self.q_table[state] = q_values.to(self.device)
        return module

    def _new_q_values(self):
        return torch.zeros(self.action_space_shape, device=self.device)

    def _get_state_q(self, state):
        q_values = self.q_table.get(state)
        if q_values is None:
            q_values = self._new_q_values()
            self.q_table[state] = q_values
        elif q_values.device != self.device:
            q_values = q_values.to(self.device)
            self.q_table[state] = q_values
        return q_values

    def obs_to_state(self, obs):
        obs_tensor = torch.as_tensor(obs, device=self.device)
        if obs_tensor.ndim > 1:
            obs_tensor = obs_tensor.reshape(-1)

        state = []
        global_att = int(obs_tensor[-1].item())
        host_obs = obs_tensor[: self.obs_space_shape - 1]
        host_obs = host_obs.reshape(self.num_hosts, -1)[:, :8]
        for host_idx in range(self.num_hosts):
            host_features = host_obs[host_idx]
            type_, sweeped, scanned, discovered, on_host, escalated, impacted = [int(v.item()) for v in host_features[:7]]
            
            if impacted == 1:
                curr = 4
            elif escalated == 1:
                curr = 3
            elif discovered == 1:
                curr = 2
            elif scanned == 1:
                curr = 1
            elif sweeped == 1 or on_host == 1:
                curr = 0
            else:
                curr = -1

            state.extend([type_, int(on_host), int(curr)])

        state.append(global_att)
        return tuple(state)
        
    def get_action_and_value(self, obs, action=None, action_mask=None):
        state = self.obs_to_state(obs)
        if action is None:
            action = self.greedy_action(state, action_mask)
        value = self.get_value(state, action)
        return action, value , None, None
        
    def get_value(self, state, action): # best action value for a given state
        return self._get_state_q(state)[int(action)].item() 
    
    def get_best_value(self, state, action_mask=None):
        q_values = self._get_state_q(state)
        if action_mask is not None:
            q_values = q_values.clone()
            q_values[~torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)] = float("-inf")
            best_value = torch.max(q_values).item()
        else:
            best_value = torch.max(q_values).item()
        return best_value
    
    def greedy_action(self, state, action_mask=None):
        action_values = self._get_state_q(state)
        if action_mask is not None:
            action_values = action_values.clone()
            action_values[~torch.as_tensor(action_mask, dtype=torch.bool, device=action_values.device)] = float("-inf")
        greedy_action = torch.argmax(action_values)
        return greedy_action.item()
    
    def select_action(self, obs, action_mask):
        """Select action using epsilon-greedy with action mask"""
        state = self.obs_to_state(obs)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        if np.random.uniform() < self.epsilon:
            valid_actions = torch.where(mask)[0]
            if len(valid_actions) > 0:
                random_idx = torch.randint(0, len(valid_actions), (1,), device=self.device)
                return int(valid_actions[random_idx].item())
            else:
                return 0
        else:
            return self.greedy_action(state, mask)

    def normalize_reward(self, reward):
        # min_v = -10 # worst reward
        # max_v = 1000 # best possible reward (capture the flag)
        # how would i do this if z-score normalization?
        # return (reward - self.reward_mean) / (self.reward_std + 1e-9)
        return reward #  (reward - min_v) / (max_v - min_v) # number between 0 and 1

    def update_q_table(self, obs, action, reward, next_obs, done, next_action_mask=None, alpha=0.1, gamma=0.99):
        """Update Q-table using Q-learning update rule"""
        state = self.obs_to_state(obs)
        next_state = self.obs_to_state(next_obs)
        normalized_reward = float(self.normalize_reward(float(torch.as_tensor(reward).item())))
        done = bool(torch.as_tensor(done).item())
        action_idx = int(torch.as_tensor(action).item())
    
        if done:
            td_target = normalized_reward
        else:
            best_next_q = self.get_best_value(next_state, next_action_mask)
            td_target = normalized_reward + gamma * best_next_q
        
        td_error = td_target - self.get_value(state, action_idx)
        update_value = alpha * td_error

        self._get_state_q(state)[action_idx] += update_value
        return float(update_value)
        # q(s,a) = q(s,a) + alpha(R + gamma maxq(nexts,a) - q(s,a))
        # q(s,a) = q(s,a) + alpha * td_error
        # der td_error = td_target - get_value(s,a)
        # og td_target = R + gamma * best_next_q