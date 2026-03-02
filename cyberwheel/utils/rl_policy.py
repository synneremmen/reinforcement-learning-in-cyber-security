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
    def __init__(self, action_space_shape=0, obs_space_shape=0, epsilon=0.1, num_hosts=10):
        super().__init__()
        # Policy table has an input layer of the size of S, and an output layer of the size of A.
        # Predicts the best action to take at the current state.
        obs_space_shape = int(np.array(obs_space_shape).prod())
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.num_hosts = num_hosts # TODO: hardcoded, bad coding...
        self.state_space = (1 + 3 * 2**6)**self.num_hosts
        self.compressed_state_space = (1 + 3 * 2*4)**self.num_hosts # given that we compress the 7 host-level attributes to 3, and keep the global attribute, we get 1 + 3*6 possible combinations per host
        print(f"Using {self.num_hosts} hosts:")
        #print(f"Calculated state space: {self.state_space}")
        #print(f"Calculated compressed state space: {self.compressed_state_space}")
        #print(f"Size of q_table: {self.compressed_state_space} x {self.action_space_shape} = {self.compressed_state_space * self.action_space_shape} entries")
        #print("Total size of q_table for compressed state space in GB:", self.compressed_state_space * self.action_space_shape * 4 / (1024**3)) # 4 bytes per float32
        #print("Total size of q_table for original state space in GB:", self.state_space * self.action_space_shape * 4 / (1024**3)) # 4 bytes per float32
        self.q_table = defaultdict(lambda: torch.zeros(self.action_space_shape))
        # self.q_table = torch.zeros(self.compressed_state_space, self.action_space_shape)
        print(f"Initialized Q-table with shape {self.q_table} ")
        self.epsilon = epsilon

    def obs_to_state(self, obs):
        state = []
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]  
        global_att = obs[:,-1] 
        host_obs = obs[:,:self.obs_space_shape - 1] 
        host_obs = host_obs.reshape(self.num_hosts, -1)[:,:8]
        for host_idx in range(self.num_hosts):
            host_features = host_obs[host_idx]
            type_, sweeped, scanned, discovered, on_host, escalated, impacted, visited = host_features
            
            if impacted == 1: # get state
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
            state.append([int(type_), int(on_host), int(curr), int(visited)])
        state = np.array(state).flatten()
        state = np.concatenate([state, global_att])
        return tuple(state)
        
    def get_action_and_value(self, obs, action=None, action_mask=None):
        state = self.obs_to_state(obs)
        if action is None:
            action = self.greedy_action(state, action_mask)
        value = self.get_value(state, action)
        return action, value , None, None
        
    def get_value(self, state, action): # best action value for a given state
        return self.q_table[state][action].item() 
    
    def greedy_action(self, state, action_mask=None):
        # action_values = self.q_table[state]
        action_values = self.q_table.setdefault(
        state, torch.zeros(self.action_space_shape) 
        # default to random values [0,1] to encourage exploration of unseen actions?
        )
        if action_mask is not None:
            action_values = action_values.clone()
            action_values[action_mask == 0] = float("-inf")
        greedy_action = torch.argmax(action_values)
        return greedy_action.item()
    
    def select_action(self, obs, action_mask):
        """Select action using epsilon-greedy with action mask"""
        state = self.obs_to_state(obs)
        if np.random.uniform() < self.epsilon:
            valid_actions = torch.where(action_mask)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions.cpu().numpy())
            else:
                return 0
        else:
            return self.greedy_action(state, action_mask)

    def update_q_table(self, obs, action, reward, next_obs, done, next_action_mask=None, alpha=0.1, gamma=0.99):
        """Update Q-table using Q-learning update rule"""
        state = self.obs_to_state(obs)
        next_state = self.obs_to_state(next_obs)
    
        if done:
            td_target = reward
        else:
            if next_action_mask is not None:
                next_action_mask_tensor = torch.tensor(next_action_mask, dtype=torch.bool)
                next_q_values = self.q_table[next_state] * next_action_mask_tensor
                best_next_q = torch.max(next_q_values).item()
            else:
                best_next_q = torch.max(self.q_table[next_state]).item()
            td_target = reward + gamma * best_next_q
        
        td_error = td_target - self.get_value(state, action)

        self.q_table[state][action] += alpha * td_error

"""
correct: 
Stepping environment with actions: {'red': array([346])}
Stepping environment with actions: {'red': array([219])}

incorrect:
Stepping environment with actions: {'red': array(0)}
""" 