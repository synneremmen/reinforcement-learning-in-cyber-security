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
    

class RLPolicyTableBased(nn.Module): # bruke table-rl? pip3 install table-rl
    """
    TODO: Add header
    """
    def __init__(self, action_space_shape=0, obs_space_shape=0, num_hosts=4):
        super().__init__()
        # Policy table has an input layer of the size of S, and an output layer of the size of A.
        # Predicts the best action to take at the current state.
        obs_space_shape = int(np.array(obs_space_shape).prod())
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.num_hosts = num_hosts # TODO: hardcoded, bad coding...
        self.state_space = (1 + 3 * 2**6)**self.num_hosts
        self.compressed_state_space = (1 + 3 * 2*4)**self.num_hosts # given that we compress the 7 host-level attributes to 3, and keep the global attribute, we get 1 + 3*6 possible combinations per host
        print(f"Calculated state space: {self.state_space}")
        print(f"Calculated compressed state space: {self.compressed_state_space}")
        self.q_table = torch.zeros(self.compressed_state_space, self.obs_space_shape, action_space_shape)
        print(f"Initialized Q-table with shape {self.q_table.shape}")
        self.epsilon = 0.1

    def obs_to_state(self, obs):
        # from H*7 to H*3
        state = []  # Initialize as a list instead of a NumPy array
        global_att = obs[:,-1] # given that standalone attributes are at the end of the obs vector]
        obs = obs[:,:self.obs_space_shape - 1] # host-level observations are at the beginning of the obs vector
        print("Original obs shape", obs.shape, "Global attribute", global_att, "obs without global attribute shape", obs.shape)
        obs = obs.reshape(-1, self.obs_space_shape // 10)[:,:7]
        print("New shape", obs.shape)
        for i, host_obs in enumerate(obs): # for each host, compress to 3 attirbutes: 1) type, 2) on_host, 3) current state (sweeped, scanned, discovered, escalated, impacted)
            type_, sweeped, scanned, discovered, on_host, escalated, impacted = host_obs
            curr = 0 if sweeped == 1 else 1 if scanned == 1 else 2 if discovered == 1 else 3 if escalated == 1 else 4 if impacted == 1 else 0 if on_host == 1 else -1
            new_obs = [type_, on_host, curr]
            print(f"Host {i}: type={type_}, on_host={on_host}, curr={curr}")
            state.append(new_obs)
        state = np.array(state).flatten()
        state = np.concatenate((state, global_att)) 
        print("Compressed state shape", state.shape)
        return state
        
    def get_action_and_value(self, obs, action=None, action_mask=None):
        state = self.obs_to_state(obs)
        print("Converted obs of shape", obs.shape, "to state of shape", state.shape)
        logits = self.q_table[state]
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.get_value(state, action)
        print(f"Q-values: {logits}, Action chosen: {action}, Value: {value}")
        return action, value # probs.log_prob(action), probs.entropy(),
        
    def get_value(self, state, action):
        return max(self.q_table[state][action])

    def select_action(self, obs) -> int:
        state = self.obs_to_state(obs)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_space_shape)
        else:
            return self.greedy_action(state)
        
    # def get_feature_vector(self, obs):
    #     # Convert the observation to a feature vector (if needed)
    #     return obs.flatten()