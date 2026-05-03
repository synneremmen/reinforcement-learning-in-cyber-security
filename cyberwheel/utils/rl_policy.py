from collections import defaultdict
import copy
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
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)
    

class RLPolicyParameterized(nn.Module):
    def __init__(self, action_space_shape=0, obs_space_shape=0, epsilon=0.2, use_target=True, eval=False, hidden_layers=None):
        super().__init__()
        obs_space_shape = int(np.array(obs_space_shape).prod())
        print(f"Initializing parameterized policy with obs space shape {obs_space_shape} and action space shape {action_space_shape}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.epsilon = epsilon
        self.use_target = use_target if use_target is not None else False
        self.hidden_layers = list(hidden_layers) if hidden_layers is not None else [64, 64]
        self.increased_depth = len(self.hidden_layers) > 2
        self.old_action_space_shape = self.hidden_layers[-1] if self.increased_depth else None
        self.model = self._build_model(self.hidden_layers)
        if self.use_target:
            self.tau = 0.005
            self.set_target_model()

    def _build_model(self, hidden_layers):
        # given hidden_layers, appends layers and returns sequential model
        layers = []
        in_features = self.obs_space_shape
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        layers.append(nn.Linear(in_features, self.action_space_shape))
        return nn.Sequential(*layers).to(self.device)

    def _set_hidden_layers(self, hidden_layers):
        # set self.hidden_layers to list given, increased depth true if more than two hidden layers
        # old action shape is the last hidden layer if increased depth
        self.hidden_layers = list(hidden_layers)
        self.increased_depth = len(self.hidden_layers) > 2
        self.old_action_space_shape = self.hidden_layers[-1] if self.increased_depth else None

    def architecture_metadata(self):
        # save action and obs shape, pluss hidden layers
        return {
            "obs_space_shape": self.obs_space_shape,
            "action_space_shape": self.action_space_shape,
            "hidden_layers": self.hidden_layers,
        }

    @staticmethod
    def hidden_layers_from_state_dict(state_dict):
        # get hidden layers by extracting shape of weights for each layer in the state dict
        layer_indices = sorted(
            int(key.split(".")[1])
            for key in state_dict.keys()
            if key.startswith("model.") and key.endswith(".weight")
        )
        hidden_layers = [state_dict[f"model.{index}.weight"].shape[0] for index in layer_indices[:-1]]
        return hidden_layers

    def soft_update(self):
        # slowly move to new model from old model by updating with 1-tau of target, and tau of new model
        # avoid "moving target"
        for tp, sp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def get_value(self, obs, action=None, action_mask=None, use_target=False):
        """Get Q-values using either main or target network. use_target=True for bootstrapping in DQN."""
        model = self.target_model if use_target and self.use_target else self.model
        q_values = model(obs)
        if action_mask is not None:
            q_values = q_values.clone()
            q_values[~torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)] = float("-inf")
        if action is None:
            values = torch.max(q_values, dim=1).values.reshape(-1, 1)
        else:
            values = q_values.gather(1, action.view(-1, 1))
        if float("-inf") in values:
            # i.e. action nothing will always be available so this should never be the case
            raise ValueError("All Q-values are -inf for the given action mask.")

        return values
    
    def get_probabilities(self, obs, mode="teacher", mapping=None, num_hosts=None, expand_teacher_probs=False):
        q_values = self.model(obs)
        temperature = 2.0
        q_values = q_values / temperature
        if mode == "teacher":
            probs = torch.nn.functional.softmax(q_values, dim=1)
        elif mode == "student":
            if expand_teacher_probs:
                probs = torch.nn.functional.log_softmax(q_values, dim=1)
            else:
                new_q_values = []
                if mapping is None or num_hosts is None:
                    raise ValueError("Mapping and number of hosts must be provided for student mode.")
                idx = 0
                for _ in range(num_hosts):
                    for count in mapping.values():
                        sum_probs = torch.sum(q_values[:, idx : idx + count], dim=1)
                        new_q_values.append(sum_probs)
                        idx += count
                new_q_values.append(q_values[:, -1]) 
                new_q_values = torch.stack(new_q_values, dim=1)
                probs = torch.nn.functional.log_softmax(new_q_values, dim=1)
        return probs
    
    def get_action_and_value(self, obs, action=None, action_mask=None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits = self.model(obs)
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, float("-inf"))

        if action is None:
            probs = Categorical(logits=logits)
            action = probs.sample() # randomization
            
        value = self.get_value(obs, action)
        return action, None, None, value
    

    def select_action(self, obs, action_mask):
        """Select action using epsilon-greedy with action mask"""
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        if np.random.uniform() < self.epsilon:
            valid_actions = torch.where(mask)[0]
            if len(valid_actions) > 0:
                random_idx = torch.randint(0, len(valid_actions), (1,), device=self.device)
                return valid_actions[random_idx].reshape(-1)
            else:
                return torch.zeros(1, dtype=torch.long, device=self.device)
        else:
            return self.greedy_action(obs, mask)
    
    def greedy_action(self, obs, action_mask=None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits = self.model(obs)
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, float("-inf"))
        greedy_action = torch.argmax(logits, dim=1)
        return greedy_action
    
    def copy_params(self, old_policy, mapping=None):
        """Copy the weights from the old policy to the new policy, using the mapping to determine which weights correspond to which actions."""
        if mapping is None:
            raise ValueError("Mapping must be provided for copy method.")
        
        new_model = self._build_model([64, 64])
        values = list(mapping.values())
        with torch.no_grad():
            for old_layer, new_layer in zip(old_policy.model, new_model):
                out_idx = 0
                # create new weigths with new_action_space_shape and divide the old weights on the corrospending features in the new weight matrix
                if new_layer.out_features == self.action_space_shape:
                    # only last action layer changes
                    for i in range(len(values)):
                        curr_count = values[i]
                        end_idx = out_idx + curr_count
                        # action shape x 64
                        new_weights = old_layer.weight[i, :] / curr_count # get old weight
                        new_bias = old_layer.bias[i] # get old bias
                        new_layer.weight[out_idx:end_idx, :].copy_(new_weights.repeat(curr_count, 1))
                        new_layer.bias[out_idx:end_idx].copy_(new_bias)
                        out_idx = end_idx
                    # last layer, output layer that we want to expand
                else:
                    # copy weights and biases for the overlapping part of the layers
                    new_layer.weight.copy_(old_layer.weight)
                    new_layer.bias.copy_(old_layer.bias)
        return new_model

    def increase_depth(self, old_policy=None, reuse_model=True):
        """Increase the depth of the model by adding one hidden layer sized to the old action space."""
        # add a hidden layer of size of the old policy
        self._set_hidden_layers([64, 64, old_policy.action_space_shape])
        new_model = self._build_model(self.hidden_layers)
        if reuse_model:
            with torch.no_grad():
                for old_layer, new_layer in zip(old_policy.model, new_model):
                    if isinstance(old_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                        new_layer.weight.copy_(old_layer.weight)
                        new_layer.bias.copy_(old_layer.bias)
            print("Increased depth and reused model weights for layers with matching shapes.")
        else:
            print("Increased depth and did NOT reuse model weights.")
        print("Hidden layers increased from ", old_policy.hidden_layers, " to ", self.hidden_layers)
        return new_model

    def kl_divergence(self, log_probs_student, probs_teacher):
        """Calculate the KL divergence between the teacher and student model's action distributions."""
        return nn.KLDivLoss(reduction='batchmean')(log_probs_student, probs_teacher)
    
    def set_target_model(self):
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

class RLPolicyTabular(nn.Module): 
    """
    TODO: Add header∂s
    """
    def __init__(self, action_space_shape=0, obs_space_shape=0, args=None):
        super().__init__()
        obs_space_shape = int(np.array(obs_space_shape).prod())
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.device = args.device
        self.args = args
        # self.state_space = (1 + 3 * 2**6)**self.num_hosts
        # self.compressed_state_space = (1 + 3 * 2*4)**self.num_hosts # given that we compress the 7 host-level attributes to 3, and keep the global attribute, we get 1 + 3*6 possible combinations per host
        #print(f"Size of q_table: {self.compressed_state_space} x {self.action_space_shape} = {self.compressed_state_space * self.action_space_shape} entries")
        self.q_table = defaultdict(self._new_q_values)
        # default to random values [0,1] to encourage exploration of unseen actions?
        print(f"Initialized Q-table with shape {len(self.q_table)} x {self.action_space_shape} = {len(self.q_table) * self.action_space_shape} entries")
        if not args.evaluation:
            self.epsilon = args.epsilon
            self.learning_rate = args.learning_rate
            self.initial_lr = args.learning_rate
            self.decay_power = getattr(args, "decay_power", 0.001)
            self.total_updates = getattr(args, "num_updates", getattr(args, "total_timesteps", 1))
            self.update = 0
        self.num_hosts = getattr(args, "num_hosts", args.max_num_hosts)

    def decay_lr(self):
        # linear decay
        self.learning_rate = self.initial_lr / (1 + self.decay_rate * self.update)
        # progress = min(self.update, self.total_updates) / max(self.total_updates, 1)
        # self.learning_rate = self.initial_lr * (1 - progress) ** self.decay_power

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

    def _new_q_values(self):
        return torch.rand(self.action_space_shape, device=self.device)

    def _get_state_q(self, obs):
        obs = self.obs_to_state(obs)
        q_values = self.q_table.get(obs)
        if q_values is None:
            q_values = self._new_q_values()
            self.q_table[obs] = q_values
        return q_values
    
    def get_action_and_value(self, obs, action=None, action_mask=None):
        if action is None:
            action = self.greedy_action(obs, action_mask)
        value = self.get_value(obs, action)
        return action, value , None, None
        
    def get_value(self, obs, action): # best action value for a given state
        return self._get_state_q(obs)[int(action)].item() 
    
    def get_best_value(self, obs, action_mask=None):
        q_values = self._get_state_q(obs)
        if action_mask is not None:
            q_values = q_values.clone()
            q_values[~torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)] = float("-inf")
        best_value = torch.max(q_values).item()
        return best_value
    
    def greedy_action(self, obs, action_mask=None):
        action_values = self._get_state_q(obs)
        if action_mask is not None:
            action_values = action_values.clone()
            action_values[~torch.as_tensor(action_mask, dtype=torch.bool, device=action_values.device)] = float("-inf")
        greedy_action = torch.argmax(action_values)
        return greedy_action.item()
    
    def select_action(self, obs, action_mask):
        """Select action using epsilon-greedy with action mask"""
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        if np.random.uniform() < self.epsilon:
            valid_actions = torch.where(mask)[0]
            if len(valid_actions) > 0:
                random_idx = torch.randint(0, len(valid_actions), (1,), device=self.device)
                return int(valid_actions[random_idx].item())
            else:
                return 0
        else:
            return self.greedy_action(obs, mask)

    def update_q_table(self, obs, action, reward, next_obs, done, next_action_mask=None, alpha=0.1, gamma=0.99):
        """Update Q-table using Q-learning update rule"""
        done = bool(torch.as_tensor(done).item())
        action_idx = int(torch.as_tensor(action).item())
    
        if done:
            td_target = reward
        else:
            best_next_q = self.get_best_value(next_obs, next_action_mask)
            td_target = reward + gamma * best_next_q
        
        td_error = td_target - self.get_value(obs, action_idx)
        update_value = self.learning_rate * td_error

        self._get_state_q(obs)[action_idx] += update_value
        self.update += 1
        # self.decay_lr()
        return float(td_error)
        # q(s,a) = q(s,a) + alpha(R + gamma maxq(nexts,a) - q(s,a))
        # q(s,a) = q(s,a) + alpha * td_error
        # der td_error = td_target - get_value(s,a)
        # og td_target = R + gamma * best_next_q