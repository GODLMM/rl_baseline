import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np


def convert_to_torch(states):
    return torch.from_numpy(states).float().unsqueeze(0)


class CatActor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(CatActor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out = torch.tanh(self.fc1(inputs))
        out = self.fc2(out)
        return out

    def choose_action(self, inputs, return_dist=False):
        if type(inputs) is np.ndarray:
            inputs = convert_to_torch(inputs)
        probs = F.softmax(self.forward(inputs), dim=-1)
        dist = Categorical(probs)
        actions = dist.sample()
        if return_dist:
            return dist
        return actions, torch.exp(dist.log_prob(actions))


class NumActor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(NumActor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.mu = nn.Linear(hidden_size, output_size)
        self.sigma = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out = torch.tanh(self.fc1(inputs))
        mm = torch.tanh(self.mu(out))
        ss = F.softplus(self.sigma(out))
        return mm, ss

    def choose_action(self, inputs, return_dist=False):
        if type(inputs) is np.ndarray:
            inputs = convert_to_torch(inputs)
        mm, ss = self.forward(inputs)
        dist = Normal(mm, ss)
        if return_dist:
            return dist
        actions = dist.sample()
        return actions.clamp_(-1, 1), dist.log_prob(actions)


class Q_Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(Q_Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=-1)
        out = torch.tanh(self.fc1(inputs))
        out = self.fc2(out)
        return out


class V_Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(V_Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out = torch.tanh(self.fc1(inputs))
        out = self.fc2(out)
        return out
