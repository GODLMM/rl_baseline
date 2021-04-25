import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym


# hyperparameter
batch_size = 32
lr = 0.01
epsilon = 0.9
gamma = 0.9
target_replace_iter = 100
memory_capacity = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
env_a_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)


class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, n_states*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if env_a_shape == 0 else action.reshape(env_a_shape)
        else:
            action = np.random.randint(0, n_actions)
            action = action if env_a_shape == 0 else action.reshape(env_a_shape)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :n_states])
        b_a = torch.LongTensor(b_memory[:, n_states:n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, n_states+1:n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, n_states+2:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_select = torch.unsqueeze(self.eval_net(b_s_).argmax(1), dim=1)
        q_next = self.target_net(b_s_).detach()
        q_select = torch.zeros_like(q_next).scatter_(1, q_select, 1)
        q_target = b_r + torch.sum(gamma*(q_next*q_select), dim=1, keepdim=True)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()


for i_episode in range(1000):
    s = env.reset()
    ep_r = 0
    while True:
        if i_episode > 300:
            env.render()
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        dqn.store_transition(s, a, r, s_)
        ep_r += r
        if dqn.memory_counter > memory_capacity:
            dqn.learn()
        if done:
            if i_episode % 50 == 0:
                print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))
            break
        s = s_