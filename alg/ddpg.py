import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, atype='cat'):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.atype = atype

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out

    def choose_action(self, inputs):
        out = self.forward(inputs)
        if self.atype == 'cat':
            actions = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
        else:
            actions = out
        return actions, out


def actor_learn(eval_critic, eval_actor, opt, states, batch_size, n_actions, atype='cat'):
    actions = eval_actor.forward(states)
    loss = -torch.mean(eval_critic(states, actions))
    opt.zero_grad()
    loss.backward()
    clip_gradient(opt, 0.1)
    opt.step()


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=-1)
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def critic_learn(
        eval_critic, target_critic, target_actor, opt,
        states, actions, rewards, next_states, batch_size, n_actions, gamma=0.99, atype='cat'):
    next_actions = target_actor(next_states).view(batch_size, -1).detach()
    target_q = rewards + gamma * target_critic(next_states, next_actions).detach()
    eval_q = eval_critic(states, actions)
    loss_fn = nn.MSELoss()
    loss = loss_fn(target_q, eval_q)
    opt.zero_grad()
    loss.backward()
    clip_gradient(opt, 0.1)
    opt.step()


def soft_update(target_net, eval_net, tau):
    for target_param, eval_param in zip(target_net.parameters(), eval_net.parameters()):
        target_param.data.copy_(tau * eval_param.data + (1-tau) * target_param.data)
    return target_net


def clip_gradient(opt, grad_clip):
    for group in opt.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class Agent:
    def __init__(
            self, n_states, n_actions, capacity=10000, actor_lr=1e-3, critic_lr=1e-3,
            gamma=0.99, tau=0.02, batch_size=32, atype='cat'):
        # init network
        self.eval_actor = Actor(n_states, n_actions, atype=atype)
        self.target_actor = Actor(n_states, n_actions, atype=atype)
        self.eval_critic = Critic(n_states+n_actions, 1)
        self.target_critic = Critic(n_states+n_actions, 1)
        self.target_actor.load_state_dict(self.eval_actor.state_dict())
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        # init optimizer
        self.actor_opt = optim.Adam(self.eval_actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.eval_critic.parameters(), lr=critic_lr)

        self.buffer = []
        self.capacity = capacity
        self.gamma = gamma
        self.atype = atype
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

    def select_buffer(self):
        samples = random.sample(self.buffer, self.batch_size)
        return samples

    def store_transition(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def choose_action(self, state):
        if self.atype == 'cat':
            state = torch.unsqueeze(self.convert_to_tensor(state, torch.float), dim=0)
            action, out = self.eval_actor.choose_action(state)
            return action, out
        else:
            state = torch.unsqueeze(self.convert_to_tensor(state, torch.float), dim=0)
            action, out = self.eval_actor.choose_action(state)
            return action, out

    def convert_to_tensor(self, inputs, dtype):
        return torch.tensor(inputs, dtype=dtype)

    def learn(self):
        if len(self.buffer) < self.batch_size: return
        samples = self.select_buffer()
        states, actions, rewards, next_states = zip(*samples)


        states = self.convert_to_tensor(states, torch.float)
        actions = torch.vstack(actions).float().detach()
        rewards = self.convert_to_tensor(rewards, torch.float).view(self.batch_size, -1)
        next_states = self.convert_to_tensor(next_states, torch.float)

        critic_learn(
            self.eval_critic, self.target_critic, self.target_actor, self.critic_opt,
            states, actions, rewards, next_states, self.batch_size, self.n_actions, self.gamma, self.atype
        )
        actor_learn(self.eval_critic, self.eval_actor, self.actor_opt, states, self.batch_size, self.n_actions, self.atype)
        self.target_critic = soft_update(self.target_critic, self.eval_critic, self.tau)
        self.target_actor = soft_update(self.target_actor, self.eval_actor, self.tau)


if __name__ == '__main__':
    import gym

    # env = gym.make('CartPole-v0')
    env = gym.make('Pendulum-v0')
    try:
        n_actions = env.action_space.shape[0]
    except:
        n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    agent = Agent(n_states, n_actions, atype='num')

    for episode in range(5000):
        s = env.reset()
        ep_r = 0
        count = 0

        while True:
            if episode > 20:
                env.render()
            a, out = agent.choose_action(s)
            s_, r, done, info = env.step(a.detach().numpy()[0])
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            agent.store_transition(s, out, r, s_)
            agent.learn()
            s = s_
            ep_r += r
            if done: break
            count += 1
        print(episode, ': ', ep_r)
        