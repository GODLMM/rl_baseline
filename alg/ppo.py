import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
from utils.model import CatActor, NumActor, V_Critic
from utils.memory import Memory
from torch.utils.tensorboard import SummaryWriter


def clip_gradient(opt, grad_clip):
    for group in opt.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class PPO:
    def __init__(
            self, n_states, n_actions, hidden_size=128, alr=2e-3, clr=2e-3,
            gamma=0.99, epochs=4, eps_clip=0.2, atype='cat'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.atype = atype
        self.memory = Memory()
        if atype == 'cat':
            self.actor = CatActor(n_states, n_actions, hidden_size)
        else:
            self.actor = NumActor(n_states, n_actions, hidden_size)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=alr)
        self.critic = V_Critic(n_states, 1, hidden_size)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=clr)
        self.MseLoss = torch.nn.MSELoss()
        self.step = 0

    def store_transition(self, s, a, r, s_, done, p):
        self.memory.states.append(s)
        self.memory.rewards.append(r)
        self.memory.actions.append(a)
        self.memory.probs.append(p)
        self.memory.next_states.append(s_)
        self.memory.is_terminals.append(done)

    def update(self, batch_size=None):
        if len(self.memory.actions) < batch_size: return
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
                self.memory.rewards[::-1], self.memory.is_terminals[::-1]):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).float()

        old_states = torch.FloatTensor(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_probs = torch.stack(self.memory.probs).detach()
        rewards = (rewards - rewards.mean()) / (rewards.std()+1e-7)

        split_res = self.memory.split(batch_size)

        for _ in range(self.epochs):
            for idxs in split_res:
                split_old_states = old_states[idxs[0]:idxs[1]]
                split_old_actions = old_actions[idxs[0]:idxs[1]]
                split_old_probs = old_probs[idxs[0]:idxs[1]]
                split_rewards = rewards[idxs[0]:idxs[1]]

                dist = self.actor.choose_action(split_old_states, True)
                log_probs = dist.log_prob(split_old_actions.squeeze())
                # diff = log_probs.squeeze() - split_old_probs.squeeze()
                # ratios = torch.exp(log_probs.squeeze()) / torch.exp(split_old_probs.squeeze())
                ratios = torch.exp(log_probs.squeeze() - split_old_probs.squeeze())

                state_values = self.critic(split_old_states).squeeze()
                advantages = split_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1-self.eps_clip, 1+self.eps_clip) * advantages
                aloss = -torch.min(surr1, surr2) - 0.01 * dist.entropy()
                
                self.actor_opt.zero_grad()
                aloss = aloss.mean()
                aloss.backward()
                clip_gradient(self.actor_opt, 0.1)
                writer.add_scalar('actor_loss', aloss.item(), self.step)
                self.actor_opt.step()

                closs = 0.5*self.MseLoss(split_rewards, state_values).mean()
                self.critic_opt.zero_grad()
                closs.backward()
                clip_gradient(self.critic_opt, 0.1)
                writer.add_scalar('critic_loss', closs.item(), self.step)
                self.critic_opt.step()
                self.step += 1



if __name__ == '__main__':
    import gym
    writer = SummaryWriter('./path/to/log')
    env = gym.make('CartPole-v0')
    # env = gym.make('Pendulum-v0')
    n_states = env.observation_space.shape[0]
    try:
        n_actions = env.action_space.shape[0]
        agent = PPO(n_states, n_actions, atype='num')   
    except:
        n_actions = env.action_space.n
        agent = PPO(n_states, n_actions, atype='cat')
    
    tt = 0
    for episode in range(1000):
        s = env.reset()
        ep_r = 0
        count = 0
        while count < 2001:
            if episode > 20:
                env.render()
            a, ap = agent.actor.choose_action(s)
            s_, r, done, info = env.step(a.detach().numpy()[0])
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            agent.store_transition(s, a, r, s_, done, ap)
            if tt % 200 == 0:
                agent.update(64)
                agent.memory.reset_memory()
            s = s_
            ep_r += r
            if done: break
            count += 1
            tt += 1
        print(episode, ': ', ep_r)
    writer.close()
        