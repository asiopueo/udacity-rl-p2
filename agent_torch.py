from collections import deque
import random
import copy
from collections import namedtuple
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from networks_torch import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Experimental values:
EPS_END = 0.05
EPS_DECAY = 3e-5

class Agent():
    def __init__(self, buffer_size, batch_size, action_size, gamma, epsilon=1.0, learn_rate=0.0005):
        if not batch_size < buffer_size:
            raise Exception()

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps = epsilon

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

        # Initialize noise
        #self.noise = OUNoise()
        self.noise = Noise()
        self.epsilon = epsilon
        self.learn_rate = learn_rate

        # Seed the random number generator
        random.seed()
        np.random.seed()

        seed = 0

        self.actor_local = Actor(seed).to(device)
        self.actor_target = Actor(seed).to(device)

        self.critic_local = Critic(seed).to(device)
        self.critic_target = Critic(seed).to(device)

        self.hard_update_target_nets()

        self.actor_optimizer = optim.Adam( self.actor_local.parameters(), lr=self.learn_rate )
        self.critic_optimizer = optim.Adam( self.critic_local.parameters(), lr=self.learn_rate )


    # Let the agent learn from experience
    # Utilizing Deep Deterministic Policy Gradient methodology (DDPG):
    def learn(self):
        # If buffer is sufficiently full, let the agent learn from his experience:
        # Put the learning procedures into the main loop below!
        if not self.replay_buffer.buffer_usage():
            return
        
        # Retrieve batch of experiences from the replay buffer:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_from_buffer()
        # Train the critic network
        # Q(s_t,a_t) = reward(s_t,a_t) + gamma * critic(s_{t+1},a_{t+1})
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0) #clip the gradient for the critic network (Udacity hint)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft updates:
        self.soft_update_target_nets(tau=0.001)
        

    # Take action according to epsilon-greedy-policy:
    def action(self, state, eps=1., add_noise=True):
        # Sample action from actor network:
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise and random.random() < eps:
            actions += self.noise.sample()
            self.eps = max(self.eps-EPS_DECAY, EPS_END)

        actions = np.clip(actions, -1, 1)
        return actions, self.eps # Returns used epsilon as info

    # Generates an array of action_size (i.e. 4) with uniformly distributed floats in [-1,1)
    def random_action(self):
        action = 2 * np.random.random_sample(self.action_size) - 1.0
        return action

    # Copy weights from short-term model to long-term model
    def soft_update_target_nets(self, tau=0.001):
        for t, l in zip(self.actor_target.parameters(), self.actor_local.parameters() ):
            t.data.copy_( (1-tau)*t.data + tau*l.data )

        for t, l in zip(self.critic_target.parameters(), self.critic_local.parameters() ):
            t.data.copy_( (1-tau)*t.data + tau*l.data )

    def hard_update_target_nets(self):
        self.soft_update_target_nets( tau=1.0 )


    def load_weights(self, path):
        filepath = os.path.join(path, "actor_weights_latest.pth")
        print("Loading actor network weights from", filepath)
        self.actor_local.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

        filepath = os.path.join(path, "critic_weights_latest.pth")
        print("Loading critic network weights from", filepath)
        self.critic_local.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        
        self.hard_update_target_nets()


    def save_weights(self, path):
        filepath = os.path.join(path, "actor_weights_latest.pth")
        print("Saving actor network weights to", filepath)
        torch.save(self.actor_net.state_dict(), filepath) 
        filepath = os.path.join(path, "critic_weights_latest.pth")
        print("Saving critic network weights to", filepath)
        torch.save(self.critic_net.state_dict(), filepath) 


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=self.buffer_size)

    # Insert experience into memory
    def insert_into_buffer(self, experience):
        self.replay_buffer.append(experience)

    # Randomly sample memory
    def sample_from_buffer(self):
        # Sample experience batch from experience buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Reorder experience batch such that we have a batch of states, a batch of actions, a batch of rewards, etc.
        # Eventually add 'if exp is not None'
        state = torch.from_numpy( np.vstack( [exp.state for exp in batch] ) ).float().to(device)
        action = torch.from_numpy( np.vstack( [exp.action for exp in batch] ) ).float().to(device)
        reward = torch.from_numpy( np.vstack( [exp.reward for exp in batch] ) ).float().to(device)
        state_next = torch.from_numpy( np.vstack( [exp.next_state for exp in batch] ) ).float().to(device)
        done = torch.from_numpy( np.vstack( [exp.done for exp in batch] ).astype(np.uint8) ).float().to(device)

        return state, action, reward, state_next, done

    # Get length of memory
    def buffer_usage(self):
        return len(self.replay_buffer) > self.batch_size


# Normally distributed noise
class Noise():
    def __init__(self, mu=0., sigma=0.2):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        #Todo: Sample from normal distribution
        return np.random.randn() # Shall be refined later!


# Ornstein-Uhlenbeck process:
class OUNoise():
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.seed = random.seed(seed)
        self.mu = mu + np.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state += dx
        return self.state
