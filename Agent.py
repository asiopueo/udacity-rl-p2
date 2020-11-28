from collections import deque
import random
import numpy as np
from collections import namedtuple

import Network




class Agent():
    def __init__(self, buffer_size, batch_size, gamma, action_size):
        if not batch_size < buffer_size:
            raise Exception()

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.action_size = action_size

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

        # Initialize noise
        self.noise = Noise()

        # Seed the random number generator
        random.seed()
        # QNetwork - We choose the simple network
        self.actor_local = Network.network_actor()
        self.actor_target = Network.network_actor()

        self.critic_local = Network.network_critic()
        self.critic_target = Network.network_critic()

    # Let the agent learn from experience
    # Utilizing Deep Deterministic Policy Gradient methodology (DDPG):
    def learn(self):
        """
        Q() = reward(s_t,a_t) + gamma * critic(s_{t+1},a_{t+1})
        """
        # Retrieve batch of experiences from the replay buffer:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_from_buffer()

        Q_target = self.critic_local.predict( state_batch, action_batch )

        X = []
        y = []

        # Batches need to be prepared before learning
        for index, state in enumerate(state_batch):    
            Q_exp = self.critic_local(state_batch, action_batch)
            
            next_action = self.actor_local(state_batch)
            if not done_batch[index]:
                Q_target = reward_batch[index] + self.gamma * self.critic_target(next_state, next_action)
            else:
                Q_target = reward_batch[index]

            print(action_batch[index].shape) # (4,)
            #Q_target[index, action_batch[index]] = Q_new

            X.append(Q_exp[index])
            y.append(Q_target)

        X_policy_action = np.array(X)
        X_noise_action = np.hstack()
        Q_target = np.array(Q_target)

        print("X_policy_action.shape: ", X_policy_action.shape)
        print("X_noise_action.shape: ", X_noise_action.shape)
        print("Q_target.shape: ", Q_target.shape)

        # Train the actor network
        self.actor_local.fit(X_policy_action, None, batch_size=self.batch_size, epochs=1, shuffle=False, verbose=1)

        # Train the critic network
        # This one is more straightforward
        self.critic_local.fit(X_noise_action, Q_target, batch_size=self.batch_size, epochs=1, shuffle=False, verbose=1)

        # Soft updates:
        #
        



    # Take action according to epsilon-greedy-policy:
    def action(self, state, epsilon=0.9):
               
        if random.random() > epsilon:
            return 2 * np.random.random_sample(self.action_size) - 1.0
        else:
            return self.actor_local(state)
        
        prob_distribution = self.actor_local.predict(state.reshape(1,-1))
        action = np.argmax(prob_distribution)
        return action

    # Copy weights from short-term model to long-term model
    def update_target_nets(self, tau=0.1):
        # Implement soft update for later:
        # get_weights()[0] -- weights
        # get weights()[1] -- bias (if existent)
        # Soft-update:
        actor_weights_local = np.array( self.actor_local.get_weights() )
        actor_weights_target = np.array( self.actor_target.get_weights() )
        self.actor_target.set_weights( tau*actor_weights_local + (1-tau)*actor_weights_target )

        critic_weights_local = np.array( self.critic_local.get_weights() )
        critic_weights_target = np.array( self.critic_target.get_weights() )
        self.critic_target.set_weights( tau*critic_weights_local + (1-tau)*critic_weights_target )



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
        state = np.vstack( [exp.state for exp in batch] )
        action = np.vstack( [exp.action for exp in batch] )
        reward = np.vstack( [exp.reward for exp in batch] )
        state_next = np.vstack( [exp.next_state for exp in batch] )
        done = np.vstack( [exp.done for exp in batch] )

        return state, action, reward, state_next, done

    # Get length of memory
    def buffer_usage(self):
        return len(self.replay_buffer) > self.batch_size



class Noise():
    def __init__():
        pass

    def sample():
        pass