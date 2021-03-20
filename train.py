from unityagents import UnityEnvironment
import numpy as np
from collections import namedtuple, deque
import time



#################################
#  Initialization:
#################################

# Select environment either with one or 20 reacher arms
#env = UnityEnvironment(file_name="./Reacher_Linux_1/Reacher.x86_64")
env = UnityEnvironment(file_name="./Reacher_Linux_20/Reacher.x86_64")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]

# Number of agents (should be 1 or 20)
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# Size of each action (should be 4)
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])



# Initialize the agent:
from agent_torch import Agent
agent = Agent(buffer_size=10000, batch_size=64, gamma=0.98, epsilon=0.1, action_size=4)




####################################
#  Main learning loop:
####################################

def training(n_episodes=200):
    tick = 0
    #eps = 1. # eps is only defined as info

    success = False # Flag which will be triggered when challenge is solved

    score_list = []
    score_trailing_list = deque(maxlen=10)
    score_trailing_avg_list = []

    #agent.load_weights("./checkpoints_torch")

    for episode in range(0, n_episodes):
        ticks = 0
        scores = np.zeros( shape=(num_agents,) )

        env_info = env.reset(train_mode=True)[brain_name]   # Reset the environment
        states = env_info.vector_observations                # Get the current state

        start = time.time()
        while True:
            # Select action according to policy:
            actions, _ = agent.action(states, add_noise=True)

            # Take action and record the reward and the successive state
            env_info = env.step(actions)[brain_name]
            
            rewards = np.array(env_info.rewards)
            next_states = env_info.vector_observations
            dones = env_info.local_done

            # Add experience to the agent's replay buffer:
            for idx in range(num_agents):
                exp = Experience(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])
                agent.replay_buffer.insert_into_buffer( exp )
            
            agent.learn()

            scores += rewards
            states = next_states
            
            if np.any(dones):
                break

            ticks += 1


        end = time.time()

        score = np.mean(scores)
        score_list.append(score)
        score_trailing_list.append(score)
        score_trailing_avg = np.mean(score_trailing_list)
        score_trailing_avg_list.append(score_trailing_avg)

        print("***********************************************")
        print("Score of episode {}: {}".format(episode, score))
        print("Trailing avg. score: {:.2f}".format(score_trailing_avg))
        print("Time consumed: {:.2f} s".format(end-start))
        print("***********************************************")
        
        if score_trailing_avg > 30.0 and success is False:
            print("===============================================")
            print("Challenge solved at episode {}".format(episode))
            print("===============================================")
            success = True

        episode += 1

        if episode % 10 == 0:
            agent.save_weights("./checkpoints_torch")

    return score_list, score_trailing_avg_list



training(50)



env.close()


