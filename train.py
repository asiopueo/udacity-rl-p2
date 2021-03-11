from unityagents import UnityEnvironment
import numpy as np
from collections import namedtuple, deque
import time



#################################
#  Initialization:
#################################

# Select environment either with one or 20 reacher arms
env = UnityEnvironment(file_name="./Reacher_Linux_1/Reacher.x86_64")
#env = UnityEnvironment(file_name="./Reacher_Linux_20/Reacher.x86_64")

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

def training(n_episodes=500):
    score = 0           
    tick = 0

    score_list = []
    score_trailing_list = deque(maxlen=10)
    score_trailing_avg_list = []

    eps = 1.0
    eps_rate = 0.995
    eps_end = 0.02

    #agent.load_weights("./checkpoints")

    for episode in range(0, n_episodes):
        ticks = 0
        score = 0

        env_info = env.reset(train_mode=True)[brain_name]   # Reset the environment
        state = env_info.vector_observations                # Get the current state

        start = time.time()
        while True:
            # Select action according to policy:
            action = agent.action(state, eps, add_noise=True)

            # Take action and record the reward and the successive state
            env_info = env.step(action)[brain_name]
            
            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            done = env_info.local_done[0]

            # Add experience to the agent's replay buffer:
            exp = Experience(state, action, reward, next_state, done)
            agent.replay_buffer.insert_into_buffer( exp )
            
            agent.learn()

            score += reward
            state = next_state
            
            eps = max( eps_rate*eps, eps_end )

            if done is True:
                break

            ticks += 1


        end = time.time()

        score_list.append(score)
        score_trailing_list.append(score)
        score_trailing_avg = np.mean(score_trailing_list)
        score_trailing_avg_list.append(score_trailing_avg)

        print("***********************************************")
        print("Score of episode {}: {}".format(episode, score))
        print("Trailing avg. score: {:.2f}".format(score_trailing_avg))
        print("Greedy epsilon used: {}".format(eps))
        print("Time consumed: {:.2f} s".format(end-start))
        print("***********************************************")

        #agent.save_weights("./checkpoints")
        episode += 1

    return score_list, score_trailing_avg_list



training(50)



env.close()


