from unityagents import UnityEnvironment
import numpy as np

#################################
#  Initialization:
#################################
env = UnityEnvironment(file_name="./Reacher_Linux_1/Reacher.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#from agent import Agent
from collections import namedtuple, deque
import time

# Reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Initialize the agent:
from agent_torch import Agent
agent = Agent(buffer_size=10000, batch_size=64, gamma=0.98, epsilon=0.01, action_size=4)


####################################
#  Replay loop:
####################################





def replay(n_episodes=200):
    score_list = []
    score_trailing_list = deque(maxlen=100)
    score_trailing_avg_list = []

    agent.load_weights("./checkpoints_torch")

    for episode in range(0, n_episodes):
        ticks = 0
        scores = np.zeros( shape=(num_agents,) )

        env_info = env.reset(train_mode=False)[brain_name]   # Reset the environment
        states = env_info.vector_observations                # Get the current state

        start = time.time()
        while True:
            # Select action according to policy:
            actions, _ = agent.action(states, add_noise=False)

            # Take action and record the reward and the successive state
            env_info = env.step(actions)[brain_name]
            
            rewards = np.array(env_info.rewards)
            next_states = env_info.vector_observations
            dones = env_info.local_done

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
  
        episode += 1

    return score_list, score_trailing_avg_list


replay()


env.close()

