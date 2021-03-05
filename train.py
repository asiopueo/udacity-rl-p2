from unityagents import UnityEnvironment
import numpy as np

#################################
#  Initialization:
#################################
env = UnityEnvironment(file_name="./Reacher_Linux_1/Reacher.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

from agent import Agent
from collections import namedtuple

# Reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Define named tuple 'Experience'; you can use a dictionary alternatively
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Initialize the agent:
agent = Agent(buffer_size=10000, batch_size=64, gamma=0.98, epsilon=0.1, action_size=4)




####################################
#  Main learning loop:
####################################

# Initial values:
state = env_info.vector_observations[0] # Get the initial state
score = 0           
tick = 0

eps = 1.0
eps_rate = 0.995
eps_end = 0.02


#agent.load_weights("./checkpoints")

while tick < 200:
    # Select action according to policy:
    action = agent.action(state, eps, add_noise=True)

    print('Action taken: ', action, 'Time: ', tick)

    # Take action and record the reward and the successive state
    env_info = env.step(action)[brain_name]
    
    reward = env_info.rewards[0]
    next_state = env_info.vector_observations[0]
    done = env_info.local_done[0] # Not really relevant in this experiment as it runs 300 turns anyway

    # Add experience to the agent's replay buffer:
    exp = Experience(state, action, reward, next_state, done)
    agent.replay_buffer.insert_into_buffer( exp )
    
    agent.learn()

    score += reward
    state = next_state
    
    eps *= eps_rate


    if tick%10 == 0:
        print("[Time: {}] Time to update the target net.".format(tick))
    elif tick%50 == 0:
        print("[Time: {}] Score {}".format(tick, score))
        print("Buffer usage: {}".format(agent.replay_buffer.buffer_usage()))

    tick += 1


####################################
#  Debriefing:
####################################

print("")
print("Total score:", score)
agent.save_weights("./checkpoints")



