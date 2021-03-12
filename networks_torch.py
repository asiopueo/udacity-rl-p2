import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# The network is used to approximate a POLICY. As a result, we suggest the following structure:
# 
# Since the observation space consists of 33 variables corresponding to position, rotation, 
# velocity, and angular velocities of the arm, the input layer shall consist of 33 neurons
#
# The output layer consists of two neurons for four floating point numbers
# between -1 and 1 which represent the torque applied to the two joints of
# the robot arm
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, seed, state_size=33, action_size=4, units_fc1 = 256, units_fc2 = 128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, units_fc1)
        self.fc2 = nn.Linear(units_fc1, units_fc2)
        self.fc3 = nn.Linear( units_fc2, action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2( x ))
        action = torch.tanh( self.fc3(x) )
        return action



class Critic(nn.Module):
    def __init__(self, seed, state_size=33, action_size=4, units_fc1 = 256, units_fc2 = 128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, units_fc1)
        self.fc2 = nn.Linear(units_fc1+action_size, units_fc2)
        self.fc3 = nn.Linear(units_fc2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.elu( self.fc1(state) )
        x = torch.cat( (x, action), dim=1 )
        x = F.elu( self.fc2(x) )
        value = self.fc3(x)
        return value

