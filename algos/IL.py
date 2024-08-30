import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_offsite = torch.tensor([0.94161665, 0.9616005 , 0.8307782 , 0.8431818 , 0.82132435,
        1.432366  , 0.94419135, 1.1348905]).to(device)
action_scale = torch.tensor([3.93085895, 4.4827265 , 4.1778248 , 3.8281758 , 3.85135935,
        4.5995945 , 4.09298865, 4.1333175]).to(device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return action_scale * torch.tanh(self.l3(a))+action_offsite

class IL(object):
    def __init__(self, args):
        self.actor = Actor(args.state_dim+args.h_state_dim, args.action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.1*args.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state=norm_state(state)
        return self.actor(state)

    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer 
        state, h_state, action,_, _, _, _, _, _ = replay_buffer.sample(batch_size)

        state=torch.cat([state, h_state],axis=1)
        state=norm_state(state)
        predict_action=self.actor(state)
        # action=action.clamp(-self.max_action, self.max_action)

        loss=F.mse_loss(predict_action,action)

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {'sl_loss':loss}

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + ".pth")
        

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + ".pth"))
        
    