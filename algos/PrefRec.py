import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import process_reward, norm_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_offsite = torch.tensor([0.94161665, 0.9616005 , 0.8307782 , 0.8431818 , 0.82132435,
        1.432366  , 0.94419135, 1.1348905 ]).to(device)
action_scale = torch.tensor([3.93085895, 4.4827265 , 4.1778248 , 3.8281758 , 3.85135935,
        4.5995945 , 4.09298865, 4.1333175]).to(device)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		
	def forward(self, state):
		# assert torch.isnan(state).int().sum().item()==0, (torch.isnan(state)==True).nonzero(as_tuple=False)	
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return action_scale * torch.tanh(self.l3(a))+action_offsite
		


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256 + action_dim, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		# assert torch.isnan(state).int().sum().item()==0 
		# assert torch.isnan(action).int().sum().item()==0

		action=torch.clamp(((action-action_offsite)/action_scale),-1,1)
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)

class V_net(nn.Module):
	def __init__(self, state_dim):
		super(V_net, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.l3(a)

class PrefRec(object):
	def __init__(self, args):
		discount, tau=args.discount, args.tau
		self.actor = Actor(args.state_dim+args.h_state_dim, args.action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.1*args.lr)

		self.critic = Critic(args.state_dim+args.h_state_dim, args.action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=args.lr, weight_decay=1e-2)

		self.V_net = V_net(args.state_dim+args.h_state_dim).to(device)
		self.V_optimizer = torch.optim.Adam(self.V_net.parameters(), lr=args.lr)

		self.discount = discount
		self.tau = tau
		self.alpha=args.alpha

		self.expectile=args.expectile
		
		self.args=args

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		state=norm_state(state)
		return self.actor(state)


	def train(self, replay_buffer, batch_size=64):
		# Sample replay buffer 
		state, h_state, action,next_action, next_state, next_h_state, response, h_response, not_done = replay_buffer.sample(batch_size)
		
		#reward = process_reward(h_response, response)
		
		state=torch.cat([state, h_state],axis=1)
		next_state=torch.cat([next_state, next_h_state],axis=1)

		state=norm_state(state)
		next_state=norm_state(next_state)

		if not self.args.true_reward:
			reward=self.reward_model.relabel(state, action) # action is not normalized
		else:
			reward = process_reward(h_response, self.args.return_type)
		
		# Compute the target Q value
		target_Q = self.critic_target(state, action)
		predict_V=self.V_net(state)
		u=target_Q-predict_V # (batch, 1)
		V_loss=(torch.abs(self.expectile-torch.le(u,0).float())*(u**2)).mean()

		self.V_optimizer.zero_grad()
		V_loss.backward()
		self.V_optimizer.step()


		target_Q = reward + (not_done * self.discount * self.V_net(next_state)).detach()
		current_Q = self.critic(state, action)
		critic_loss=F.mse_loss(target_Q, current_Q)
        
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss

		actor_loss = -self.critic(state, self.actor(state)).mean()
		actor_loss=actor_loss*self.alpha
		
		# Optimize the actor 	
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()): # not use
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return {'critic_loss':critic_loss, 'actor_loss':actor_loss,'v_loss':V_loss}
		
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		torch.save(self.V_net.state_dict(), filename + "_v_net")
		torch.save(self.V_optimizer.state_dict(), filename + "_v_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

		self.V_net.load_state_dict(torch.load(filename + "_v_net"))
		self.V_optimizer.load_state_dict(torch.load(filename + "_v_optimizer"))

	def load_actor(self, filename):
		self.actor.load_state_dict(torch.load(filename + ".pth"))