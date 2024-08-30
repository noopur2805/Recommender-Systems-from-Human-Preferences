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
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return action_scale * torch.tanh(self.l3(a))+action_offsite


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		action=torch.clamp(((action-action_offsite)/action_scale),-1,1)
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		action=torch.clamp(((action-action_offsite)/action_scale),-1,1)
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(self, args):
		discount, tau, policy_noise, noise_clip, policy_freq, alpha = args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq, args.alpha
		self.actor = Actor(args.state_dim+args.h_state_dim, args.action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.1*args.lr)

		self.critic = Critic(args.state_dim+args.h_state_dim, args.action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

		
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0

		self.args=args

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		state=norm_state(state)
		return self.actor(state)


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

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
		
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + (noise*action_scale+action_offsite)
			)
			# perform clip
			next_action=torch.max(torch.min(next_action, action_offsite+action_scale), action_offsite-action_scale)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha

			actor_loss = -lmbda * Q.mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			return {'critic_loss':critic_loss, 'actor_loss':actor_loss}
		return {'critic_loss':critic_loss}

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)