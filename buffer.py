import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, args):
		
		self.max_size = args.max_buffer_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((self.max_size, args.state_dim))
		self.h_state = np.zeros((self.max_size, args.h_state_dim))
		self.action = np.zeros((self.max_size, args.action_dim))
		self.next_action = np.zeros((self.max_size, args.action_dim))
		self.next_state = np.zeros((self.max_size, args.state_dim))
		self.next_h_state = np.zeros((self.max_size, args.h_state_dim))
		self.response = np.zeros((self.max_size, args.response_dim))
		self.h_response = np.zeros((self.max_size, args.h_response_dim))
		self.not_done = np.zeros((self.max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, h_state, action, next_action, next_state, next_h_state, response, h_response, done):
		num=len(state)
		end_ptr=self.ptr+num
		self.state[self.ptr:end_ptr] = state
		self.h_state[self.ptr:end_ptr] = h_state
		self.action[self.ptr:end_ptr] = action
		self.next_action[self.ptr:end_ptr] = next_action
		self.next_state[self.ptr:end_ptr] = next_state
		self.next_h_state[self.ptr:end_ptr] = next_h_state
		self.response[self.ptr:end_ptr] = response
		self.h_response[self.ptr:end_ptr] = h_response
		self.not_done[self.ptr:end_ptr] = 1. - np.array(done)

		self.ptr = (self.ptr + num) % self.max_size
		self.size = min(self.size + num, self.max_size)
		# if self.size==self.max_size:
		# 	print("Replay buffer is full!")
		assert self.size<=self.max_size

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.h_state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.next_h_state[ind]).to(self.device),
			torch.FloatTensor(self.response[ind]).to(self.device),
			torch.FloatTensor(self.h_response[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def normalize_states(self, eps = 1e-6):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		# self.state = (self.state - mean)/std
		# self.next_state = (self.next_state - mean)/std
		return mean, std
	
	def stat_actions(self, eps= 1e-3):
		mean = self.action.mean(0,keepdims=False)
		std= self.action.std(0,keepdims=False) + eps
		max_a=self.action.max(axis=0)
		min_a=self.action.min(axis=0)
		return mean.tolist(), std.tolist(), max_a.tolist(), min_a.tolist()

	def normal_actions(self, eps= 1e-3, mean=None, std=None):
		if mean is None:
			mean = self.action.mean(0,keepdims=True)
		if std is None:
			std= self.action.std(0,keepdims=True) + eps
		self.action=(self.action-mean)/std

	def save(self, path):
		np.savez(path, state=self.state[:self.size], h_state=self.h_state[:self.size], action=self.action[:self.size],next_action=self.next_action[:self.size], next_state=self.next_state[:self.size], next_h_state=self.next_h_state[:self.size], response=self.response[:self.size], h_response=self.h_response[:self.size], not_done=self.not_done[:self.size])

	def load(self, path):
		stored_array=np.load(path)
		self.size=len(stored_array["state"])
		self.state = stored_array["state"]
		self.h_state = stored_array["h_state"]
		self.action = stored_array["action"]
		self.next_action = stored_array["next_action"]
		self.next_state = stored_array["next_state"]
		self.next_h_state = stored_array["next_h_state"]
		self.response = stored_array["response"]
		self.h_response = stored_array["h_response"]
		self.not_done = stored_array["not_done"]


class PrefBuffer(object):
	def __init__(self, args):
		self.max_size = args.pref_buffer_size
		self.ptr = 0
		self.size = 0

		self.seg1=np.zeros([self.max_size, args.seg_length, args.state_dim+args.h_state_dim+args.action_dim])
		self.seg2=np.zeros([self.max_size, args.seg_length, args.state_dim+args.h_state_dim+args.action_dim])
		self.label=np.zeros([self.max_size, 3])

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.seg1[ind]).to(self.device),
			torch.FloatTensor(self.seg2[ind]).to(self.device),
			torch.LongTensor(self.label[ind]).to(self.device),
		)

	def add(self,sa1,sa2,label):
		num=1
		end_ptr=self.ptr+num
		
		self.seg1[self.ptr:end_ptr] = sa1
		self.seg2[self.ptr:end_ptr] = sa2
		self.label[self.ptr:end_ptr]=label

		self.ptr = (self.ptr + num) % self.max_size
		self.size = min(self.size + num, self.max_size)
		assert self.size<=self.max_size

	def save(self, path):
		np.savez(path, seg1=self.seg1[:self.size], seg2=self.seg2[:self.size], label=self.label[:self.size])
	
	def load(self, path):
		stored_array=np.load(path)
		self.size=len(stored_array["seg1"])
		self.seg1 = stored_array["seg1"]
		self.seg2 = stored_array["seg2"]
		self.label = stored_array["label"]