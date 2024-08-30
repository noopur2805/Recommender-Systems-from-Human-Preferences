import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import process_reward, norm_state
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_offsite = torch.tensor([0.94161665, 0.9616005 , 0.8307782 , 0.8431818 , 0.82132435,
        1.432366  , 0.94419135, 1.1348905 ]).to(device)
action_scale = torch.tensor([3.93085895, 4.4827265 , 4.1778248 , 3.8281758 , 3.85135935,
        4.5995945 , 4.09298865, 4.1333175]).to(device)
LOG_STD_MAX = 2
LOG_STD_MIN = -20
class Predictor(nn.Module):
    def __init__(self, state_dim, h_state_dim, action_dim):
        super(Predictor, self).__init__()

        self.l1 = nn.Linear(state_dim+h_state_dim+action_dim, 256)
        self.l2 = nn.Linear(256 , 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state_h_state, action):
        # assert torch.isnan(state).int().sum().item()==0 
        # assert torch.isnan(action).int().sum().item()==0

        action=torch.clamp(((action-action_offsite)/action_scale),-1,1)
        inputs=torch.cat([state_h_state, action], -1)
        r = F.relu(self.l1(inputs))
        r = F.relu(self.l2(r))
        return self.l3(r)

class VIBPredictor(nn.Module):
    def __init__(self, state_dim, h_state_dim, action_dim):
        super(VIBPredictor, self).__init__()
        self.l1 = nn.Linear(state_dim+h_state_dim+action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        self.encoder = nn.Sequential(*[self.l1,nn.ReLU(inplace=True),self.l2])
        self.mu_layer = nn.Linear(256, 64)
        self.log_std_layer = nn.Linear(256, 64)

        self.decoder = nn.Sequential(nn.Linear(64, 64),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(64, 1))

    def forward(self, state_h_state, action, deterministic=False):
        action=torch.clamp(((action-action_offsite)/action_scale),-1,1)
        inputs=torch.cat([state_h_state, action], -1)
        net_out = self.encoder(inputs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            r = mu
        else:
            r = pi_distribution.rsample()
        r=self.decoder(r)     
        return r

    def dist(self, state_h_state, action):
        action=torch.clamp(((action-action_offsite)/action_scale),-1,1)
        inputs=torch.cat([state_h_state, action], -1)
        net_out = self.encoder(inputs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

class RewardModel(object):
    def __init__(self,args):
        self.predictor=[Predictor(args.state_dim, args.h_state_dim, args.action_dim).to(device) for _ in range(args.n_predictor)]
        self.predictor_optimizer = [torch.optim.Adam(predictor.parameters(),lr=args.lr, weight_decay=1e-2) for predictor in self.predictor]
        self.vibpredictor=[VIBPredictor(args.state_dim, args.h_state_dim, args.action_dim).to(device) for _ in range(args.n_vibpredictor)]
        self.vibpredictor_optimizer=[torch.optim.Adam(predictor.parameters(),lr=args.lr, weight_decay=1e-2) for predictor in self.vibpredictor]
        self.args=args

    def train(self, pref_replay_buffer, batch_size=64, debug=False):
        seg1,seg2,label=pref_replay_buffer.sample(batch_size)
        if debug:
            l_loss=[]
            l_acc=[]
        loss=0
        acc=0
        for idx in range(self.args.n_predictor):
            l,a=self.train_predictor(self.predictor[idx],self.predictor_optimizer[idx],False,seg1,seg2,label)
            loss+=l
            acc+=a
            if debug:
                l_loss.append(l)
                l_acc.append(a)
        for idx in range(self.args.n_vibpredictor):
            l,a=self.train_predictor(self.vibpredictor[idx],self.vibpredictor_optimizer[idx],True,seg1,seg2,label)
            loss+=l
            acc+=a 
            if debug:
                l_loss.append(l)
                l_acc.append(a)
        if debug:
            print(l_loss)
            print(l_acc)  
        return loss/(self.args.n_predictor+self.args.n_vibpredictor), acc/(self.args.n_predictor+self.args.n_vibpredictor) 
    
    def train_predictor(self,predictor, opt, if_vib, seg1,seg2, original_label):
        label=copy.deepcopy(original_label)
        batch_size=label.shape[0]
        r1=self.seg_to_return(seg1,predictor)
        r2=self.seg_to_return(seg2,predictor)
        logits=torch.stack([r1,r2],dim=1) #[batch,2]
        
        label=label[:,self.args.return_type] #batch
        uniform_index = label == 0
        label-=1
        label[uniform_index] = 0
        label_onehot = torch.zeros_like(logits).scatter(1, label.unsqueeze(1), 1)
        if sum(uniform_index) > 0:
            label_onehot[uniform_index] = 0.5
        loss = self.softXEnt_loss(logits, label_onehot.detach())

        if if_vib:
            state_h_state=seg1[:,:,:self.args.state_dim+self.args.h_state_dim]
            action=seg1[:,:,self.args.state_dim+self.args.h_state_dim:]
            mean, std=predictor.dist(state_h_state,action) #[batch, seg_length, 64]
            loss += (-0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean())*0.5
            state_h_state=seg2[:,:,:self.args.state_dim+self.args.h_state_dim]
            action=seg2[:,:,self.args.state_dim+self.args.h_state_dim:]
            mean, std=predictor.dist(state_h_state,action) #[batch, seg_length, 64]
            loss += (-0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean())*0.5


        opt.zero_grad()
        loss.backward()
        opt.step()

        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == label).sum().item()
        acc=correct/batch_size
        return loss.item(), acc

    def relabel(self, state, action): # (batch, dim)
        rewards=[]
        for p in self.predictor:
            rewards.append(p(state,action))
        for p in self.vibpredictor:
            rewards.append(p(state,action,True))
        rewards=torch.cat(rewards,dim=-1)
        return rewards.mean(-1,True) # (batch, 1)
    
    def seg_to_return(self, seg, predictor):
        seg=seg.detach()
        state_h_state=seg[:,:,:self.args.state_dim+self.args.h_state_dim]
        action=seg[:,:,self.args.state_dim+self.args.h_state_dim:]
        returns=predictor(state_h_state,action) #[batch, seg_length, 1]
        return torch.sum(returns, (1,2))
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]