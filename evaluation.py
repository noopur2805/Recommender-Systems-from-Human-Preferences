import json
from torch.distributions.normal import Normal
from data_to_buffer import *
from utils import process_reward_eva

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean=torch.tensor([1.4507225122503788,0.9811812772854842,1.352303211127227,0.9243272381031921,0.5362567258084541,1.1423187782609598,0.7444246553715433,1.4604244948003298]).to(device)
std=torch.tensor([0.5023057117013698,0.5621463472517462,0.6454519272171669,0.4639104642469882,0.49302427360992496,0.5044147906509101,0.5143820049444064,0.5528706872967144]).to(device)

def w_offline_ab(policy, file_path, return_type):
    # calculate sequentially, could be calculate in batch as well.
    R=0
    with open(file_path, 'r') as f:
        trajactory=json.load(f)
        prob_traj=[]
        rewards=[]
        
        with torch.no_grad():
            try:
                state,h_state,action,next_action,next_state,next_h_state, response,h_response,done=parse_json(trajactory)
                for array in [state,h_state,action,next_action,next_state,next_h_state, response,h_response,done]:
                    assert not np.isnan(array).any()

                for t in range(len(state)):
                    s=np.concatenate([state[t],h_state[t]])
                    policy_action=policy.select_action(s)                   
                    dist=Normal(policy_action, std)
                    
                    a=torch.tensor(action[t]).to(device)
                    log_prob=dist.log_prob(a).mean(axis=-1) # should be sum, mean for larger value
                    prob=torch.exp(log_prob)
                    prob_traj.append(prob.cpu().item())
                    rewards.append(process_reward_eva(h_response[t],return_type))
                prob_traj=np.array(prob_traj)
                norm_prob=prob_traj/prob_traj.sum()
                
                R=(norm_prob*(np.array(rewards))).sum().item()
                
                # for t in range(len(norm_prob)):    
                #     print(norm_prob[t])
                    
                return R
            except:
                pass

def stat(returns):
        avg_ret = np.mean(returns)
        se = scipy.stats.sem(returns)
        h = se * sp.stats.t._ppf((1 + 0.95) / 2., len(returns) - 1)
        print(f"mean: {avg_ret}, h: {h}")
        return avg_ret, h
