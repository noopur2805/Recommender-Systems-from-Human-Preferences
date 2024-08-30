import os, random
from buffer import *
import json
from tqdm import tqdm

action_dim=8

def sample_user(replay_buffer, number, fold , folder_path, save_path):
    n_user=0
    n_session=0
    n_request=0
    assert number <= len(os.listdir(folder_path))
    #user_list=random.sample(os.listdir(folder_path),number)
    user_list=os.listdir(folder_path)[fold*number:(fold+1)*number]
    for user in tqdm(user_list):
        file_path=os.path.join(folder_path,user)
        with open(file_path, 'r') as f:
            trajactory=json.load(f)
            try:
                state,h_state,action,next_action,next_state,next_h_state, response,h_response,done=parse_json(trajactory) # transitions in the trajactory of a user
                for array in [state,h_state,action,next_action,next_state,next_h_state, response,h_response,done]:
                    assert not np.isnan(array).any()
                replay_buffer.add(state,h_state,action,next_action,next_state,next_h_state, response,h_response,done)
                n_user+=1
                n_session+=len(trajactory)
                n_request+=len(state)
            except Exception as e:
                pass
    with open(os.path.join(save_path,f"stat-{fold}-fold.json"),"w") as f:
        json.dump({"n_user":n_user, "n_session":n_session, "n_request":n_request},f,separators=(',', ':'),indent=4)
    # The calculation is incorrect as the buffer is padded with zeros.
    # with open(os.path.join(save_path,f"stat-action-{i}-fold.json"),"w") as f:
    #     mean_a, std_a, max_a, min_a=replay_buffer.stat_actions()
    #     json.dump({"mean":mean_a, "std_a":std_a, "max":max_a,"min":min_a},f,separators=(',', ':'),indent=4)
    replay_buffer.save(os.path.join(save_path,f"{fold}-fold")) 
    return replay_buffer

def parse_json(trajactory): 
    state=[]
    h_state=[]
    action=[]
    response=[]
    h_response=[]

    for session in trajactory:
        for i, request in enumerate(session["request"]):
            state.append(request["state"])
            h_state.append(request["h_state"])
            assert len(request["action"])>=8 and len(request["action"])<=12, f'invalid action: {len(request["action"])}'
            action.append(request["action"][:action_dim])
            response.append(request["response"])
            h_response.append(request["h_response"])

    next_action=action[1:]
    next_action.append(action[-1])

    next_state=state[1:]
    next_state.append(state[-1])
    
    next_h_state=h_state[1:]
    next_h_state.append(h_state[-1])

    done=np.zeros_like(list(range(len(state))))
    done[-1]=1  
    done=np.expand_dims(done, axis=1)
    return state,h_state,action,next_action,next_state,next_h_state, response,h_response,done

def stat_actions(action, eps= 1e-3):
    shape=list(action.shape)
    mean = action.mean(0,keepdims=True)
    std= action.std(0,keepdims=False) + eps
    max_a=action.max(axis=0)
    min_a=action.min(axis=0)
    return mean.tolist(), std.tolist(), max_a.tolist(), min_a.tolist(), shape
            
def seg_to_buffer(pref_buffer, number, length, fold, folder_path, save_path, thresh_rate=0.15):
    user_list=os.listdir(folder_path)
    pbar=tqdm(total=number)
    N=0
    while True:
        user1, user2=np.random.choice(user_list, 2, replace=False)
        sa1, r1= extract_sa_r(user1, folder_path)
        if sa1 is None:
            continue
        sa2, r2= extract_sa_r(user2, folder_path)
        if sa2 is None:
            continue
        r1=np.sum(r1, axis=0) # [2]
        r1=np.append(r1, r1[0]*0.7+r1[1]*0.3)
        r2=np.sum(r2, axis=0) 
        r2=np.append(r2, r2[0]*0.7 +r2[1]*0.3)
        label=np.zeros(3)
        for k in range(3):
            if abs(r1[k]-r2[k]) < thresh_rate*(abs(r1[k])+abs(r2[k])):
                label[k] = 0 # equal
            elif r1[k]-r2[k] > 0:
                label[k] = 1
            elif r1[k]-r2[k] < 0:
                label[k] = 2
            else:
                continue
        pref_buffer.add(sa1,sa2,label)
        N+=1
        pbar.update(1)
        if N>=number:
            pref_buffer.save(os.path.join(save_path,f"{fold}-fold"))
            break
    return pref_buffer

def extract_sa_r(user, folder_path):
    file_path=os.path.join(folder_path,user)
    with open(file_path, 'r') as f:
        trajactory=json.load(f)
        try:
            state,h_state,action,next_action,next_state,next_h_state, response,h_response,done=parse_json(trajactory) # transitions in the trajactory of a user
            for array in [state,h_state,action,next_action,next_state,next_h_state, response,h_response,done]:
                assert not np.isnan(array).any()
            if len(state) < length:
                return None, None
            state=np.array(state[:length])
            h_state=np.array(h_state[:length])
            action=np.array(action[:length])
            h_response=np.array(h_response[:length]) # [length,3]
            sa=np.concatenate([state,h_state,action],axis=1) #[length, -1]
            return sa, h_response
        except Exception as e:
            return None, None
