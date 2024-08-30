import os
import copy
import json
import numpy as np
from tqdm import tqdm
import random

hive_folder="./ResAct/sample_data"
phase1_folder="./ResAct/phase1" # path to the folder to store the raw sessions for each user
phase2_folder="./ResAct/phase2"
stat_folder="/ResAct/data_stat"

session_template={"starttime": None, "request": []}
request_template={"state":[],"action":[],"response":[]}

pxtr=[ "pctr","plvtr","psvtr","pvtr","pltr","phtr","pwtr","pftr","pptr","pdtr","pcmtr","pcmef","pepstr"]
count=["show_count","follow_count","like_count","profile_enter_count","forward_count","comment_stay_ms","profile_stay_ms","photo_play_ms","duration_ms"]
response_count=["follow_count","like_count","profile_enter_count","forward_count","comment_stay_ms","profile_stay_ms","photo_play_ms","duration_ms"]

pxtr_scaler=np.array([0.001,0.001,0.001,0.001,0.001,100,0.1,0.1,0.1,1,0.001,0.001,1])
count_scaler=np.array([0.001,10,0.001,0.001,1,1e-5,1e-8,1e-8,1e-9])
n_h_state_clip,n_l_state_clip,n_response_clip=0,0,0

def prepare_state_response(raw_state, raw_response, first_request):
    global n_h_state_clip, n_l_state_clip, n_response_clip
    photo_pxtr_average=raw_state["history_attrs"]["photo_pxtr_average"][0]
    request_count_average=raw_state["history_attrs"]["request_count_average"][0]
    photo_pxtr_average=np.array([photo_pxtr_average[key] for key in pxtr])
    request_count_average=np.array([request_count_average[key] if key in request_count_average else 0 for key in count])

    pxtr_percentile=raw_state["context_attrs"]["pxtr_percentile"]
    pxtr_percentile=np.array([[pxtr_percentile[i][key] for key in pxtr] for i in range(len(pxtr_percentile))])
    pxtr_topk=raw_state["context_attrs"]["pxtr_topk"]
    pxtr_topk=np.array([[pxtr_topk[i][key] for key in pxtr] for i in range(len(pxtr_topk))]).mean(0)
    pxtr_percentile=((pxtr_percentile*100)/photo_pxtr_average).flatten()
    pxtr_topk=(pxtr_topk*100)/photo_pxtr_average
    
    gender=np.zeros(3)
    gender[raw_state["user_info"]["gender"]]=1
    age=np.zeros(10)
    age[raw_state["user_info"]["age_segment"]]=1
    
    h_state=np.concatenate([gender,age,pxtr_percentile,pxtr_topk,photo_pxtr_average*pxtr_scaler, request_count_average*count_scaler]) #(178,)
    high_level_state=np.clip(h_state,-1,100)
    if not (h_state==high_level_state).all():
        n_h_state_clip+=1
        #print("perform high level state clipping.")

    history_pxtr=raw_state["history_attrs"]["photo_pxtr_average"][1:]
    history_count=raw_state["history_attrs"]["request_count_average"][1:]
    history_pxtr=np.array([[history_pxtr[i][key] for key in pxtr] for i in range(3)])
    history_count=np.array([[history_count[i][key] if key in history_count[i] else 0 for key in count] for i in range(3)])
    history_pxtr=((history_pxtr*100)/photo_pxtr_average).flatten()
    history_count=((history_count*10)/(request_count_average+0.0000001)).flatten()
    
    l_state=np.concatenate([high_level_state,history_pxtr,history_count,np.array([first_request])]) #(245,)
    low_level_state=np.clip(l_state,-1,100)
    if not (l_state==low_level_state).all():
        n_l_state_clip+=1
        #print("perform low level state clipping.")

    response=raw_response["immediate_attrs"]["sum_count"]
    response=np.array([response[key] if key in response else 0 for key in response_count])
    r_before_clip=(response*10)/(request_count_average[1:]+0.0000001)
    r_before_clip[-1]=r_before_clip[-2]/(r_before_clip[-1]+0.0000001) #(8,) ["follow_count","like_count","profile_enter_count","forward_count","comment_stay_ms","profile_stay_ms","photo_play_ms","ratio"]
    
    r=np.clip(r_before_clip,-1,10)
    if not (r==r_before_clip).all():
        n_response_clip+=1
        #print("perform response clipping.")

    return high_level_state.tolist(), low_level_state.tolist(), r.tolist(), response.tolist()

def prepare_action(raw_action):
    return raw_action["embedding"]

def write_sessions_byline(file_folder):
    # 1. from raw hive table to did.txt
    # 2. one session a line
    # 3. the response contains only low-level rewards
    # r_stat=[]
    # response_stat=[]
    n_request=0
    n_sessions=0
    for hive_table in tqdm(os.listdir(file_folder)):
        file_path=os.path.join(file_folder,hive_table)
        with open(file_path, 'r') as f:
            for l in tqdm(f.readlines()):
                raw=json.loads(l)
                session=copy.deepcopy(session_template)
                session["starttime"]=raw["timestamp"]
                did=raw["states"][0]["user_info"]["device_id"]
                try:
                    high_level_state,_,_,_=prepare_state_response(raw["states"][0],raw["rewards"][0],0)
                    n_steps=len(raw["states"])
                    for t in range(n_steps):
                        raw_state=raw["states"][t]
                        raw_action=raw["actions"][t]
                        raw_response=raw["rewards"][t]
                        if t==0:
                            _, low_level_state,r,response=prepare_state_response(raw_state, raw_response, 1)
                        else:
                            _, low_level_state,r,response=prepare_state_response(raw_state, raw_response, 0)
                        action=prepare_action(raw_action)
                        session["request"].append({"state":low_level_state,"h_state":high_level_state, "action":action,"response":r})
                        # r_stat.append(r) # reward of each request
                        # response_stat.append(response) # raw response of each request
                        n_request+=1
                    with open(os.path.join(phase1_folder, f"{did}.txt"),"a") as user:
                        user.writelines(json.dumps(session)+'\n')
                        n_sessions+=1
                except Exception as e:
                    pass
        # overwrite every hive table
        # n_request=len(r_stat)
        stats=f"n_session: {n_sessions}, n_request: {n_request}, n_h_state_clip: {n_h_state_clip}, n_l_state_clip: {n_l_state_clip}, n_response_clip: {n_response_clip}."
        with open(os.path.join(stat_folder,"stats.txt"),"w") as des:
            des.writelines(stats)
        # with open(os.path.join(stat_folder,"r_per_request.txt"),"w") as r_des:
        #     json.dump(r_stat, r_des)
        # with open(os.path.join(stat_folder,"response_per_request.txt"),"w") as response_des:
        #     json.dump(response_stat, response_des)

# write_sessions_byline(hive_folder)

def cal_stat_per_user():
    return_time_stat=[]
    session_length_stat=[]
    session_number_stat=[]
    for user in tqdm(os.listdir(phase1_folder)):
        with open(os.path.join(phase1_folder,user),'r') as f:
            sessions=[json.loads(l) for l in f.readlines()]
            sessions=sorted(sessions, key=lambda d: d["starttime"])
            timestamps=[s["starttime"] for s in sessions]
            session_length=[len(s["request"]) for s in sessions]
            n_sessions=len(timestamps)
            session_length_stat.append(np.array(session_length).mean().item())
            session_number_stat.append(n_sessions)
            if n_sessions==1:
                return_time_stat.append(1e9)
            else:
                t=np.array(timestamps[:-1])
                next_t=np.array(timestamps[1:])
                avg_return_time=(next_t-t).mean()
                return_time_stat.append(avg_return_time)
        with open(os.path.join(stat_folder,"return_time_stat.txt"),"w") as stat:
            json.dump(return_time_stat, stat)
        with open(os.path.join(stat_folder,"session_length_stat.txt"),"w") as stat:
            json.dump(session_length_stat, stat)
        with open(os.path.join(stat_folder,"session_number_stat.txt"),"w") as stat:
            json.dump(session_number_stat, stat)

def merge_sessions():
    # 1. sort sessions by time stamp for each user
    # 2. calculate high-level rewards
    r_return_time_stat=[]
    r_session_length_stat=[]
    for user in tqdm(os.listdir(phase1_folder)):
        with open(os.path.join(phase1_folder,user),'r') as f:
            sessions=[json.loads(l) for l in f.readlines()]
            sessions=sorted(sessions, key=lambda d: d["starttime"])

            timestamps=[s["starttime"] for s in sessions]
            if len(timestamps) >1:
                t=np.array(timestamps[:-1])
                next_t=np.array(timestamps[1:])
                avg_return_time=(next_t-t).mean().item()
                # if not ((next_t-t)>0).all():
                #     print("time wrong!!!")
                r_return_time=np.clip(min(avg_return_time,40606)//(next_t-t+0.0000001),0,5).tolist() # 75% percentile
                r_return_time.append(0)
                r_return_time_stat+=r_return_time
            else:
                r_return_time=[0]
                r_return_time_stat+=r_return_time

            session_length=[len(s["request"]) for s in sessions]
            avg_session_length=np.array(session_length).mean().item()
            r_session_length=[min(l//(avg_session_length*0.7),5) for l in session_length]
            r_session_length_stat+=r_session_length

            for i,session in enumerate(sessions):
                for t in range(len(session["request"])):
                    if t==0:
                        session["request"][t]["h_response"]=[r_return_time[i],r_session_length[i]]
                    else:
                        session["request"][t]["h_response"]=[0,0]
        if random.random() < 0.8:
            with open(os.path.join(phase2_folder,"train",user),'w') as f:
                f.writelines(json.dumps(sessions))
        else:
            with open(os.path.join(phase2_folder,"test",user),'w') as f:
                f.writelines(json.dumps(sessions))
    with open(os.path.join(stat_folder,"r_return_time_stat.txt"),"w") as stat:
        json.dump(r_return_time_stat, stat)
    with open(os.path.join(stat_folder,"r_session_length_stat.txt"),"w") as stat:
        json.dump(r_session_length_stat, stat)
