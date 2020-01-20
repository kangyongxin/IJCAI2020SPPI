# Social influence based intrisic reward qlearning
#kangyongxin2018@ia.ac.cn
'''
#to do
# 基本流程模块化
#   探索过程!
#   结果评估!
#   状态编码!
#   状态计数，分量求取!
#   融合到 q 中去!
#   设计流程
# 实验部署
#   qlearning为baseline
#   SIIR Q 
#   SIIR Q 用自身属性
#   SIIR Q 用关联关系
'''
from SPPI.GridMaze.maze_env10 import Maze
from SPPI.GridMaze.SIIRQBrain import SIIRQBrain
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

def generate_trjs(M_trjs,N_steps,RL,env):

    trjs =[]
    for eps in range(M_trjs):
        trj = []
        step = 0
        observation = env.reset()

        while step < N_steps and RL.getRflag == False:
            step += 1
            env.render()
            #action = RL.random_action(observation)
            state = RL.obs2state(observation)
            trj.append(state)
            action = RL.choose_action_q(state)
            observation_, reward, done = env.step(action)
            state_ = RL.obs2state(observation_)
            if done:
                print("done during generate")
                RL.getRflag = True
                state=RL.obs2state(observation)
                trj.append(state_)
                #RL.resetIm(trj,reward)
                break   
            observation = observation_         
        trjs.append(trj)
    return trjs


def main_MAZE(env):
    n_trjs = 1000
    n_steps =10
    SHOW_FLAG = True
    save_path = './SPPI/GridMaze/results/'
    path_reward_episode = save_path+'random'+'_'+str(n_trjs)+'_'+str(n_steps)+'_reward_episode.csv'
    path_reward_step = save_path+'random'+'_'+str(n_trjs)+'_'+str(n_steps)+'_reward_step.csv'
    agent = SIIRQBrain(env)
    g_n_trjs=100# 生成轨迹数目
    g_n_steps=50# 生成轨迹的步数
    mineststep =50
    #agent.getRflag=True


    reward_list=[]
    step_r=[]

    main_loop = 40
    for main_loop_i in range(main_loop):
        curr_horizon = g_n_steps+ mineststep + 10
        print("current horizon ",curr_horizon)
        step_each_eps=[]
        if not agent.getRflag:
            trjs = generate_trjs(g_n_trjs,g_n_steps-5,agent,env)#这里的g_n_steps需要渐进
            p = agent.stochastic_trjs(trjs)
            print("p",p)
            r_SI = agent.SI(p,main_loop_i+1)
            print("r_si",r_SI)
            maxSI= agent.StateFunctionValueDict['StateValue'].max()
            print("maxsi",maxSI)
        episode = 0 
        
        while episode < n_trjs:
            #print("episode",episode)
            episode += 1
            observation = env.reset()
            step = 0
            r_episode=0
            trj_record = []
            while step <curr_horizon:
                step += 1
                if SHOW_FLAG:
                    env.render()
                state = agent.obs2state(observation)
                trj_record.append(state)
                #action= agent.random_action(state)
                action = agent.choose_action_q(state)
                observation_, reward, done = env.step(action)
                state_ = agent.obs2state(observation_) 
                if state_ in agent.StateFunctionValueDict.index:
                    if agent.StateFunctionValueDict.loc[state_,'FirstVisit']==True:
                        r_intrinsic=agent.StateFunctionValueDict.loc[state_,'StateValue']
                        
                    else:
                        pass
                else:
                    r_intrinsic =0
                reward = reward+ r_intrinsic
                r_episode += reward
                step_r.append(reward)
                agent.learn_q(state,action,reward,state_)
                if done:
                    agent.getRflag == True
                    print("get reward ! episode %d done!"%episode)
                    break
                
                if state_ in agent.StateFunctionValueDict.index:
                    if agent.StateFunctionValueDict.loc[state_,'StateValue']==maxSI:
                        print("get local reward! episode %d done!"%episode)
                        mineststep = step
                        break
                observation = observation_
  
   
            step_each_eps.append(step)
            reward_list.append(r_episode)
            #判断是否收敛
            if episode >10:
                if agent.getRflag or step > curr_horizon:
                    pass
                else:
                    if sum(step_each_eps[episode-11:episode-1])==step_each_eps[episode-1]*10:
                        print("step each eps",step_each_eps)
                        print("main loop %d convergence "%main_loop_i)
                        for sta in trj_record:
                            if sta not in agent.StateFunctionValueDict.index:
                                print("add trj to SI")
                                agent.StateFunctionValueDict = agent.StateFunctionValueDict.append(
                                    pd.Series(
                                        {'StateValue':0.0, 'FirstVisit':False},
                                        name = sta,
                                    )
                                )  
                        break

        

    plt.plot(reward_list)
    plt.show()
    temp_step_r=[]
    for i in range(len(step_r)):
        if i<200 :
            temp_step_r.append(step_r[i])
        else:
            temp_step_r.append(sum(step_r[(i-190):i])/190)
    plt.plot(temp_step_r)
    plt.show()
    reward_episode=pd.DataFrame({'reward_eps':reward_list,'step_each_eps':step_each_eps})
    reward_episode.to_csv(path_reward_episode,encoding='gbk')
    reward_step =pd.DataFrame({'reward_step':step_r,'reward_step_sum':temp_step_r})
    reward_step.to_csv(path_reward_step,encoding='gbk')
    
if __name__ == "__main__":
    env = Maze(10,20)
    main_MAZE(env)