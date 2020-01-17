# Social influence based intrisic reward qlearning
#kangyongxin2018@ia.ac.cn
'''
#to do
# 基本流程模块化
#   探索过程!
#   结果评估!
#   状态编码!
#   状态计数，分量求取
#   融合到 q 中去
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
                RL.getRflag = False
                state=RL.obs2state(observation)
                trj.append(state_)
                RL.resetIm(trj,reward)
                break   
            observation = observation_         
        trjs.append(trj)
    return trjs


def main_MAZE(env):
    n_trjs = 10
    n_steps =100
    SHOW_FLAG = True
    save_path = './SPPI/GridMaze/results/'
    path_reward_episode = save_path+'random'+'_'+str(n_trjs)+'_'+str(n_steps)+'_reward_episode.csv'
    path_reward_step = save_path+'random'+'_'+str(n_trjs)+'_'+str(n_steps)+'_reward_step.csv'
    agent = SIIRQBrain(env)
    g_n_trjs=10# 生成轨迹数目
    g_n_steps=3# 生成轨迹的步数

    reward_list=[]
    step_r=[]
    step_each_eps=[]
    episode = 0 
    while episode < n_trjs:
        episode += 1
        observation = env.reset()
        step = 0
        r_episode=0
        while step <n_steps:
            step += 1
            if SHOW_FLAG:
                env.render()
            state = agent.obs2state(observation)
            #action= agent.random_action(state)
            action = agent.choose_action_q(state)
            observation_, reward, done = env.step(action)
            r_episode += reward
            step_r.append(reward)
            state_ = agent.obs2state(observation_)
            agent.learn_q(state,action,reward,state_)
            
            if done:
                print("done!")
                break
            observation = observation_
        step_each_eps.append(step)
        reward_list.append(r_episode)
        trjs = generate_trjs(g_n_trjs,g_n_steps,agent,env)#这里的g_n_steps需要渐进
        


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
    env = Maze(5,5)
    main_MAZE(env)