'''
整体流程：

根据是否得到奖励，将过程分为两个部分

第一部分，未得到奖励之前的探索，一旦得到奖励，就入第二部分，直接利用得到奖励的轨迹进行模仿

罗列所有可能参数，并观察之间的关系
需要进行验证的参数是各个权重之间的配比
需要超参数进行调节的：

先得到一个能够求解问题的方法
再想着如何比较参数

'''

from SPPI.GridMaze.maze_env10 import Maze
from SPPI.GridMaze.SIIRQBrain import SIIRQBrain
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
def LearnByQ(agent,env,SuccessTrj):
    print("learning by Q")
    #TODO
    #将终点的R 平均分给各个路途中的点，以此来重新初始化Q表
    episode = 0
    steps = []#记录每个回合的步骤数
    reward_list = []
    while episode<1000: #这是已经得到奖励之后的，所以不成功就得一直训下去
        episode +=1

        step = 0
        curr_trj=[]#用来记录成功轨迹
        observation = env.reset()
        while step <100:#每个回合要限定步数，如果这些步没到，R就为0
            step += 1
            state = agent.obs2state(observation)
            curr_trj.append(state)
            action = agent.choose_action_q(state)
            observation_, reward, done = env.step(action)
            state_ = agent.obs2state(observation_) 
            agent.learn_q(state,action,reward,state_)
            if done:
                agent.getRflag == True
                print("get reward in %d steps!"%step)
                break
            observation = observation_
        steps.append(step)
        reward_list.append(reward)
        if episode>11:
            #print(episode,steps,reward_list)
            if sum(steps[episode-11:episode-1])==steps[episode-1]*10 and sum(reward_list[episode-11:episode-1])==2000:
                print("main loop %d convergence"%episode)
                break
def generate_trjs(M_trjs,N_steps,RL,env):
    RL.epsilon = 0.5
    trjs =[]
    SuccessTrj =[]
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
        SuccessTrj = trj.copy()       
        trjs.append(trj)
    return trjs#,SuccessTrj    


def LearnBySIQ(agent,env,iloop):
    curr_horizon = 5+iloop 
    #curr_horizon = minstep + 5 #经历过的状态数目开根号
    g_n_trjs = 100 #可以尝试多产生几个
    g_n_steps = curr_horizon
    print("current horizon ",curr_horizon)
    trjs = generate_trjs(g_n_trjs,g_n_steps,agent,env)#这里的g_n_steps需要渐进
    #print(trjs)
    p = agent.stochastic_trjs(trjs)
    #print("p",p)
    r_SI, maxid = agent.SI(p,iloop+1)
    print("maxid",maxid)
    #print("r_si",r_SI)
    maxSI= agent.StateFunctionValueDict['StateValue'].max()
    #print("maxsi",maxSI)
    episode =0
    agent.epsilon = 0.9
    steps = []#记录每个回合的步骤数
    reward_list = []
    while episode < 1000:
        episode += 1
        observation = env.reset()
        step = 0
        while step < 1000:
            step += 1
            state = agent.obs2state(observation)
            action = agent.choose_action_q(state)
            observation_, reward, done = env.step(action)
            state_ = agent.obs2state(observation_) 
            if state_ == maxid :
                reward = reward + iloop*5
            agent.learn_q(state,action, reward, state_)
            if done:
                print("done")
                agent.getRflag = True
                break
            if state_== maxid:
                print("get local reward")
                break
            observation = observation_
        steps.append(step)
        reward_list.append(reward)
        if done:
            break
        if episode > 11:
            #print("reward list",reward_list)
            if sum(steps[episode-11:episode-1])==steps[episode-1]*10 and sum(reward_list[episode-11:episode-1])==10*reward_list[episode-1]:
                print("local %d convergence"%episode)
                break





def main_MAZE(env):
    agent = SIIRQBrain(env) #初始化
    MainLoop = 2*10 # 主循环个数，如果按照每次扩展一个视野来看，至少有网格行数的二倍才可以
    SuccessTrj = []
    #agent.getRflag = True
    iloop =0
    while iloop<MainLoop:
        if agent.getRflag:#如果已经可以通过某种手段得到奖励,那么一定有一条成功的轨迹，那么直接用Q探索就行
            iloop = MainLoop
            LearnByQ(agent,env,SuccessTrj)
        else:
            iloop = iloop+1
            LearnBySIQ(agent,env,iloop)


if __name__ == "__main__":
    env = Maze(10,40)
    main_MAZE(env)