"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.95):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action
    def random_action(self,observation):
        action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        #print("self.q",self.q_table)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
class UptrendVS():
    def __init__(self,env,actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.95):
        self.actions = actions
        self.epsilon = e_greedy
        self.lr = learning_rate
        self.gamma = reward_decay
        self.S_space = env.state_space()
        self.env = env
        self.Im = pd.DataFrame(columns=['sta_visul','ImS', 'beforestate','getstate'],
                          dtype=np.float64)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.getRflag = False
        self.minstep = 10000

        #self.Ci = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def random_action(self,observation):
        action = np.random.choice(self.actions)
        return action

    def choose_action(self, observation):
        self.check_state_exist(observation)
        #self.check_state_exist_Im(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def obs_to_state(self,obs):#,env)
        """"
        这里用来构造从观测到状态的函数
        """
        states = obs
        return states #int(states)

    def stochastic_trj(self,seq):
        """

        :param seq:  Ti

        :return: Ci= DataFrame(index=[sta],columns=['StateCount','StateOrder'], dtype=np.float64)
        """
        s=seq[0]
        sta=str(s) #因为做索引的不能是数组，所以将他变成了字符串
        sta_visul = self.env.state_visualization(s)  # 因为我们要直观的看到最后的结果所以把他转换成可视化的编码
        Ci = pd.DataFrame([{'sta_visul':sta_visul,'StateCount':seq.count(s),'StateOrder':1}],index=[sta],columns=['sta_visul','StateCount','StateOrder'], dtype=np.float64)
        k= 1
        for s in seq:
            sta = str(s)
            if s == 'terminal':
                sta_visul = s
            else:
                sta_visul = self.env.state_visualization(s)  # 因为我们要直观的看到最后的结果所以把他转换成可视化的编码
            if sta not in Ci.index:
                k += 1
                Ci=Ci.append(
                    pd.Series({'sta_visul':sta_visul,'StateCount':seq.count(s),'StateOrder':k},
                              name=sta,
                              )
                )
        return Ci

    def stochastic_trjs(self,trjs):
        """

        :param trjs: 众多轨迹
        'StateCount': 每条轨迹中状态stat出现的次数，Ci（Ci.loc[:,'StateCount']）
        SumWik:only calculate the sum of Wik, where the Wik is the order of state k in the list Ui( Ci.index)
        Pk:the number of S_k in all the trjs
        iPk: 每条轨迹某个状态出现只算一次，看看所有轨迹中有的多少条轨迹包含当前状推
        BEk：the stochastic of states BEfore S_k in each trj
        AFk : the stochastic of states AFter S_k in each trjs
        :return: P， 未顺序排列的统计特性，O，按照出现数量有高到底排列

        """
        seq = trjs[0]
        s=seq[0]

        sta = str(s)  # 因为做索引的不能是数组，所以将他变成了字符串
        sta_visul = self.env.state_visualization(s)  # 因为我们要直观的看到最后的结果所以把他转换成可视化的编码
        iPk = 1
        BEk = 0
        AFk = 0
        P= pd.DataFrame([{'sta_visul':sta_visul,'StateOrder':1,'SumWik':0,'Pk':seq.count(sta),'iPk':iPk,'BEk':BEk,'AFk':AFk,'BEplusAF':(BEk+AFk)}],index=[sta],columns=['sta_visul','StateOrder','SumWik','Pk','iPk','BEk','AFk','BEplusAF'], dtype=np.float64)
        # self.Im = pd.DataFrame([{'ImS':0, 'beforestate':1}],index =[sta] ,columns=['ImS', 'beforestate'],
        #                   dtype=np.float64)
        k=1
        for trj in trjs:
            #每一条轨迹,
            temp = self.stochastic_trj(trj)
            # print("temp \n",temp)
            for StateID in temp.index:
                #每一个状态
                BEk = 0
                AFk = 0
                SO = int(temp.loc[StateID,'StateOrder'])
                if SO == 1:
                    continue#第一个状态不算, 无论之前还是之后
                else:
                    for i in range(1,SO):#本身不算
                        da = temp[temp['StateOrder']==i]
                        BEk = BEk + da.iloc[0,1]# StateCount在第二个
                    for i in range(SO+1,len(temp)+1):
                        da = temp[temp['StateOrder']==i]
                        AFk = AFk + da.iloc[0,1]# StateCount在第二个

                if StateID not in P.index:
                    #如果之前没有出现过
                    k+=1
                    P = P.append(
                        pd.Series(
                            {'sta_visul': temp.loc[StateID,'sta_visul'],'StateOrder': k, 'SumWik':temp.loc[StateID,'StateOrder'],'Pk':seq.count(sta),'iPk':iPk,'BEk':BEk,'AFk':AFk,'BEplusAF':(BEk+AFk)},
                            name=StateID,
                        )
                    )
                else:
                    P.loc[StateID,'SumWik'] += temp.loc[StateID,'StateOrder']
                    P.loc[StateID,'Pk'] += temp.loc[StateID,'StateCount']
                    P.loc[StateID,'iPk'] += 1
                    P.loc[StateID,'BEk'] += BEk
                    P.loc[StateID, 'AFk'] += AFk
                    P.loc[StateID, 'BEplusAF'] += (BEk+AFk)
        return P

    def Im_s(self, P, tempn):

        #C = P[P['BEplusAF'] == P['BEplusAF'].max()]  # sort
        C = P.sort_values(by = "BEplusAF",ascending = False)
        print("max stateID \n", C)
        for StateID in C.index:
            self.check_state_exist_Im(StateID)
            if self.Im.loc[StateID, 'beforestate']>0:
                continue
            else:
                self.Im.loc[StateID, 'sta_visul'] = P.loc[StateID, 'sta_visul']
                self.Im.loc[StateID, 'ImS'] = tempn * (tempn + 1) / 2
                self.Im.loc[StateID, 'beforestate'] =1
                break

        return self.Im
    def Im_s1(self,P,tempn):

        C = P[P['BEplusAF'] == P['BEplusAF'].max()]  # argmax??
        print("max stateID \n", C)
        for StateID in C.index:
            self.check_state_exist_Im(StateID)
            self.Im.loc[StateID,'sta_visul']= P.loc[StateID,'sta_visul']
            self.Im.loc[StateID, 'ImS'] = tempn*(tempn+1)/2

        return self.Im

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        #self.check_state_exist_Im(s_)
        #print('q_table before update \n', self.q_table)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            #q_target = r + self.Im.loc[s_,'ImS']+ self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
            q_target = r + self.gamma * self.q_table.loc[s_,:].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        #print('q_table after update \n',self.q_table)
    def check_state_exist_Im(self, state):
        if state not in self.Im.index:
            # append new state to q table
            self.Im = self.Im.append(
                pd.Series(
                    [0]*4,
                    index=self.Im.columns,
                    name=state,
                )
            )

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    def resetIm(self,trj,reward):
        print("we get R, resetIm")

        C = self.stochastic_trj(trj)
        i = 1
        #print("Im be \n",self.Im)
        for StateID in C.index:
            i=i+1
            maxIm = self.Im['ImS'].max()
            self.Im.loc[StateID, 'ImS'] = maxIm+i
            self.Im.loc[StateID, 'sta_visul'] = C.loc[StateID, 'sta_visul']
            self.Im.loc[StateID, 'beforestate'] = 2



        #print("Im AF \n",self.Im)
        #print("C \n",C)
    def getstatereset(self):
        for StateID in self.Im.index:
            self.Im.loc[StateID,'getstate']=0

    def resetIm1(self,trj,reward):
        print("we get R, HOW TO DO")
        C=self.stochastic_trj(trj)
        i = 1
        print("Im be \n",self.Im)
        for StateID in C.index:
            i=i+1
            self.check_state_exist_Im(StateID)
            maxIm = self.Im['ImS'].max()
            if self.Im.loc[StateID, 'beforestate']>0:
                continue
            else:
                if i>len(C):
                    self.Im.loc[StateID, 'ImS'] = reward * i * (i + 1) / 2
                else:
                    self.Im.loc[StateID, 'ImS'] = reward* i * (i + 1) / 2
                    self.Im.loc[StateID, 'beforestate'] = 1

        print("Im AF \n",self.Im)
        print("C \n",C)


class InternalModel(object):
    """
    Description:
        We'll create a tabular model for our simulated experience. Please complete the following code.
    """

    def __init__(self):
        # self.model = dict()
        self.model = []
        self.rand = np.random

    def store(self, state, action, next_state, reward):
        """
        TODO:
            Store the previous experience into the model.
        Return:
            NULL
        """
        exp = (state, action, next_state, reward)
        self.model.append(exp)

    def sample(self):
        """
        TODO:
            Randomly sample previous experience from internal model.
        Return:
            state, action, next_state, reward
        """
        import random
        (stateP, actionP, next_stateP, rewardP) = random.sample(self.model, 1)[0]
        return (stateP, actionP, next_stateP, rewardP)
    def check(self,state, action, next_state, reward):
        exp = (state, action, next_state, reward)
        if exp in self.model:
            return True
        else:
            return False


