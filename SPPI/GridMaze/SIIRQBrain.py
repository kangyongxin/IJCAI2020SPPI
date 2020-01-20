
import pandas as pd
import numpy as np

class SIIRQBrain:
    def __init__(self,env,learning_rate=0.01, reward_decay=0.9, e_greedy=0.95):
        self.env = env
        self.actions = env.action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.StateAttributesDict=pd.DataFrame(columns=['attribute'])
        self.StateFunctionValueDict = pd.DataFrame(columns=['StateValue','FirstVisit'])
        self.getRflag = False

    def random_action(self,state):
        action = np.random.choice(self.actions)
        return action

    def obs2state(self,obs):
        if obs == 'terminal':
            line = int(self.env.MAZE_H-1 ) #从 00，开始编码
            row = int(self.env.MAZE_W-1)
        else:
            line = int((obs[1] + 15.0 - 20.0) / 40)
            row=int((obs[0] + 15.0 - 20.0) / 40)
        sta =str(list([line,row]))
        feature = list([line, row])
        if sta not in self.StateAttributesDict.index:
            self.StateAttributesDict = self.StateAttributesDict.append(
                pd.Series(
                    {'attribute' : feature},
                    name = sta,
                )
            )
        return sta

    def check_state_exist_in_q(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


    def choose_action_q(self,state):
        self.check_state_exist_in_q(state)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action= self.q_table.loc[state, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        #print(action)
        return action


    def learn_q(self, s, a, r, s_):
        self.check_state_exist_in_q(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        #print("self.q",self.q_table)
    def learn_SIq(self, s, a, r, s_):
        self.check_state_exist_in_q(s_)
        if s_ in self.StateFunctionValueDict.index:
            #print("rold:",r)
            r= r+self.StateFunctionValueDict.loc[s_,'StateValue']
            #print("rnew :",r)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        #print("self.q",self.q_table) 
    def stochastic_trj(self,seq):
        sta = seq[0]
        sta_attribute = self.StateAttributesDict.loc[sta,'attribute']
        Ci = pd.DataFrame([{'sta_attribute':sta_attribute,'StateCount':seq.count(sta),'StateOrder':1}],index=[sta],columns=['sta_attribute','StateCount','StateOrder'], dtype=np.float64)
        k= 1
        for sta in seq:
            sta_attribute = self.StateAttributesDict.loc[sta,'attribute']
            if sta not in Ci.index:
                k += 1
                Ci=Ci.append(
                    pd.Series({'sta_attribute':sta_attribute,'StateCount':seq.count(sta),'StateOrder':k},
                              name=sta,
                              )
                )
        return Ci

    def stochastic_trjs(self,trjs):
        seq = trjs[0] #第一条轨迹
        sta = seq[0] #第一个状态
        sta_attribute = self.StateAttributesDict.loc[sta,'attribute']
        iPk = 1
        BEk = 0
        AFk = 0
        P= pd.DataFrame([{'sta_attribute':sta_attribute,'StateOrder':1,'SumWik':0,'Pk':seq.count(sta),'iPk':iPk,'BEk':BEk,'AFk':AFk,'BEplusAF':(BEk+AFk)}],index=[sta],columns=['sta_attribute','StateOrder','SumWik','Pk','iPk','BEk','AFk','BEplusAF'], dtype=np.float64)

        k=1
        for trj in trjs:
            #每一条轨迹,
            temp = self.stochastic_trj(trj)

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
                            {'sta_attribute': temp.loc[StateID,'sta_attribute'],'StateOrder': k, 'SumWik':temp.loc[StateID,'StateOrder'],'Pk':seq.count(sta),'iPk':iPk,'BEk':BEk,'AFk':AFk,'BEplusAF':(BEk+AFk)},
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
    
    def SI(self,p,tempn):
        #rint("before  we do the dict is ",self.StateFunctionValueDict)
        tempdict=pd.DataFrame(columns=['StateValue'])
        for StateID in p.index:
            sumk=p.loc[StateID,'BEplusAF'] #对比实验的时候要单独用它
            af = p.loc[StateID,'AFk']
            be = p.loc[StateID,'BEk']
            pk = p.loc[StateID,'Pk']
            #print("stateID",StateID,"sum",sumk,"pk",pk)
            tempdict = tempdict.append(
                pd.Series(
                    {'StateValue': (be/pk)},
                    name = StateID,
                )
            )
        #print("curren dict",tempdict)
        C = tempdict.sort_values(by = "StateValue",ascending = False)
        #print("c",C)
        #print("sorted dict",C)# 所有之前见过的状态中只留一个
        for StateID in  C.index:
            if StateID not in self.StateFunctionValueDict.index:
                for st in self.StateFunctionValueDict.index:
                    self.StateFunctionValueDict.loc[st,'FirstVisit']=False

                self.StateFunctionValueDict= self.StateFunctionValueDict.append(
                    pd.Series(
                        {'StateValue':10*tempn*(tempn+1)/2,'FirstVisit': True},
                        name = StateID,
                    )
                )
                maxid = StateID
                break
            else: 
                self.StateFunctionValueDict.loc[StateID,'StateValue'] = 0
        return self.StateFunctionValueDict, maxid
    # def check_state_exit(self,state,attributen
    #     if sta in self.StateAttributesDict:
    #         return  True
    #     else:
