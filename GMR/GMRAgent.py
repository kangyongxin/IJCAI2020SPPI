# graph based memory reconstruction agents
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Canopy import Canopy as Cluster

class GMRAgent:
    def __init__(self, actions,e_greedy=0.95,gamma=0.9,lr=0.01):
        self.actions=actions
        #print(actions)
        self.epsilon=e_greedy
        self.gamma = gamma
        self.lr = lr
        self.node_vec=[]
        self.Gmemory=nx.DiGraph()
        self.StateAttributesDict={}
        
        
    def obs2state(self,observation):
        state=str(observation)
        if observation == 'terminal':
            self.StateAttributesDict[state]=list([165.0,165.0,195.0,195.0])
        else:
            self.StateAttributesDict[state]=observation #为了把值传到后面重构部分进行计算
        return state

    def random_action(self,state):
        action = np.random.choice(self.actions)
        return action

    def MemoryWriter(self, re_vec):
        '''
        输入是一条轨迹，先把每个状态上的值算出来
        '''

        # G = 0 
        for i in range(len(re_vec)):
        #for i in range(len(re_vec)-1,-1,-1):
           
            self.PairWriter(re_vec[i][0],re_vec[i][1],re_vec[i][2],re_vec[i][3])

    def update_edge(self,state, action, w, state_):
        old_w= self.Gmemory.edges[state,state_]['weight']
        old_visits = self.Gmemory.edges[state,state_]['visits']
        if state_ != 'terminal':
            edge_list = self.Gmemory.edges(state_)
            if len(edge_list)==0:
                #下一个状态既不是终点
                tar = w
            else:
                weight_list = []
                for edge_i in edge_list:
                    weight_list.append(self.Gmemory.edges[edge_i]['weight'])
                weight_max = np.max(weight_list)
                tar =  w + self.gamma*weight_max
        else:
            tar = w
        delta_w = tar - old_w
        new_w = old_w + self.lr*delta_w
        self.Gmemory.add_edge(state,state_,weight=new_w,labels=action,visits=old_visits+1)

    def PairWriter(self, state, action, w, state_):
        '''
        节点是状态，用一个列表表示（状态本身的特征需要另外表示，比如，出现次数，距离远近，用轨迹算得到的值函数）
        边是动作转移（边的特征可能会有很多，比如，出现次数，状态动作值函数，针对确定性环境，一个状态动作只能对应下一个状态动作）
        (这里只记一步可达状态作为边，权重先用reward)
        '''
        if [state,state_] in self.Gmemory.edges():
            #修边，用state_对应的边权来修正state,action的比安全
            self.update_edge(state,action,w,state_)
        else:
            #加边
            if self.check_state_exist(state):
                if self.check_state_exist(state_):
                    pass
                else:
                    self.Gmemory.add_node(state_,attributes=self.StateAttributesDict[state_])
            else:
                if self.check_state_exist(state_):
                    self.Gmemory.add_node(state,attributes=self.StateAttributesDict[state])
                else:
                    self.Gmemory.add_node(state,attributes=self.StateAttributesDict[state])
                    self.Gmemory.add_node(state_,attributes=self.StateAttributesDict[state_])
            self.Gmemory.add_edge(state,state_,weight=w,labels=action,visits=1)

    def plotGmemory(self):
        #print('节点向量的长度',len(self.node_vec))
        print("输出全部节点：{}".format(self.Gmemory.nodes()))
        # print("输出全部边：{}".format(self.Gmemory.edges()))
        print("输出全部边的数量：{}".format(self.Gmemory.number_of_edges()))
        nx.draw(self.Gmemory)
        plt.show()

    def MemoryReader(self, state):
        #与state一步相连的状态
        temp = self.Gmemory[state]
        next_state_candidates=list(temp)
        #print('next_state_candidates',next_state_candidates)   
        action_candidates=[]
        value_list =[]
        for i in range(len(next_state_candidates)):
            action_candidates.append(self.Gmemory.edges[state,next_state_candidates[i]]['labels'])
            value_list.append(self.Gmemory.edges[state,next_state_candidates[i]]['weight'])
        return action_candidates,value_list,next_state_candidates

    def check_state_exist(self,state):
        '''
        检查是否见过该状态，或者相近状态，如果没有，新加；有，返回中心值
        '''
        if state in list(self.Gmemory.nodes):
            return True
        else:
            return False

    def get_action_value(self, state):
        '''
        根据图中节点找到可执行的边的权重??????
        同时还要加入未执行的边，零权重
        '''
        [action_candidates,value_list,next_state_candidates]=self.MemoryReader(state)
        for i in range(len(self.actions)):
            if self.actions[i] in action_candidates:
                pass
            else:
                action_candidates.append(self.actions[i])
                value_list.append(0)
        return action_candidates,value_list

    def ActAccordingToGM(self, state):
        '''
        在记忆库中查询state
        进行推演
        返回各个可能动作对应的值函数
        '''
        #print("actions",self.actions)
        if self.check_state_exist(state):
            #之前存在（相似状态），找到相应的值，按照贪心策略执行 
            if np.random.uniform() < self.epsilon:
                action_candidates,action_values= self.get_action_value(state)
                # some actions may have the same value, randomly choose on in these actions
                max_SA=np.max(action_values)
                acts=[]
                for i in range(len(action_values)):
                    if action_values[i]==max_SA:
                        acts.append(action_candidates[i])#这里宜直接使用动作的索引，而不是用i
                #print("candidates acts",acts)
                #print("action_values",action_values)
                action = np.random.choice(acts)
            else:
                # choose random action
                action = np.random.choice(self.actions)
        else:
            #之前不存在，
            action = np.random.choice(self.actions)
        return action
    
    def MemoryReconstruction(self,t1,t2):
        dataset=[]
        for node in list(self.Gmemory.nodes()):
            dataset.append(self.Gmemory.nodes[node]['attributes'])
        print("dataset",dataset)
        gc = Cluster(dataset)
        gc.setThreshold(t1,t2)
        canopies = gc.clustering()
        print('Get %s initial centers.' % len(canopies))
        for i in range(len(canopies)):
            print(canopies[i])
