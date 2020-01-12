# graph based memory reconstruction agents
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GMRAgent:
    def __init__(self, actions,e_greedy=0.95):
        self.actions=actions
        #print(actions)
        self.epsilon=e_greedy
        self.node_vec=[]
        self.Gmemory=nx.DiGraph()
        
    def obs2state(self,observation):
        state=str(observation)
        return state
    def random_action(self,state):
        action = np.random.choice(self.actions)
        return action

    def MemoryWriter(self, state, action, reward, state_):
        '''
        节点是状态，用一个列表表示（状态本身的特征需要另外表示，比如，出现次数，距离远近，用轨迹算得到的值函数）
        边是动作转移（边的特征可能会有很多，比如，出现次数，状态动作值函数，针对确定性环境，一个状态动作只能对应下一个状态动作）
        (这里只记一步可达状态作为边，权重先用reward)
        '''
        if state not in self.node_vec:
            self.node_vec.append(state)
        self.Gmemory.add_node(state)
        self.Gmemory.add_edge(state,state_,weight=-reward,labels=action)
        
    def plotGmemory(self):
        print('节点向量的长度',len(self.node_vec))
        print("输出全部节点：{}".format(self.Gmemory.nodes()))
        print("输出全部边：{}".format(self.Gmemory.edges()))
        print("输出全部边的数量：{}".format(self.Gmemory.number_of_edges()))
        nx.draw(self.Gmemory)
        plt.show()

    def MemoryReader(self, state):
        #与state一步相连的状态
        temp = self.Gmemory[state]
        next_state_candidates=list(temp)
        print('next_state_candidates',next_state_candidates)   
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
        '''
        [action_candidates,value_list,next_state_candidates]=self.MemoryReader(state)
        return action_candidates,value_list

    def ActAccordingToGM(self, state):
        '''
        在记忆库中查询state
        进行推演
        返回各个可能动作对应的值函数

        '''
        if self.check_state_exist(state):
            #之前存在（相似状态），找到相应的值，按照贪心策略执行 
            if np.random.uniform() < self.epsilon:
                action_candidates,action_values= self.get_action_value(state)
                # some actions may have the same value, randomly choose on in these actions
                if len(action_candidates)==0:
                    action = np.random.choice(self.actions)
                else:
                    max_SA=np.max(action_values)
                    actions=[]
                    for i in range(len(action_values)):
                        if action_values[i]==max_SA:
                            actions.append(action_candidates[i])#这里宜直接使用动作的索引，而不是用i
                    action = np.random.choice(actions)
            else:
                # choose random action
                action = np.random.choice(self.actions)
        else:
            #之前不存在，
            action = np.random.choice(self.actions)
        return action
    
