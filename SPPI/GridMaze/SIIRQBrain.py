
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
        print("self.q",self.q_table)

    # def check_state_exit(self,state,attribute):
    #     if sta in self.StateAttributesDict:
    #         return  True
    #     else:
