from baselines.SPPI.miniMaze.minimaze import Maze
from baselines.SPPI.miniMaze.RL_brain_QLearning import QLearningTable
def main_MAZE(env):
    n_trj = 1000
    RL = QLearningTable(actions=list(range(env.n_actions)))
    for eps in range(n_trj):
        observation = env.reset()
        step = 0

        while step <100:
            step +=1
            env.render()

            #action = RL.random_action(str(observation))
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            # if reward == -1:
            #     break
            RL.learn(str(observation),action,reward,str(observation_))
            observation = observation_

            if done:
                print("done!")
                break



if __name__ == "__main__":
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    # main_MR()
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env)