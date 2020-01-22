
from SPPI.miniMaze.minimaze import Maze
from SPPI.miniMaze.RL_brain import DemoSV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def ShowInHeatmap(p,S_space):
    pk = np.zeros(len(S_space))
    be = np.zeros(len(S_space))
    af = np.zeros(len(S_space))
    fin = np.zeros(len(S_space))
    print(p.shape[0])
    t=p.iloc[2]
    print(t.loc["SumWik"])
    print(p.iloc[2,2])
    for i in S_space:
        for j in range(p.shape[0]):
            if int(i)==int(p.iloc[j,0]):#'sta_visul'
                pk[i-1] = p.iloc[j,3]#'Pk'
                be[i-1] = p.iloc[j,5]
                af[i-1] = p.iloc[j,6]
                if pk[i-1]==0:
                    fin[i-1]=0#be[i-1]+af[i-1]
                else:
                    fin[i-1]=(be[i-1]+af[i-1])/np.sqrt(pk[i-1])#(pow(pk[i-1],1/3)+np.sqrt(pk[i-1]))
    pk[0]=0
    be[0]=0
    af[0]=0
    print("pk",pk,"\n","be",be,"\n","af",af,"\n",'fin',fin,"\n")
    max_pk = pk.max()
    min_pk = pk.min()
    max_be = be.max()
    min_be = be.min()
    max_af = af.max()
    min_af = af.min()
    sumk = af + be
    max_sum = sumk.max()
    min_sum = sumk.min()
    max_fin = fin.max()
    min_fin = fin.min() 

    

    for te in range(len(S_space)):
        pk[te] = (pk[te]-min_pk)/(max_pk-min_pk)
        be[te] = (be[te]-min_be)/(max_be-min_be)
        af[te] =  (af[te]-min_af)/(max_af-min_af)
        sumk[te] = (sumk[te]-min_sum)/(max_sum-min_sum)
        fin[te] = (fin[te]-min_fin)/(max_fin - min_fin)


        
    pkarray= np.zeros([6,6])
    bearray= np.zeros([6,6])
    afarray= np.zeros([6,6])
    sumkarray = np.zeros([6,6])
    finarray = np.zeros([6,6])
    for i in range(6):
        for j in range(6):
            pkarray[i,j]= pk[i*6+j]
            bearray[i,j]= be[i*6+j]
            afarray[i,j]= af[i*6+j]
            sumkarray[i,j]=sumk[i*6+j]
            finarray[i,j]=fin[i*6+j]
    sns.heatmap(afarray, cmap='Reds')
    plt.show()
    sns.heatmap(pkarray,cmap='Blues')
    plt.show()
    sns.heatmap(bearray,cmap='Greens')
    plt.show()
    sns.heatmap(sumkarray,cmap='Greys')
    plt.show()
    sns.heatmap(finarray,cmap='YlGnBu')
    plt.show()


    
'''
Colormap Pink is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, 
CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r,
 Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, 
Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, 
Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, 
YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool,
 cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, 
gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, 
gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, 
hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r,
 pink, pink_r, plasma, plasma_r, prism, prism_r, 
rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r,
 tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r
'''

def main_MAZE(env,S_space):
    n_trj = 1000
    RL = DemoSV(env,actions=list(range(env.n_actions)))
    trjs =[]
    for eps in range(n_trj):
        trj=[]
        observation = env.reset()
        state=RL.obs2state(observation)
        step = 0
        trj.append(state)
        while step <100:
            step +=1
            env.render()

            action = RL.random_action(str(observation))
            observation_, reward, done = env.step(action)

            state_ = RL.obs2state(observation_)
            trj.append(state_)

            observation = observation_

            if done:
                print("done!")
                break
        trjs.append(trj)
    # #show trjs
    # for trj in trjs:
    #     print("trj :")
    #     for i in range(len(trj)):
    #         print(trj[i])
    #     print("\n")

    p = RL.stochastic_trjs(trjs)
    print(p)

    ShowInHeatmap(p,S_space)
    

    


if __name__ == "__main__":
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    # main_MR()
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env,S_space)