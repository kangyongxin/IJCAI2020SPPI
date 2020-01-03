# IJCAI2020SPPI

## catalogue

### 1 MiniMaze 

### 2 Grid Maze

    +  Q
    +  SI Q
    +  SI Q( self attributes only)
    +  SI Q( relationship only)

### 3  Atari 2600

    + count
    + sil
    + count+sil
    + sppi
    + sppi + sil
    + a2c
    + a2c + SI

### 4 Mujoco

    + HER
    + SI-A2C
    + Count

## 1 Minimaze

\# python -m SPPI.miniMaze.run_tests

\# python -m SPPI.miniMaze.run_demovs

we get 3 heatmaps 

## 2 Grid Maze

### Q

\# python -m SPPI.GridMaze.run_tests

\# python -m SPPI.GridMaze.run_qlearning

how to evaluate the results: total steps or rewards ?


### SI-Q

\# python -m SPPI.GridMaze.run_SIQ

a new framework should be proposed here, ablation experiments

analyze the hyper Paras(N,N0,max, r,gamma, beta) , illustrate by the experiments results

