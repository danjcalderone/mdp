# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:10:25 2018

@author: sarah
"""
import mdpRoutingGame as mrg
import mdp as mdp
import dynamicProgramming as dp



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);

sGame = mrg.mdpRoutingGame(seattle,Time);
seattleGraph=sGame("G");
sGame.setQuad();
#nx.draw(seattleGraph, pos = sGame("graphPos"),with_labels=True);
#plt.show()
#p0 = np.ones((seattleGraph.number_of_nodes()))/seattleGraph.number_of_nodes();
p0 = np.zeros((seattleGraph.number_of_nodes()));
#p0[0] = 1.0;
# make all drivers start from residential areas 6 of them
residentialNum = 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

print "Solving primal unconstrained case";
optRes = sGame.solve(p0, verbose=False,returnDual=False);
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optRes/10.,
#                          startAtOne = True);
#
cState = 6;                               
sGame.setConstrainedState(cState, 10);
print "Solving constrained case, state 7 >= 0.5 case";
optCRes = sGame.solveWithConstraint(p0,verbose = False);
print "optimal dual: ", sGame("optDual")
print "upper bound" , sGame("constrainedUpperBound")
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optCRes/10.,
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
#                          constrainedUpperBound = sGame("constrainedUpperBound"));
####
print "Solving unconstrained problem with new Toll";
optCSol = sGame.solve(p0,withPenalty=True,verbose = False, returnDual = False)
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optCSol/10., 
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
#                          constrainedUpperBound = sGame("constrainedUpperBound"));
#     
# plot constrained state
#timeLine = np.linspace(1,Time,20)
#cTraj = np.sum(optCSol[cState,:,:],axis=0)           
#traj = np.sum(optRes[cState,:,:],axis=0)  
#fig = plt.figure();  
#plt.plot(timeLine,traj,label = "unconstrained trajectory");
#plt.plot(timeLine,cTraj,label = "constrained trajectory"); 
#plt.legend();
#plt.title("State 7 Constrained vs Unconstrained Trajectories")
#plt.xlabel("Time");
#plt.ylabel("Driver Density")
#plt.show();
#yT = optRes[:,:,Time-1];                           
#print "Solving dynamic programming problem of unconstrained problem"; 
#cR = mdp.constrainedReward3D(sGame("reward"),
#                             sGame("optDual") + 0.01, # make dual the interiors
#                             sGame("constrainedState"));         
tolls = np.concatenate((np.zeros(3),sGame("optDual")));         
dpVC, dpSolC = dp.dynamicPLinearCost(sGame("reward"),
                                     sGame("C"),
                                     sGame("probability"),
                                     optCSol,
                                     p0, 
                                     hasToll = True,
                                     toll = tolls + 0.01, 
                                     tollState = 6);
                                     
                                     
#optTraj_old = np.einsum("ijk->ik", optCSol); 

    
mdp.drawOptimalPopulation(Time,
                          sGame("graphPos"),
                          sGame("G"),
                          dpSolC, 
                          constrainedState = sGame("constrainedState"), 
                          constrainedUpperBound = sGame("constrainedUpperBound"),
                          # only set is2D to true for dynamic programming
                          is2D = True, 
                          startAtOne = True);
                          
timeLine = np.linspace(1,Time,20)
cTraj = np.sum(optCSol[cState,:,:],axis=0)
dpTraj = dpSolC[cState,:];           
traj = np.sum(optRes[cState,:,:],axis=0)  
fig = plt.figure();  
plt.plot(timeLine,traj,label = "unconstrained trajectory");
plt.plot(timeLine,cTraj,label = "constrained trajectory"); 
plt.plot(timeLine,dpTraj,label = "dynamic trajectory"); 
plt.legend();
plt.title("State 7 Constrained vs Unconstrained Trajectories")
plt.xlabel("Time");
plt.ylabel("Driver Density")
plt.show();
yT = optRes[:,:,Time-1];                           
print "Solving dynamic programming problem of unconstrained problem"; 

