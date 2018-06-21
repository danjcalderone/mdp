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
seattle = mrg.gParam("seattle", None, None);

sGame = mrg.mdpRoutingGame(seattle,Time);
seattleGraph=sGame("G");
sGame.setQuad();
#nx.draw(seattleGraph, pos = sGame("graphPos"),with_labels=True);
#plt.show()
p0 = np.ones((seattleGraph.number_of_nodes()))/seattleGraph.number_of_nodes();
#p0 = np.zeros((seattleGraph.number_of_nodes()));
#p0[0] = 1.0;


print "Solving primal unconstrained case";
optRes = sGame.solve(p0, verbose=False,returnDual=False);
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optRes,
#                          startAtOne = True);
#

#yT = optRes[:,:,Time-1];                           
print "Solving dynamic programming problem of unconstrained problem"; 
##cR = mdp.constrainedReward3D(sGame("reward"),
##                             sGame("optDual") + 0.01, # make dual the interiors
##                             sGame("constrainedState"));                           
dpVC, dpSolC = dp.dynamicPLinearCost(sGame("reward"),
                                     sGame("probability"),
                                     optRes,
                                     p0);
mdp.drawOptimalPopulation(Time,
                          sGame("graphPos"),
                          sGame("G"),
                          dpSolC, 
                          # only set is2D to true for dynamic programming
                          is2D = True, 
                          startAtOne = True);

                                     
#sGame.setConstrainedState(6, 0.5);
#print "Solving constrained case, state 7 >= 0.5 case";
#optCRes = sGame.solveWithConstraint(p0,verbose = False);
#print "optimal dual: ", sGame("optDual")
#print "upper bound" , sGame("constrainedUpperBound")
##mdp.drawOptimalPopulation(Time,
##                          sGame("graphPos"),
##                          sGame("G"),
##                          optCRes,
##                          startAtOne = True,
##                          constrainedState = sGame("constrainedState"), 
##                          constrainedUpperBound = sGame("constrainedUpperBound"));
####
#print "Solving unconstrained problem with new Toll";
#optCSol = sGame.solve(p0,withPenalty=True,verbose = False, returnDual = False)
##mdp.drawOptimalPopulation(Time,
##                          sGame("graphPos"),
##                          sGame("G"),
##                          optCSol, 
##                          startAtOne = True,
##                          constrainedState = sGame("constrainedState"), 
##                          constrainedUpperBound = sGame("constrainedUpperBound"));
#                          
#yT = optCSol[:,:,Time-1];                           
#print "Solving dynamic programming problem with new Toll"; 
##cR = mdp.constrainedReward3D(sGame("reward"),
##                             sGame("optDual") + 0.01, # make dual the interiors
##                             sGame("constrainedState"));                           
#dpVC, dpSolC = dp.dynamicPLinearCost(sGame("reward"),
#                                     sGame("probability"),
#                                     yT,
#                                     p0, 
#                                     hasToll =True,
#                                     toll = sGame("optDual") + 0.01,
#                                     tollState = 6);
##print sGame("constrainedUpperBound")*sum(sGame("optDual"))                          
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          dpSolC, 
#                          # only set is2D to true for dynamic programming
#                          is2D = True, 
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
#                          constrainedUpperBound = sGame("constrainedUpperBound"));
                             


