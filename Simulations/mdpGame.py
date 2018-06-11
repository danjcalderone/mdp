# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:36:45 2018

@author: sarah
"""
import mdpRoutingGame as mrg
import mdp as mdp
import dynamicProgramming as dp

import numpy as np
Time = 20;
grid = mrg.gParam("grid", 3,5);
p0 = np.zeros((3*5));
p0[0] = 1.0;

gridGame = mrg.mdpRoutingGame(grid,Time);
print "Solving primal unconstrained case";
optRes = gridGame.solve(p0, verbose=False,returnDual=False);

gridGame.setConstrainedState(6);
print "Solving constrained case, state 6 <= 0.2 case";
optCRes = gridGame.solveWithConstraint(p0,verbose = False);
print "optimal dual: ", gridGame("optDual")
#mdp.drawOptimalPopulation(Time,
#                          gridGame("graphPos"),
#                          gridGame("G"),
#                          optCRes, 
#                          is2D = False, 
#                          constrainedState = gridGame("constrainedState"));

# solve problem again using unconstrained case: 
print "Solving unconstrained problem with new Toll";
optCSol = gridGame.solve(p0,withPenalty=True,verbose = False, returnDual = False)
#mdp.drawOptimalPopulation(Time,
#                          gridGame("graphPos"),
#                          gridGame("G"),
#                          optCSol, 
#                          is2D = False, 
#                          constrainedState = gridGame("constrainedState"));
# for dynamic programming
cR = mdp.constrainedReward3D(gridGame("reward"),
                             gridGame("optDual") + 0.01, # make dual the interiors
                             gridGame("constrainedState"));
                             
dpVC, dpSolC = dp.dynamicP(cR,gridGame("probability"),p0);
print 0.2*sum(gridGame("optDual"))

#to draw any of these solutions
#mdp.drawOptimalPopulation(Time,
#                          gridGame("graphPos"),
#                          gridGame("G"),
#                          dpSolC, 
#                          # only set is2D to true for dynamic programming
#                          is2D = True, 
#                          # only put in contraint when there is constraint
#                          constrainedState = gridGame("constrainedState"));