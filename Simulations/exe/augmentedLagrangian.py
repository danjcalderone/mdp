# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:23:20 2018

@author: craba
"""

import Algorithms.mdpRoutingGame as mrg
import util.mdp as mdp
import util.utilities as ut
import Algorithms.dynamicProgramming as dp

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ------------- Part 0: Define parameters and solve social game ------------ #
sysStart = time.time();
Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);
socialGame = mrg.mdpRoutingGame(seattle,Time);
seattleGraph=socialGame("G");
socialGame.setQuad();

p0 = mdp.resInit();

socialRes, socialObj = socialGame.solve(p0, verbose=False,returnDual=False, isSocial = True );
#-----------Part 1: define feasible flows and constraints ------------------- #
userGame = mrg.mdpRoutingGame(seattle,Time);
userGame.setQuad();

thresh = 1e-1;
costs, y0 = dp.MSA(userGame.States, userGame.Actions, userGame.Time, 
                   p0,
                   userGame("reward"),
                   userGame("C"),
                   userGame("probability"),
                   thresh); # y0 = state x action x time

                   
tolls = 1.;    
optDiff = ut.truncate(socialRes- y0, tolls);
nonZeroDiff = ut.nonZeroEntries(optDiff);
constraintList = ut.constraints(nonZeroDiff, socialRes, y0);
constraintNumber = len(constraintList);


# Initial mu guess = constraintList
# Initial flow guess = f0

# optCRes = sGame.solveWithConstraint(p0,verbose = False, constraintList = constraintList);
# optDual = ut.matGen(constraintList, sGame.optimalDual, [sGame.States, sGame.Actions, Time]);
# tolledObj[incre] = sGame.socialCost(optCRes); # +  tolls[incre]* np.sum(np.multiply(optDual, optCRes));
# print "Social cost at constrained User Optimal";
# print tolledObj[incre];