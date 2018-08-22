# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:06:00 2018

@author: craba
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:10:25 2018

@author: sarah
"""
import Algorithms.mdpRoutingGame as mrg
import util.mdp as mdp
import Algorithms.dynamicProgramming as dp
import util.utilities as ut


import numpy as np
import numpy.linalg as la
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
numPlayers = 100;
#p0[0] = 1.0*numPlayers;
# make all drivers start from residential areas 6 of them
residentialNum = 6;
residentialList = [2,3,7,8,10,11];
p0[2] = 1.*numPlayers/residentialNum;
p0[3] = 1.*numPlayers/residentialNum;
p0[7] = 1.*numPlayers/residentialNum;
p0[8] = 1.*numPlayers/residentialNum;
p0[10] =1.*numPlayers/residentialNum;
p0[11] =1.*numPlayers/residentialNum;

print "Solving primal unconstrained case";
optRes, optObj = sGame.solve(p0, verbose=False,returnDual=False);
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optRes,
#                          startAtOne = True,
#                          numPlayers= p0[0]);

#cleanOpt = ut.truncate(optRes);
#costs, yt = dp.MSA(sGame.States, sGame.Actions, sGame.Time, 
#                               p0,
#                               sGame("reward"),
#                               sGame("C"),
#                               sGame("probability"));
#                               
#print"----------    Dynamic Programming value     --------------";
#yDP = ut.truncate(yt);
#totalDiff = la.norm(ut.truncate(abs(yDP - cleanOpt)))
#print "Maximum difference   ", (ut.truncate(abs(yDP - cleanOpt))).max()/numPlayers;
    

cState = 6;      
cThresh = 0.2*numPlayers;   
isLB = True;                      
sGame.setConstrainedState(cState, cThresh, isLB = isLB);
print "Solving constrained case, state 7 >= 150 case";
optCRes = sGame.solveWithConstraint(p0,verbose = False);
#print "optimal dual: ", sGame("optDual")
#print "upper bound" , sGame("constrainedUpperBound")
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optCRes/10.,
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
#                          constrainedUpperBound = sGame("constrainedUpperBound"));
if isLB: 
    optimalDual = np.concatenate((np.zeros(3), sGame("optDual") )) + 0.01;
else:
    optimalDual = -np.concatenate((np.zeros(3), sGame("optDual"))) + 0.01;
#####
print "Solving unconstrained problem with new Toll";
optCSol, optTolledObj = sGame.solve(p0,withPenalty=True,verbose = False, returnDual = False)
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optCSol/10., 
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
##                          constrainedUpperBound = sGame("constrainedUpperBound"));
##     

cleanOpt = ut.truncate(optCSol);
barC =  sGame("C") + ut.toll2Mat(cState, optimalDual, [sGame.States, sGame.Actions, sGame.Time], isLB);

# Simulate the values converging to optimal solution
threshIter = 20;
ytThresh = np.zeros((sGame.States, sGame.Actions, sGame.Time, threshIter));
threshVal = np.zeros((threshIter));
normDiff = np.zeros((threshIter, 2)); # two norm and infinity norm
for iter in range(threshIter):
    print "iteration = ", iter;
    threshVal[iter] = 91.0 - (iter/2.)**2;    
    costs, ytThresh[:,:,:,iter] = dp.MSA(sGame.States, sGame.Actions, sGame.Time, 
                                   p0,
                                   sGame("reward"),
                                   barC,
                                   sGame("probability"),
                                   threshVal[iter] );
    yDP = ut.truncate(ytThresh[:,:,:,iter]);
    normDiff[iter, 0] = la.norm(ut.truncate(abs(yDP - cleanOpt)))/numPlayers;
    normDiff[iter, 1] = (ut.truncate(abs(yDP - cleanOpt))).max()/numPlayers;
    
#print"----------    Dynamic Programming value     --------------";
#yDP = ut.truncate(yt);
#totalDiff = la.norm(ut.truncate(abs(yDP - cleanOpt)))
#print "Maximum difference   ", (ut.truncate(abs(yDP - cleanOpt))).max()/numPlayers;
                   
timeLine = np.linspace(1,Time,20)
fig = plt.figure();  
plt.plot(timeLine,np.sum(optCSol[cState,:,:],axis=0),label = "primal constrained trajectory"); 
for i in range(threshIter):         
    plt.plot(timeLine,np.sum(ytThresh[cState,:,:, i],axis=0),dashes=[4, 2],label =r'$\epsilon$ = %.2f'%(threshVal[i])); 
    
plt.legend(fontsize = 'xx-small');
#plt.title("Optimal Dual Solution with Decreasing Termination Tolerance")
plt.xlabel("Time");
plt.ylabel("Number of Drivers")
plt.show();

fig = plt.figure();
plt.plot(threshVal, normDiff[:,0], label = r'$||\cdot||_2$');
plt.plot(threshVal, normDiff[:,1], label = r'$||\cdot||_{\infty}$');
plt.legend();
#plt.title("Difference in Norm as a function of termination tolerance")
plt.xlabel("Termination tolerance");
plt.ylabel("$||y - y^*||$")
plt.xscale("log")
plt.show();