# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 12:56:58 2018

@author: craba
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import Algorithms.mdpRoutingGame as mrg
import Algorithms.frankWolfe as fw
import util.mdp as mdp
import Algorithms.dualAscent as da
import util.utilities as ut
plt.close('all');
Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);
#----------------set up game -------------------#
sGame = mrg.mdpRoutingGame(seattle,Time);
seattleGraph=sGame("G");
sGame.setQuad();
numPlayers = 100;
p0 = mdp.resInit(seattleGraph.number_of_nodes(), residentialNum=6./numPlayers);
#---------------set up constrained game -----------------#
cState = 6;   cThresh = 10;                             
sGame.setConstrainedState(cState, cThresh, isLB = True);
optCRes = sGame.solveWithConstraint(p0,verbose = False);
cleanToll = abs(ut.truncate(sGame("optDual")));
cleanToll = np.concatenate((np.array([0,0,0]), cleanToll));
barC = ut.toll2Mat(cState, cleanToll, [sGame.States, sGame.Actions, sGame.Time], True);



#---------------set up admm algorithm-------------------#

testN = 5;
dualPlot = plt.figure(1);
densityPlot = plt.figure(2);
certificatePlot = plt.figure(3);
rhoVal= 1.0;
rhoVal = np.linspace(0.5, 2.5, testN);
Iterations= np.linspace(1,100, 100);
for penaltyIter in range(testN):
    rho = rhoVal[penaltyIter];
    print "solving penalty ", rho;
    def gameObj(yk, length = 1):
        obj = None;
        if length  == 1:
            obj = np.sum(0.5*np.multiply(np.multiply(sGame.R, sGame.R),yk) + np.multiply(sGame.C,yk)); # + rho*(Ay - b)
        else:
            obj = np.zeros(length);
            for i in range(length):
                yki = yk[:,:,:,i];
                obj[i] =  np.sum(0.5*np.multiply(np.multiply(sGame.R, sGame.R),yki) + np.multiply(sGame.C,yki))# + rho*(np.sum(yki[cState,:];        
        return obj;

    lambda0 = np.zeros((sGame.States, sGame.Actions,sGame.Time));
    y0 = np.zeros((sGame.States, sGame.Actions, sGame.Time)); 
      
    for i in range(3,Time):
        lambda0[6,:, i] += 600.;
    
    yHist, lambHist, certificate,finalLamb = da.admm(lambda0, rho, y0, p0, sGame.P, 6,10.0, sGame.R, sGame.C, maxErr = 1.0, optVar= barC);
    plt.figure(2);
    plt.plot(Iterations, 1.0*yHist, label = '%.1f'%rho );
    plt.figure(1);
    plt.plot(Iterations,1.0*lambHist/la.norm(barC), label = '%.2f'%rho);
    plt.figure(3);
    plt.plot(Iterations, 1.0*certificate/gameObj(optCRes), label = '%.2f'%rho);

densityPlot.legend();
densityPlot.show(); plt.grid(densityPlot);

dualPlot.legend();
dualPlot.show();plt.grid(dualPlot);

certificatePlot.legend();
certificatePlot.show();plt.grid(certificatePlot);

#------------------ apply regular penalty -----------------------#
# FW of values converging
threshVal = 1e-3;
def gradF(x):
  return -np.multiply(sGame("reward"), x) + sGame("C") + finalLamb;
x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
ytThresh, ytHist = fw.FW(x0, p0, sGame("probability"), gradF, True, threshVal, maxIterations = 500);
ytHistArr = np.zeros(len(ytHist));
for i in range(len(ytHist)):
    ytHistArr[i] = la.norm((ytHist[i] - optCRes));
    
fig = plt.figure();
blue = '#1f77b4ff';
orange = '#ff7f0eff';
#plt.plot(np.linspace(1, len(ytHist),len(ytHist)), ytHistArr/la.norm(optCRes), linewidth = 2, label = r'regular penalty',color = blue);

#---------------- exact penalty -------------------------#
for j in range(5):
    curDelta = 0.1**j + 1;
    def exactGrad(x):
        grad = -np.multiply(sGame("reward"), x)+ sGame("C");
        for time in range(Time):
            xDensity = np.sum(x[cState,:,time]);
            if xDensity<= cThresh: # put actual constraint here
                grad[cState,:,time] += curDelta*finalLamb[cState,:,time]
        return grad;
    # This is the not state constrained case
    x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
    ytCThresh, ytCHist = fw.FW(x0, p0, sGame("probability"), exactGrad, True, threshVal, maxIterations = 500);
    ytCHistArr = np.zeros(len(ytCHist));
    for i in range(len(ytCHist)):
        ytCHistArr[i] = la.norm((ytCHist[i]  - optCRes));
    plt.plot(np.linspace(1, len(ytHist),len(ytCHist)), ytCHistArr/la.norm(optCRes), linewidth = 2, label = str(curDelta - 1));

plt.legend();
#plt.title("Difference in Norm as a function of termination tolerance")
plt.xlabel(r"Iterations");
plt.ylabel(r"$\frac{||y^{\epsilon} - y^{\star}||}{||y^{\star}||}$");
#plt.xscale('log')
plt.yscale("log");
plt.grid();
plt.show();
#----------