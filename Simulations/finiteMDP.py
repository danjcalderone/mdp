# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:49:38 2018

@author: craba
"""
import routes as route
import mdp as mdp

import cvxpy as cvx
import numpy as np
import networkx as nx
import scipy.linalg as sla
import matplotlib.pyplot as plt



# networks Set up 
# Create Graph Object
rowSize = 3; colSize = 5;

G = nx.grid_graph([rowSize,colSize])
e = G.number_of_edges();
v = G.number_of_nodes();
G = nx.convert_node_labels_to_integers(G)
pos=nx.spring_layout(G);

#plt.figure();
#nx.draw(G, pos=pos, node_color='g', edge_color='k', with_labels=True, font_weight='bold')
#plt.show();

P,c = mdp.generateGridMDP(15,5,G)
time = 30;
states = rowSize*colSize;

optRes = mdp.solveMDP(time, P, c);
#final = np.sum(optRes[:,:,time-1],axis=1);

#plt.figure()
#timeLine = np.linspace(0,time,time);
#for i in range(states):
#    plt.plot(timeLine,np.sum(optRes[i,:,:], axis=0),label=(r'node %i'%(i+1)));
#plt.xlabel('Time [s]')
#plt.ylabel('Node States')
#plt.title('MDP Node Evolution Over time')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show();
#
#
#
#mag = 3000;
#nodesize=[final[f]*mag for f in G]
#cap = mag* np.ones(len(nodesize));

#plt.figure();
#nx.draw(G, pos=pos, node_color='w',with_labels=True, font_weight='bold')
#nx.draw_networkx_nodes(G,pos,node_size=3.2/3*cap,node_color='r',alpha=1)
#nx.draw_networkx_nodes(G,pos,node_size=cap,node_color='w',alpha=1)
#
#nx.draw_networkx_nodes(G,pos,node_size=nodesize,node_color='c',alpha=1)
#
#plt.show();

#frames = [];
frameNumber = time;

fig = plt.figure()
#ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
#line, = ax.plot([], [], lw=2)
i = 0;
mag = 3000;
cap = mag* np.ones(v);    
nx.draw(G, pos=pos, node_color='w',with_labels=True, font_weight='bold');
nx.draw_networkx_nodes(G,pos,node_size=3.2/3*cap,node_color='r',alpha=1);
dontStop = True;

try:
    print('running')

except KeyboardInterrupt:
    print('paused')
    inp =input('continue? (y/n)')

for i in range(frameNumber):
    try:  
        frame = np.sum(optRes[:,:,i],axis=1);
        nodesize=[frame[f]*mag for f in G]
        nx.draw_networkx_nodes(G,pos,node_size=cap,node_color='w',alpha=1)
        nx.draw_networkx_nodes(G,pos,node_size=nodesize,node_color='c',alpha=1)
#        anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=200, interval=20, blit=True)

    except KeyboardInterrupt:
        dontStop = False;
    plt.show();
    plt.pause(0.5);

   
   


