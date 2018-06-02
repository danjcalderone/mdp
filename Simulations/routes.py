# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 11:59:12 2018

@author: craba
"""
from cvxpy import *
import numpy as np
import networkx as nx
import scipy.linalg as sla
import mosek
import matplotlib.pyplot as plt
import matplotlib
#%precision %.2f
np.set_printoptions(precision=2)
# Generate routes matrix
def RouteGen(graph, edgeSet, sourceN, sinkN):
    routes =  list(nx.all_simple_paths(graph, source=sourceN, target=sinkN));
    RouteMat = np.zeros((graph.number_of_edges(), len(routes)));
    for route in range(0,len(routes)):
        curRoute = routes[route];#Route is in nodes
        #Look for the index of corresponding edge
        for edge in range(0,len(curRoute)-1):
            #Unpack edge
            start= curRoute[edge]; end = curRoute[edge+1];
            edgeInd = 0;
            #find the edge index in list
            try:
                edgeInd = edgeSet.index((end,start));    
            except ValueError:
                edgeInd = edgeSet.index((start,end)); 
                
#             print(edgeInd)
            RouteMat[edgeInd,route]= 1;
    return RouteMat;