import sys
import random
import numpy as np
import networkx as nx
from itertools import combinations


class HyGraph(object):
    def __init__(self,train_pos=None, train_neg=None,test_pos=None, test_neg=None):
        '''
        Data and (train_pos,neg) should not be None at the same time 
        '''

        # if data==None and train_pos==None:
        #     print("No legal input")
        #     sys.exit()

        # self.data=[tuple(sorted(x)) for x in data] if data else None
        train_pos=[tuple(sorted(x)) for x in train_pos] if train_pos else None
        test_pos=[tuple(sorted(x)) for x in test_pos] if test_pos else None
        train_neg=[tuple(sorted(x)) for x in train_neg] if train_neg else None
        test_neg=[tuple(sorted(x)) for x in test_neg] if test_neg else None


        self.data=train_pos
        self.train=train_pos+train_neg
        self.test=test_pos+test_neg
        self.train_label=[1]*len(train_pos)+[0]*len(train_neg)
        self.test_label=[1]*len(test_pos)+[0]*len(test_neg)

        self.nodes=list(set([y for x in self.data+self.test+self.train for y in x]))
        
        node_edge={}
        for node in self.nodes:
            node_edge[node]=[]

        for i,x in enumerate(self.data):
            for node in x:
                node_edge[node].append(i)
        self.node_edge=node_edge






def decomp_graph(nodes_use,graph,max_nodes_per_hop=None,max_edges_per_hop=None,r=1,MAX=8):

    edges=graph.data
    node_edge=graph.node_edge

    visited=set([x for x in nodes_use])
    fringe=set([x for x in nodes_use])
    new_nodes=set(nodes_use)

    MAX=max(MAX,r+2)

    edges_index=[]
    for i in range(r):
        if len(fringe)>0:

            edges_use=[]
            for x in fringe:
                edges_use+=node_edge[x]

            edges_use=list(set(edges_use))

            if max_edges_per_hop is not None:
                if max_edges_per_hop<len(edges_use):
                    edges_use=random.sample(edges_use,max_edges_per_hop)

            temp_node=[y for i in edges_use for y in edges[i]]
            edges_index+=edges_use
            fringe=fringe.union(set(temp_node))
            fringe=fringe-visited
            visited=visited.union(fringe)

            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe):
                    fringe = random.sample(fringe, max_nodes_per_hop)

            new_nodes=new_nodes.union(fringe)




    new_nodes=sorted(list(new_nodes))
    PATH=np.ones((len(new_nodes),len(nodes_use)))*MAX

    edges_use=[]
    for x in new_nodes:
        edges_use+=node_edge[x]

    edges_use=list(set(edges_use))

    to_pop=[]
    for i in [x for x in edges_use if len(graph.data[x])==len(nodes_use)]:
        temp=[x in graph.data[i] for x in nodes_use]
        if sum(temp)==len(graph.data[i]):
            to_pop.append(i)

    to_pop=sorted(to_pop,reverse=True)
    for i in to_pop:
        del edges_use[edges_use.index(i)]


    edges_combine=[y for i in edges_use for y in combinations(graph.data[i],2)]
    subgraph=nx.Graph(edges_combine)


    for i,x in enumerate(new_nodes):
        for j,y in enumerate(nodes_use):
            if x in subgraph and y in subgraph and x!=y:
                if nx.has_path(subgraph,x,y):
                    PATH[i,j]=min(nx.shortest_path_length(subgraph,x,y)+1,MAX)


    hyper_all=[]
    # edge_nodes=[x for i in edges_index for x in edges[i]]
    # missed_nodes=list(set(nodes_use)-set(edge_nodes))
    # hyper_temp=[0]*len(new_nodes)

    # for x in missed_nodes:
    #     hyper_use=hyper_temp.copy()
    #     hyper_use[new_nodes.index(x)]=1
    #     hyper_all.append(hyper_use)

    for i in edges_index:
        hyper_use=[0]*len(new_nodes)
        for x in edges[i]:
            if x in new_nodes:
                hyper_use[new_nodes.index(x)]=1
        hyper_all.append(hyper_use)

    hyperedges=np.array(hyper_all)

    return new_nodes,PATH,hyperedges

