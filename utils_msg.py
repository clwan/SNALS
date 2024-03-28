### process to multiple 

import os, sys
import random
import numpy as np
import multiprocessing as mp
import networkx as nx


from tqdm import tqdm
from math import comb,sqrt,log
from itertools import combinations
from functools import partial
from collections import Counter,defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from numpy import linalg as LA
from sklearn import metrics
from typing import Union
from scipy.sparse import coo_matrix, block_diag

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch import FloatTensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

NetIO = Union[FloatTensor, Variable]

sys.path.append("/N/slate/wanc/Hypergraph/set_transformer")

from modules import SAB, PMA


def SVD_emb(graph,k,weighted=None):
    edges=graph.data
    nodes=graph.nodes
    nodes.sort()
    nodes={v:i for i,v in enumerate(nodes)}
    N=len(nodes)

    data=[]
    col=[]
    row=[]
    for i,v in enumerate(edges):
        if weighted is None:
            value=1
        else:
            value=log(N/len(v))
        for v1 in v:
            data.append(value)
            col.append(i)
            row.append(nodes[v1])

    MAT_B=csr_matrix((data, (row, col)),shape=(N,len(edges)),dtype="float32")
    row_start_stop = np.lib.stride_tricks.as_strided(MAT_B.indptr, shape=(MAT_B.shape[0], 2),strides=2*MAT_B.indptr.strides)
    for start, stop in row_start_stop:   
        row = MAT_B.data[start:stop]
        row /= LA.norm(row)

    U,S,V=svds(MAT_B,k,which="LM")
    U=U*np.sqrt(S)
    node_emb={k:U[v,:] for k,v in nodes.items()}

    return node_emb



##### need some process here


class GNNGraph(object):
    def __init__(self,label,nodes,PATH,hyperedges):
        '''
            g: a networkx graph
            node_tags: a dict of node tags
            node_features: a dict of node features
        '''


        if len(hyperedges.shape)==1:
            self.hyperedges=None
            self.nodedegs=None
            self.edgedegs=None
        else:
            # self.hyperedges=hyperedges
            # self.nodedegs=np.sum(hyperedges,0)
            # self.edgedegs=np.sum(hyperedges,1)
            
            nodedegs=np.sum(hyperedges,0)
            if 0 in nodedegs:
                nodes=[x for i,x in enumerate(nodes) if nodedegs[i]>0]
                PATH=PATH[nodedegs>0,:]
                hyperedges=hyperedges[:,nodedegs>0]
                nodedegs=np.sum(hyperedges,0)

            self.hyperedges=hyperedges
            self.nodedegs=np.sum(hyperedges,0)
            self.edgedegs=np.sum(hyperedges,1)

        self.nodes=nodes
        self.label=label
        self.PATH=PATH
        self.num_nodes=len(nodes)

    def calculate_svd_diff(self):
        if min(self.PATH.shape)>1:
            _,s,_=LA.svd(self.PATH)
            if s[1]>0:
                self.svd_diff=log(s[0]/s[1]+1)
            else:
                self.svd_diff=0
        else:
            self.svd_diff=0

    def add_svd_emb(self,node_emb):
        nodes=self.nodes
        emb=[node_emb[y] for y in nodes]
        emb=np.vstack(emb)
        self.node_emb=emb







def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)



def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)

class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)






class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU',combine='sum',combine1='sum'):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.combine=combine
        self.combine1=combine1
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_n2e = nn.ModuleList()
        self.conv_e2n = nn.ModuleList()
        self.conv_n2e.append(nn.Linear(num_node_feats, latent_dim[0]))
        self.conv_e2n.append(nn.Linear(latent_dim[0],latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_n2e.append(nn.Linear(latent_dim[i-1], latent_dim[i]))
            self.conv_e2n.append(nn.Linear(latent_dim[i], latent_dim[i]))



        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]


        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        weights_init(self)



    #def forward(self,graph_list,node_feat1,node_feat2):
    def forward(self,batch_graph,node_feat):

        graph_sizes=[0]*len(batch_graph)
        node_degs=[torch.Tensor(0) + 1]*len(batch_graph)
        edge_degs=[torch.Tensor(0) + 1]*len(batch_graph)

        hyperedge_all=[]
        for i in range(len(batch_graph)):
            if batch_graph[i].nodedegs is not None:
                graph_sizes[i]=batch_graph[i].num_nodes
                node_degs[i]=torch.Tensor(batch_graph[i].nodedegs)
                edge_degs[i]=torch.Tensor(batch_graph[i].edgedegs)
                hyperedge_all.append(coo_matrix(batch_graph[i].hyperedges))


        node_degs=torch.cat(node_degs).unsqueeze(1)
        edge_degs=torch.cat(edge_degs).unsqueeze(1)
        COO=block_diag(hyperedge_all)

        values = COO.data
        indices = np.vstack((COO.row, COO.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = COO.shape

        hyperedge=torch.sparse.FloatTensor(i, v, torch.Size(shape))
        hypernode=torch.transpose(hyperedge,0,1)

        h = self.sortpooling_embedding(node_feat,node_degs,hyperedge,edge_degs,hypernode,graph_sizes)

        return h

    def sortpooling_embedding(self,node_feat,node_degs,hyperedge,edge_degs,hypernode,graph_sizes):

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []

        while lv < len(self.latent_dim):
            n2epool = gnn_spmm(hyperedge, cur_message_layer)
            edge_linear = self.conv_n2e[lv](n2epool)


            if self.combine=="mean":
                edge_linear=edge_linear.div(edge_degs)

            edge_message=torch.tanh(edge_linear)

            e2npool=gnn_spmm(hypernode, edge_message)
            node_linear = self.conv_e2n[lv](e2npool) 
            node_normalize = node_linear.div(node_degs)
            cur_message_layer = torch.tanh(node_normalize)

            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)


        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]

        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)


        accum_count = 0
        for i in range(len(graph_sizes)):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim)

                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]


        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))

        ### here to add svd
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)

        to_dense = conv1d_res.view(len(graph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)






# class InvariantModel(nn.Module):
#     def __init__(self, phi: nn.Module, rho: nn.Module,combine="max"):
#         super().__init__()
#         self.phi = phi
#         self.rho = rho
#         self.combine=combine

#     def forward(self, x: NetIO) -> NetIO:
#         # compute the representation for each data point
#         x = self.phi.forward(x)

#         #x = torch.mean(x, dim=0, keepdim=True)
#         #x=torch.max(x,dim=0,keepdim=True)
#         if self.combine=="max":
#             x,_=torch.max(x,0,keepdim=True)
#         elif self.combine=='mean':
#             x=torch.mean(x,0,keepdim=True)
#         elif self.combine=='sum':
#             x=torch.sum(x,0,keepdim=True)

#         out = self.rho.forward(x)

#         return out


# class LinearPhi(nn.Module):
#     def __init__(self, input_size: int, output_size: int=20):
#         super().__init__()
#         self.input_size=input_size
#         self.output_size=output_size

#         self.fc1 = nn.Linear(self.input_size, 50)
#         self.fc1_drop = nn.Dropout()
#         self.fc2 = nn.Linear(50, self.output_size)

#     def forward(self, x: NetIO) -> NetIO:
#         x = F.relu(self.fc1(x))
#         x = self.fc1_drop(x)
#         x = F.relu(self.fc2(x))
#         return x


# class LinearRho(nn.Module):
#     def __init__(self, input_size: int=20, output_size: int = 20):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size

#         self.fc1 = nn.Linear(self.input_size, 30)
#         self.fc1_drop = nn.Dropout()
#         self.fc2 = nn.Linear(30, self.output_size)

#     def forward(self, x: NetIO) -> NetIO:
#         x = F.relu(self.fc1(x))
#         x = self.fc1_drop(x)
#         x = self.fc2(x)
#         return x



class SmallDeepSet(nn.Module):
    def __init__(self, pool="max",outdim=20):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=outdim),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=0,keepdim=True)[0]
        elif self.pool == "mean":
            x = x.mean(dim=0,keepdim=True)
        elif self.pool == "sum":
            x=torch.sum(x,0,keepdim=True)
        x = self.dec(x)
        return x




class SmallSetTransformer(nn.Module):
    def __init__(self,outdim=20):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=outdim),
        )
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x



class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits





class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        # if args.set_method=="deepsets":
        #     # self.phi=LinearPhi(input_size=1)
        #     # self.pho=LinearRho(output_size=args.deepsets_outdim)
        self.pathset=SmallDeepSet(pool=args.deepset_combine,outdim=args.deepsets_outdim)
        # elif args.set_method=="settrans":
        #     self.pathset=SmallSetTransformer(outdim=args.deepsets_outdim)

        self.use_embedding=args.use_embedding


        self.gnn = DGCNN(num_node_feats=args.node_feat_size,latent_dim=args.latent_dim,output_dim=args.out_dim,k=args.sortpooling_k,conv1d_activation=args.conv1d_activation,combine=args.msg_combine,combine1=args.msg_combine1)
        out_dim=self.gnn.dense_dim

        self.add_svd=args.add_svd
        if self.add_svd:
            self.mlp = MLPClassifier(input_size=out_dim+1, hidden_size=args.hidden, num_class=args.num_class, with_dropout=args.dropout)
        else:
            self.mlp = MLPClassifier(input_size=out_dim, hidden_size=args.hidden, num_class=args.num_class, with_dropout=args.dropout)


    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))

        n_nodes=0
        concat_tag=[]
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            if batch_graph[i].nodedegs is not None:
                n_nodes+=batch_graph[i].num_nodes
                PATH=batch_graph[i].PATH
                for i in range(len(PATH)):
                    temp=PATH[i,:][np.newaxis].T
                    temp=torch.from_numpy(temp).type('torch.FloatTensor')
                    concat_tag.append(temp)

        if self.use_embedding:
            node_emb=[]
            for i in range(len(batch_graph)):
                if batch_graph[i].nodedegs is not None:
                    node_emb.append(torch.tensor(batch_graph[i].node_emb))
            return labels,concat_tag,node_emb
        else:
            return labels,concat_tag


    def forward(self, batch_graph):
        feature_info=self.PrepareFeatureLabel(batch_graph)
        if len(feature_info)==3:
            labels,concat_tag,concat_feat=feature_info
        else:
            labels,concat_tag=feature_info

        node_tag=[self.pathset(x) for x in concat_tag]
        node_tag=torch.cat(node_tag,0)

        if len(feature_info)==3:
            concat_feat=torch.cat(concat_feat,0)
            node_feat=torch.cat([node_tag,concat_feat],1)
        else:
            node_feat=node_tag

        embed = self.gnn(batch_graph,node_feat)

        if self.add_svd:
            svd_diff=[x.svd_diff for x in batch_graph]
            svd_diff=np.array([svd_diff]).T
            svd_diff=torch.from_numpy(svd_diff).float()
            embed=torch.cat([svd_diff,embed],1)

        return self.mlp(embed,labels)


def loop_dataset(args, g_list, classifier, sample_idxes, optimizer=None):
    bsize=args.batch_size
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None))//bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc = classifier(batch_graph)
        all_scores.append(logits[:, 1].cpu().detach())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.data.cpu().detach().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
        total_loss.append( np.array([loss, acc]) * len(selected_idx))
        n_samples += len(selected_idx)

    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    if args.printAUC:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        precision, recall, _=metrics.precision_recall_curve(all_targets, all_scores, pos_label=1)
        f1_all=2*precision*recall/(precision+recall)
        f1_all = f1_all[np.logical_not(np.isnan(f1_all))]
        f1=np.max(f1_all)
        avg_loss = np.concatenate((avg_loss, [auc,f1]))
    else:
        avg_loss = np.concatenate((avg_loss, [0.0,0.0]))
    return avg_loss







