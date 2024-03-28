import os,sys
import pickle
import argparse
import math
import numpy as np

sys.path.append("/N/slate/wanc/Hypergraph/KDD_hypergraph/Datasets3/code_random/")

from utils_subgraph import *
from utils_msg import *


parser = argparse.ArgumentParser(description='Hyper decomp for link prediction')

parser.add_argument('--file_path', default=None, help='pathway to file')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--use-attribute', action='store_true', default=False,help='whether to use node attributes')
parser.add_argument('--save-model', action='store_true', default=False,help='save the final model')
parser.add_argument("--embedding_size",default=64,help='GNN model')

parser.add_argument('--use-embedding', action='store_true', default=False,help='whether to use node2vec node embeddings')
parser.add_argument('--add_svd', action='store_true', default=True,help='use the info of two svd norm')

### DGCNN set up
parser.add_argument("--gm",default='DGCNN',help='GNN model')
parser.add_argument("--sortpooling_k",default=0.6,help='the size of sort pooling')
parser.add_argument("--latent_dim",default=[32,32,32,1],help="latent dimension for subgraph embeddings")
parser.add_argument("--hidden",default=128,help="XXX")
parser.add_argument("--out_dim",default=0,help="out dimension for subgraph embedding, default 0 for automated generation")
parser.add_argument("--dropout",action='store_true',default=True,help="use dropout for training")
parser.add_argument("--num_class",default=2, help="2 for binary classification")
parser.add_argument("--num_epochs",default=30,help="number of epoches for training")
parser.add_argument("--learning_rate",default=1e-4,help="learning rate in training")
parser.add_argument("--printAUC",action="store_true",default=True,help="print AUC in training")
parser.add_argument('-conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
parser.add_argument('-deepsets_outdim', default=20, help='the dimension of deep sets output')
parser.add_argument('--batch-size', type=int, default=50)

parser.add_argument('--deepset_combine', default='sum', help='the method to combine the path information max, mean or sum')
parser.add_argument('--msg_combine', default='mean', help='the method to combine in msg passing the path information mean or sum')
parser.add_argument('--msg_combine1', default='sum', help='the method to combine in msg passing the path information mean or sum')


args = parser.parse_args()
args.num_epochs=int(args.num_epochs)

# args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# print(args)

# args.file_path="NDC-classes-unique-hyperedges_0_.pkl"


# if os.path.exists(args.file_path.replace("_.pkl","")+"_"+args.data_name+'_auc_results.txt'):
#     sys.exit()


with open(args.file_path,"rb") as f:
    graph=pickle.load(f)



train_graph_path=args.file_path.replace(".pkl","")+"msgtraingraph.pkl"
if not os.path.exists(train_graph_path):
    train_graphs=[decomp_graph(graph.train[i],graph,args.max_nodes_per_hop,args.max_edges_per_hop,1) for i in tqdm(range(len(graph.train)))]
    with open(train_graph_path,"wb") as f:
        pickle.dump(train_graphs,f)
else:
    with open(train_graph_path,"rb") as f:
        train_graphs=pickle.load(f)

test_graph_path=args.file_path.replace(".pkl","")+"msgtestgraph.pkl"
if not os.path.exists(test_graph_path):
    test_graphs=[decomp_graph(graph.test[i],graph,args.max_nodes_per_hop,args.max_edges_per_hop,1) for i in tqdm(range(len(graph.test)))]
    with open(test_graph_path,"wb") as f:
        pickle.dump(test_graphs,f)
else:
    with open(test_graph_path,"rb") as f:
        test_graphs=pickle.load(f)




train_graphs=[GNNGraph(graph.train_label[i],*train_graphs[i]) for i in tqdm(range(len(graph.train)))]
test_graphs=[GNNGraph(graph.test_label[i],*test_graphs[i]) for i in tqdm(range(len(graph.test)))]


args.node_feat_size=args.deepsets_outdim
if args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs])
    k_ = int(math.ceil(args.sortpooling_k * len(num_nodes_list))) - 1
    args.sortpooling_k = max(10, num_nodes_list[k_])
    print('k used in SortPooling is: ' + str(args.sortpooling_k))

# if args.use_embedding:
#     args.node_feat_size=args.embedding_size+args.deepsets_outdim
# else:
#     args.node_feat_size=args.deepsets_outdim


# args.deepset_combine="sum"
# random.shuffle(train_graphs)
FILE_out=args.file_path.replace(".pkl","")

if args.add_svd:
    print("added SVD")
    FILE_out+="_svd_"
    for i in tqdm(range(len(train_graphs))):
        train_graphs[i].calculate_svd_diff()
    for i in tqdm(range(len(test_graphs))):
        test_graphs[i].calculate_svd_diff()


FILE_out+='msg_f1_results.pkl'
if os.path.exists(FILE_out):
    sys.exit()


random.shuffle(train_graphs)

# args.msg_combine="sum"
# args.msg_combine1="sum"
classifier = Classifier(args)
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None
classifier_keep=None
result_train=[]
result_test=[]
for epoch in range(args.num_epochs):
    random.shuffle(train_graphs)
    _=classifier.train()
    avg_loss = loop_dataset(args, train_graphs, classifier, train_idxes, optimizer=optimizer)
    result_train.append(avg_loss)
    if not args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f f1 %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2],avg_loss[3]))
    _=classifier.eval()
    test_loss = loop_dataset(args,test_graphs, classifier, list(range(len(test_graphs))))
    result_test.append(test_loss)
    if not args.printAUC:
        test_loss[2] = 0.0
    print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f f1 %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2],test_loss[3]))
    if best_loss is None:
        best_loss=test_loss
        classifier_keep=classifier
    else:
        if best_loss[3]<test_loss[3]:
            classifier_keep=classifier
            best_loss=test_loss
            epoch_keep=epoch




#FILE_out=+'msg_f1_results.pkl'
with open(FILE_out,'wb') as f:
    pickle.dump((best_loss,result_train,result_test,classifier_keep),f)



