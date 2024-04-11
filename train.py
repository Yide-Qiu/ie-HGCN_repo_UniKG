import argparse
from tqdm import tqdm
import pdb
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
filterwarnings("ignore")
import torch
import dill
import torchmetrics
import pickle

from ogb.nodeproppred import Evaluator

parser = argparse.ArgumentParser(description='Training GNN on UniKG benchmark')

parser.add_argument('--data_dir', type=str, default='./dataset/UniKG_1M.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./hgt_4layer',
                    help='The address for storing the trained models.')
parser.add_argument('--plot', action='store_true',
                    help='Whether to plot the loss/acc curve')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn', 'ieHGCN'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=512,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=4,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=520,
                    help='How many nodes to be sampled per layer per type')

parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epoch to run')
parser.add_argument('--loss_efficient', type=int, default=1000,
                    help='Loss efficient.')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Number of output nodes for training')  
parser.add_argument('--clip', type=float, default=1.0,
                    help='Gradient Norm Clipping') 
parser.add_argument('--log_steps', type=int, default=1,
                    help='Log once every epoch.') 
                    

parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--use_RTE',   help='Whether to use RTE',     action='store_true')

args = parser.parse_args()
args_print(args)

def label_platten(labels, num_classes):
    # input [b,16]
    y = torch.zeros(size=(labels.shape[0], num_classes), dtype=torch.float32)
    valid_mask = (labels != -1)
    has_zero = torch.any(labels==0, dim=1)
    labels = torch.where(valid_mask, labels, torch.tensor(0))
    y.scatter_(1, labels, 1)
    y[:,0][~has_zero] = 0
    # ouput [b,c]
    return y

def ogbn_sample(seed, samp_nodes):
    np.random.seed(seed)
    feature, times, edge_list, indxs, _ = sample_subgraph(graph, \
                inp = {str_entity: np.concatenate([samp_nodes, np.array(graph.years)[samp_nodes]]).reshape(2, -1).transpose()}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, \
                    feature_extractor = feature_MAG)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    train_mask = graph.train_mask[indxs[str_entity]]
    test_mask  = graph.test_mask[indxs[str_entity]]
    y_multilabel = torch.LongTensor(graph.y[indxs[str_entity]]) # [i, 16]
    y_multilabel = label_platten(y_multilabel, num_mutilabels) # [i,2000]
    return node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, test_mask), y_multilabel
    
def prepare_data(pool, task_type = 'train', s_idx = 0, n_batch = args.n_batch, batch_size = args.batch_size):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    # pdb.set_trace()
    if task_type == 'train':
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), np.random.choice(target_nodes, args.batch_size, replace = False)]))
            jobs.append(p)
    elif task_type == 'sequential':
        for i in np.arange(n_batch):
            target_papers = graph.test_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    elif task_type == 'variance_reduce':
        target_papers = graph.test_paper[s_idx * args.batch_size : (s_idx + 1) * args.batch_size]
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    return jobs

def evaluate(logits, y):
    logits = torch.sigmoid(logits).cpu()
    # pdb.set_trace()
    min_vals, _ = logits.min(dim=1, keepdim=True)
    max_vals, _ = logits.max(dim=1, keepdim=True)
    logits = (logits - min_vals) / (max_vals - min_vals)

    predictions = (logits > 0.5).int()
    targets = y.cpu()
 
    subset_acc = (predictions == targets).all(dim=1).float().mean()

    precision = torchmetrics.Precision(task='multilabel', num_labels=y.shape[1])
    precision_value = precision(predictions, targets)

    recall = torchmetrics.Recall(task='multilabel', num_labels=y.shape[1])
    recall_value = recall(predictions, targets)

    f1_value = (2*precision_value*recall_value) / (precision_value+recall_value)

    return subset_acc, precision_value, recall_value, f1_value

def average(a,b,c,d,n):
    return a/n, b/n, c/n, d/n

str_entity = 'entity'
graph = dill.load(open(args.data_dir, 'rb'))
evaluator = Evaluator(name='ogbn-mag')
device = torch.device("cuda:%d" % args.cuda)
target_nodes = np.arange(len(graph.node_feature[str_entity]))
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[str_entity][0]), \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1,\
          prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = args.use_RTE)
num_mutilabels = 2000
classifier = Classifier(args.n_hid, num_mutilabels)

model = nn.Sequential(gnn, classifier).to(device)

print('GNN #Params: %d' % get_n_params(gnn))
print('Classifier #Params: %d' % get_n_params(classifier))
print('Model #Params: %d' % get_n_params(model))
# criterion = nn.NLLLoss()
criterion = nn.BCEWithLogitsLoss()
# pdb.set_trace()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
    ]


optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-02)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.1, anneal_strategy='linear', final_div_factor=10,\
                        max_lr = 5e-4, total_steps = args.n_batch * args.n_epoch + 1)

stats = []
res   = []
best_val   = 0
train_step = 0

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)

# pdb.set_trace()
if args.data_dir in ['./dataset/UniKG_1M.pk']:
    dataset = 'UniKG-1M'
if args.data_dir in ['./dataset/UniKG_10M.pk']:
    dataset = 'UniKG-10M'
if args.data_dir in ['./dataset/UniKG_full.pk']:
    dataset = 'UniKG-full'
log = open(f'{args.conv_name}-{dataset}.txt','w')

for epoch in np.arange(args.n_epoch) + 1:
    datas = [job.get() for job in jobs]
    pool.close()
    pool.join()
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))

    # import pdb
    # pdb.set_trace()

    epoch_start = time.time()
    model.train()
    stat = []
    for node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, test_mask), y_multilabel in datas:
        time1 = time.time()
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        # pdb.set_trace()
        # y_multilabel = torch.LongTensor(y_multilabel).to(device) [i,2000]
        train_res  = classifier.forward(node_rep[:int(y_multilabel.shape[0])][train_mask])
        # valid_res  = classifier.forward(node_rep[:len(ylabel)][valid_mask])
        test_res   = classifier.forward(node_rep[:int(y_multilabel.shape[0])][test_mask])

        train_loss = criterion(train_res, y_multilabel[train_mask].to(device)) * args.loss_efficient

        ###
        # check
        print(torch.sum(y_multilabel), y_multilabel.shape)
        print(train_res, y_multilabel[train_mask])


        ###


        optimizer.zero_grad() 
        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_step += 1
        scheduler.step(train_step)

        time2 = time.time() - time1
        if epoch % args.log_steps == 0:
            acc, precision, recall, f1 = evaluate(test_res, y_multilabel[test_mask])
            txt = f'Epoch: {epoch:02d}, Time:{time2:.2f}, Loss: {train_loss:.4f}, Acc: {100 * acc:.2f}%, precision: {100 * precision:.2f}%, recall: {100 * recall:.2f}%, f1: {100 * f1:.2f}\n'
            print(txt)
        log.write(txt)
        log.flush()

    epoch_time = time.time() - epoch_start
    epoch_time_txt = f'epoch {epoch} cost time: {epoch_time:.2f}s \n'
    log.write(epoch_time_txt)
    log.flush()
