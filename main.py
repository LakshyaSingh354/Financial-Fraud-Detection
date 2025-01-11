import glob
import os

from train_fns import get_model, initial_record, normalize, save_model, train_fg
from utils import construct_graph, get_edgelists, get_labels

import torch


args = {
    "compute_metrics": True,
    "dropout": 0.2,
    "edges": 'relation*',
    "embedding_size": 360,
    "labels": 'tags.csv',
    "lr": 0.01,
    "model_dir": './model',
    "n_epochs": 1000,
    "n_hidden": 16,
    "n_layers": 3,
    "new_accounts": 'test.csv',
    "nodes": 'features.csv',
    "num_gpus": 0,
    "optimizer": 'adam',
    "output_dir": './output',
    "target_ntype": 'TransactionID',
    "threshold": 0,
    "training_dir": '/kaggle/input/rgcn-ieee',
    "weight_decay": 0.0005
}

file_list = glob.glob('data/*edgelist.csv')

edges = ",".join(map(lambda x: x.split("/")[-1], [file for file in file_list if "relation" in file]))

args["edges"] = edges

args["edges"] = get_edgelists('relation*', args["training_dir"])

g, features, target_id_to_node, id_to_node = construct_graph(args["training_dir"],
                                args["edges"],
                                args["nodes"],
                                args["target_ntype"])

mean, stdev, features = normalize(torch.from_numpy(features))

print('feature mean shape:{}, std shape:{}'.format(mean.shape, stdev.shape))

g.nodes['target'].data['features'] = features

print("Getting labels")
n_nodes = g.number_of_nodes('target')

labels, _, test_mask = get_labels(target_id_to_node,
                        n_nodes,
                        args["target_ntype"],
                        os.path.join(args["training_dir"], args["labels"]),
                        os.path.join(args["training_dir"], args["new_accounts"]))
print("Got labels")

labels = torch.from_numpy(labels).float()
test_mask = torch.from_numpy(test_mask).float()

n_nodes = torch.sum(torch.tensor([g.number_of_nodes(n_type) for n_type in g.ntypes]))
n_edges = torch.sum(torch.tensor([g.number_of_edges(e_type) for e_type in g.etypes]))

print("""----Data statistics------'
        #Nodes: {}
        #Edges: {}
        #Features Shape: {}
        #Labeled Test samples: {}""".format(n_nodes,
                            n_edges,
                            features.shape,
                            test_mask.sum()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Initializing Model")
in_feats = features.shape[1]
n_classes = 2

ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}

model = get_model(ntype_dict, g.etypes, args, in_feats, n_classes, device)
print("Initialized Model")

features = features.to(device)

labels = labels.long().to(device)
test_mask = test_mask.to(device)
g = g.to(device)

loss = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

print("Starting Model training")

initial_record()

model, class_preds, pred_proba = train_fg(model, optim, loss, features, labels, g, g,
                        test_mask, device, args["n_epochs"],
                        args["threshold"],  args["compute_metrics"])
print("Finished Model training")

print("Saving model") 

if not os.path.exists(args["model_dir"]):
    os.makedirs(args["model_dir"])

save_model(g, model, args["model_dir"], id_to_node, mean, stdev)
print("Model and metadata saved")