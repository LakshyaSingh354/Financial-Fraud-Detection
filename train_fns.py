import copy
import time
from matplotlib import pyplot as plt
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve


def get_f1_score(y_true, y_pred):
    """
    Only works for binary case.
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]

    :param y_true: A list of labels in 0 or 1: 1 * N
    :param y_pred: A list of labels in 0 or 1: 1 * N
    :return:
    """
    # print(y_true, y_pred)

    cf_m = confusion_matrix(y_true.cpu(), y_pred)
    # print(cf_m)

    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)

    return precision, recall, f1

def evaluate(model, g, features, labels, device):
    "Compute the F1 value in a binary classification case"

    preds = model(g, features.to(device))
    preds = th.argmax(preds, axis=1).cpu().numpy()
    precision, recall, f1 = get_f1_score(labels, preds)

    return precision


def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    unnormalized_preds = model(g, features.to(device))
    pred_proba = th.softmax(unnormalized_preds, dim=-1)
    if not threshold:
        return unnormalized_preds.argmax(axis=1).detach().cpu().numpy(), pred_proba[:,1].detach().cpu().numpy()
    return np.where(pred_proba.detach().cpu().numpy() > threshold, 1, 0), pred_proba[:,1].detach().cpu().numpy()

def save_model(g, model, model_dir, id_to_node, mean, stdev):

    # Save Pytorch model's parameters to model.pth
    th.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    # Save graph's structure information to metadata.pkl for inference codes to initialize RGCN model.
    etype_list = g.canonical_etypes
    ntype_cnt = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({'etypes': etype_list,
                     'ntype_cnt': ntype_cnt,
                     'feat_mean': mean,
                     'feat_std': stdev}, f)

    # Save original IDs to Node_ids, and trained embedding for non-target node type
    # Covert id_to_node into pandas dataframes
    # for ntype, mapping in id_to_node.items():

    #     # ignore target node
    #     if ntype == 'target':
    #         continue

    #     # retrieve old and node id list
    #     old_id_list, node_id_list = [], []
    #     for old_id, node_id in mapping.items():
    #         old_id_list.append(old_id)
    #         node_id_list.append(node_id)

    #     # retrieve embeddings of a node type
    #     node_feats = model.embed[ntype].detach().cpu().numpy()

    #     # get the number of nodes and the dimension of features
    #     num_nodes = node_feats.shape[0]
    #     num_feats = node_feats.shape[1]

    #     # create id dataframe
    #     node_ids_df = pd.DataFrame({'~label': [ntype] * num_nodes})
    #     node_ids_df['~id_tmp'] = old_id_list
    #     node_ids_df['~id'] = node_ids_df['~label'] + '-' + node_ids_df['~id_tmp']
    #     node_ids_df['node_id'] = node_id_list

    #     # create feature dataframe columns
    #     cols = {'val' + str(i + 1) + ':Double': node_feats[:, i] for i in range(num_feats)}
    #     node_feats_df = pd.DataFrame(cols)

    #     # merge id with feature, where feature_df use index
    #     node_id_feats_df = node_ids_df.merge(node_feats_df, left_on='node_id', right_on=node_feats_df.index)
    #     # drop the id_tmp and node_id columns to follow the Grelim format requirements
    #     node_id_feats_df = node_id_feats_df.drop(['~id_tmp', 'node_id'], axis=1)

    #     # dump the embeddings to files
    #     node_id_feats_df.to_csv(os.path.join(model_dir, ntype + '.csv'),
    #                             index=False, header=True, encoding='utf-8')


def get_model(ntype_dict, etypes, hyperparams, in_feats, n_classes, device):

    model = HeteroRGCN(ntype_dict, etypes, in_feats, hyperparams['n_hidden'], n_classes, hyperparams['n_layers'], in_feats)
    model = model.to(device)

    return model

def initial_record():
    if os.path.exists('./output/results.txt'):
        os.remove('./output/results.txt')
    with open('./output/results.txt','w') as f:    
        f.write("Epoch,Time(s),Loss,Precision\n")   


def normalize(feature_matrix):
    mean = th.mean(feature_matrix, axis=0)
    stdev = th.sqrt(th.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0])
    return mean, stdev, (feature_matrix - mean) / stdev

def save_roc_curve(fpr, tpr, roc_auc, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC curve')
    plt.legend(loc="lower right")
    f.savefig(location)


def save_pr_curve(fpr, tpr, pr_auc, ap, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Model PR curve: AP={0:0.2f}'.format(ap))
    plt.legend(loc="lower right")
    f.savefig(location)

def get_metrics(pred, pred_proba, labels, mask, out_dir):
    labels, mask = labels, mask
    labels, pred, pred_proba = labels[np.where(mask)], pred[np.where(mask)], pred_proba[np.where(mask)]

    acc = ((pred == labels)).sum() / mask.sum()

    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()

    precision = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f1 = 2*(precision*recall)/(precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = pd.DataFrame(np.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])

    ap = average_precision_score(labels, pred_proba)

    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)

    save_roc_curve(fpr, tpr, roc_auc, os.path.join(out_dir, "roc_curve.png"))
    save_pr_curve(prc, rec, pr_auc, ap, os.path.join(out_dir, "pr_curve.png"))

    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix

def train_fg(model, optim, loss, features, labels, train_g, test_g, test_mask,
             device, n_epochs, thresh, compute_metrics=True):
    """
    A full graph verison of RGCN training
    """

    duration = []
    best_loss = 1
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        tic = time.time()
        loss_val = 0.
        model = model.to(device)
        pred = model(train_g, features.to(device))

        l = loss(pred, labels)

        optim.zero_grad()
        l.backward()
        optim.step()

        loss_val += l

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, device)
        # print("Epoch {:05d}, Time(s) {:.4f}, Loss {:.4f}, Precision {:.4f} ".format(
        #         epoch+1, np.mean(duration), loss_val, metric), end='\r')
        pbar.set_postfix({
            "Loss": f"{loss_val:.4f}",
            "Precision": f"{metric:.4f}"
        })

        
        epoch_result = "{:05d},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, np.mean(duration), loss_val, metric)
        with open('./output/results.txt','a+') as f:    
            f.write(epoch_result)  

        if loss_val < best_loss:
            best_loss = loss_val
            best_model = copy.deepcopy(model)


    class_preds, pred_proba = get_model_class_predictions(best_model,
                                                          test_g,
                                                          features,
                                                          labels,
                                                          device,
                                                          threshold=thresh)

    if compute_metrics:
        acc, f1, p, r, roc, pr, ap, cm = get_metrics(class_preds, pred_proba, labels.cpu().numpy(), test_mask.cpu().numpy(), './output/')
        print("Metrics")
        print("""Confusion Matrix:
                                {}
                                f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}, pr: {:.4f}, ap: {:.4f}
                             """.format(cm, f1, p, r, acc, roc, pr, ap))

    return best_model, class_preds, pred_proba