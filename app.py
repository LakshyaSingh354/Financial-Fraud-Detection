import pickle
import dgl
import numpy as np
import torch
from flask import Flask, request, jsonify
import onnx
import onnxruntime as ort

from rgcn import HeteroRGCN



def load_model_and_infer_onnx(ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, embedding_size, weights_path, onnx_path, device, input_sample):
    """
    Recreate, load the RGCN model, export it to ONNX, and use it for inference.
    
    Parameters:
        ntype_dict: Dictionary defining node types.
        etypes: List of edge types.
        in_size: Input size of features.
        hidden_size: Hidden layer size.
        out_size: Output size.
        n_layers: Number of layers.
        embedding_size: Size of embeddings.
        weights_path: Path to the saved weights.
        onnx_path: Path to save the ONNX model.
        device: The device to use (CPU/GPU).
        input_sample: Sample input tensor for ONNX export (ensure it's on CPU).
    """
    # Step 1: Create the PyTorch model and load weights
    model = HeteroRGCN(
        ntype_dict=ntype_dict,
        etypes=etypes,
        in_size=in_size,
        hidden_size=hidden_size,
        out_size=out_size,
        n_layers=n_layers,
        embedding_size=embedding_size
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.eval()

    # Step 2: Export the model to ONNX
    torch.onnx.export(
        model, 
        input_sample,  # Provide a sample input tensor
        onnx_path,
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=12,  # ONNX opset version (ensure compatibility)
        do_constant_folding=True,  # Optimize constant folding for inference
        input_names=["input"],  # Input tensor names
        output_names=["output"],  # Output tensor names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    # Step 3: Verify the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved to {onnx_path} and verified successfully.")

def preprocess_transaction(transaction, g, mean, stdev, target_id_to_node, id_to_node):
    """
    Preprocess a single transaction and rebuild the graph dynamically to integrate it.
    """
    # Extract transaction details
    src_node, dst_node = transaction['source'], transaction['target']
    features = transaction['features']  # A numpy array or list of features

    # Normalize features
    features = features + [0] * (390 - len(features))
    features = (torch.tensor(features) - mean) / stdev
    features = features.float()

    # Extract current graph data
    node_dict = {ntype: g.nodes(ntype) for ntype in g.ntypes}
    edge_dict = {etype: g.edges(etype=etype) for etype in g.canonical_etypes}
    edge_data = {etype: g.edges[etype].data for etype in g.canonical_etypes}

    # Ensure the node types 'source' and 'target' exist
    if 'source' not in node_dict:
        node_dict['source'] = torch.tensor([], dtype=torch.int64)
    if 'target' not in node_dict:
        node_dict['target'] = torch.tensor([], dtype=torch.int64)

    # Define the canonical edge type for 'relation'
    relation_key = ('source', 'relation', 'target')

    # Add new edge to the edge dictionary
    if relation_key not in edge_dict:
        edge_dict[relation_key] = (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))
    edge_dict[relation_key] = (
        torch.cat((edge_dict[relation_key][0], torch.tensor([src_node], dtype=torch.int64))),
        torch.cat((edge_dict[relation_key][1], torch.tensor([dst_node], dtype=torch.int64)))
    )

    # Define num_nodes_dict explicitly for all node types
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    if 'source' not in num_nodes_dict:
        num_nodes_dict['source'] = src_node + 1
    if 'target' not in num_nodes_dict:
        num_nodes_dict['target'] = dst_node + 1

    # Rebuild the graph with updated edges
    new_g = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)

    # Copy over node data
    for ntype in g.ntypes:
        new_g.nodes[ntype].data.update(g.nodes[ntype].data)

    # Copy over edge data for existing edge types
    for etype in g.canonical_etypes:
        if etype in edge_data:
            new_g.edges[etype].data.update(edge_data[etype])

    # Add features for the new edge
    if 'features' in new_g.edges[relation_key].data:
        new_g.edges[relation_key].data['features'] = torch.cat(
            (new_g.edges[relation_key].data['features'], features.unsqueeze(0))
        )
    else:
        new_g.edges[relation_key].data['features'] = features.unsqueeze(0)

    return new_g

def infer_single_transaction(transaction, model, g, mean, stdev, target_id_to_node, id_to_node, device):
    """
    Run inference on a single transaction and return fraud likelihood.
    """
    # Preprocess the transaction
    g = preprocess_transaction(transaction, g, mean, stdev, target_id_to_node, id_to_node)

    # Move graph to the correct device
    g = g.to(device)

    # Extract features for target nodes
    target_features = g.nodes['target'].data['features'].to(device)

    # Run the model
    with torch.no_grad():
        logits = model(g, target_features)

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Return fraud likelihood (class 1 probability)
    return probabilities[:, 1].cpu().numpy()

def load_normalization_metadata():
    """
    Load normalization metadata (mean and std) from the saved file.
    """
    metadata = pickle.load(open("model/metadata.pkl", "rb"))
    mean = metadata['feat_mean']
    stdev = metadata['feat_std']
    return mean, stdev

def load_saved_graph():
    """
    Load the graph saved during training.
    """
    graphs, _ = dgl.load_graphs("output/graph.bin")
    return graphs[0]

app = Flask(__name__)

# Load model and metadata
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metadata = pickle.load(open("model/metadata.pkl", "rb"))
ntype_dict = metadata['ntype_cnt']
etypes = metadata['etypes']
# print(etypes)
etypes = [node for etype in etypes for node in etype]
# print("-"*50)
# print(etypes)
in_feats = metadata['feat_mean'].shape[0]

model = load_model_and_infer_onnx(ntype_dict, etypes, in_feats, 20, 2, 3, 390, "model/model.pth", "model/model.onnx", device)
mean, stdev = load_normalization_metadata()
# print(g.canonical_etypes)

@app.route('/predict', methods=['POST'])
def predict():
    g = load_saved_graph()
    # Parse transaction from request
    transaction = request.json
    node_metadata = pickle.load(open("output/node_mappings.pkl", "rb"))
    target_id_to_node = node_metadata['target_id_to_node']
    id_to_node = node_metadata['id_to_node']
    fraud_likelihood = infer_single_transaction(transaction, model, g, mean, stdev, target_id_to_node, id_to_node, device)
    return jsonify({"fraud_likelihood": fraud_likelihood.tolist()[-1]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)