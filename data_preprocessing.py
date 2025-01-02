import pandas as pd
import numpy as np
import dgl
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch

# Step 1: Feature Engineering using featmap_gen
def featmap_gen(tmp_card, tmp_df=None):
    """
    Generate aggregated features for transactions grouped by a specific card/account (Source).
    """
    time_span = [5, 20, 60, 300, 600]  # Example time spans in seconds
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']  # Transaction timestamps
    post_fe = []

    # Iterate through each transaction for the given card
    for trans_idx, trans_feat in tmp_df.iterrows():
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount

        # Generate features for each time span
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]

            # Aggregated features for the time span
            new_df[f'trans_at_avg_{tname}'] = correct_data['Amount'].mean()
            new_df[f'trans_at_totl_{tname}'] = correct_data['Amount'].sum()
            new_df[f'trans_at_std_{tname}'] = correct_data['Amount'].std()
            new_df[f'trans_at_bias_{tname}'] = temp_amt - correct_data['Amount'].mean()
            new_df[f'trans_at_num_{tname}'] = len(correct_data)
            new_df[f'trans_target_num_{tname}'] = len(correct_data['Target'].unique())
            new_df[f'trans_location_num_{tname}'] = len(correct_data['Location'].unique())
            new_df[f'trans_type_num_{tname}'] = len(correct_data['Type'].unique())

        post_fe.append(new_df)
    return pd.DataFrame(post_fe)


def preprocess_and_build_graph(data_path, graph_output_path):
    """
    Preprocess the S-FFSD dataset and transform it into a graph format using DGL.
    """
    # Load and sort data by time
    data = pd.read_csv(data_path)
    data = data[data["Labels"] != 2]
    data['Time'] = pd.to_numeric(data['Time'])  # Ensure time is numeric
    data = data.sort_values(by='Time')  # Sort transactions chronologically

    # Step 2: Feature Generation
    print("Starting feature generation...")
    all_features = []
    for source, group in tqdm(data.groupby('Source'), desc="Generating Features"):
        all_features.append(featmap_gen(source, group))
    processed_data = pd.concat(all_features).reset_index(drop=True)

    # Save the processed data for inspection (optional)
    processed_data.to_csv("sffsd_preprocessed.csv", index=False)
    print("Feature engineering completed. Saved as sffsd_preprocessed.csv.")

    # Step 3: Graph Construction
    print("Constructing graph...")
    data = processed_data.reset_index(drop=True)
    alls, allt = [], []
    edge_per_trans = 3
    pair = ["Source", "Target", "Location", "Type"]

    # Build edges based on shared attributes and temporal proximity
    for column in pair:
        src, tgt = [], []
        for c_id, c_df in tqdm(data.groupby(column), desc=f"Building edges for {column}"):
            c_df = c_df.sort_values(by="Time")
            df_len = len(c_df)
            sorted_idxs = c_df.index
            src.extend([sorted_idxs[i] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
            tgt.extend([sorted_idxs[i + j] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)

    # Create a DGL graph
    alls = np.array(alls)
    allt = np.array(allt)
    g = dgl.graph((alls, allt))

    # Encode categorical columns as numerical features
    cal_list = ["Source", "Target", "Location", "Type"]
    for col in cal_list:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)

    # Add node features and labels to the graph
    labels = data["Labels"]
    feat_data = data.drop("Labels", axis=1)
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

    # Add edge features (example using Amount, Location, Type, and Time difference)
    edge_features = []
    for idx in range(len(alls)):
        edge_feats = []
        # Add edge-specific features like Amount, Location, Type, etc.
        edge_feats.append(data.loc[alls[idx], 'Amount'])
        edge_feats.append(data.loc[alls[idx], 'Location'])
        edge_feats.append(data.loc[alls[idx], 'Type'])
        
        # Temporal feature: time difference between transactions
        time_diff = data.loc[alls[idx], 'Time'] - data.loc[allt[idx], 'Time']
        edge_feats.append(time_diff)
        
        edge_features.append(edge_feats)

    # Convert to numpy array
    edge_features = np.array(edge_features)

    # Add edge features to the graph
    g.edata['feat'] = torch.from_numpy(edge_features).to(torch.float32)

    # Save the graph
    dgl.data.utils.save_graphs(graph_output_path, [g])
    print(f"Graph saved to {graph_output_path}")


# Run the combined preprocessing and graph construction
preprocess_and_build_graph("S-FFSD.csv", "graph-S-FFSD.bin")