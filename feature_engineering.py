import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import zscore
import dgl
import torch

def calcu_trading_entropy(data_2: pd.DataFrame) -> float:
    """Calculate trading entropy of given data."""
    if len(data_2) == 0:
        return 0

    amounts = np.array([data_2[data_2['Type'] == type]['Amount'].sum()
                        for type in data_2['Type'].unique()])
    proportions = amounts / amounts.sum() if amounts.sum() else np.ones_like(amounts)
    ent = -np.array([proportion * np.log(1e-5 + proportion)
                    for proportion in proportions]).sum()
    return ent

def span_data_2d(data: pd.DataFrame, time_windows: list = [1, 3, 5, 10, 20, 50, 100, 500]) -> np.ndarray:
    """Transform transaction records into feature matrices."""
    data = data[data['Labels'] != 2]  # Exclude unlabeled entries
    
    nume_feature_ret, label_ret = [], []
    for row_idx in tqdm(range(len(data))):
        record = data.iloc[row_idx]
        acct_no = record['Source']
        feature_of_one_record = []

        for time_span in time_windows:
            feature_of_one_timestamp = []
            prev_records = data.iloc[(row_idx - time_span):row_idx, :]
            prev_and_now_records = data.iloc[(row_idx - time_span):row_idx + 1, :]
            prev_records = prev_records[prev_records['Source'] == acct_no]

            # AvgAmountT
            feature_of_one_timestamp.append(prev_records['Amount'].sum() / time_span)
            # TotalAmountTs
            feature_of_one_timestamp.append(prev_records['Amount'].sum())
            # BiasAmountT
            feature_of_one_timestamp.append(record['Amount'] - feature_of_one_timestamp[0])
            # NumberT
            feature_of_one_timestamp.append(len(prev_records))
            # TradingEntropyT
            old_ent = calcu_trading_entropy(prev_records[['Amount', 'Type']])
            new_ent = calcu_trading_entropy(prev_and_now_records[['Amount', 'Type']])
            feature_of_one_timestamp.append(old_ent - new_ent)

            feature_of_one_record.append(feature_of_one_timestamp)

        nume_feature_ret.append(feature_of_one_record)
        label_ret.append(record['Labels'])

    nume_feature_ret = np.array(nume_feature_ret).transpose(0, 2, 1)
    assert nume_feature_ret.shape == (len(data), 5, len(time_windows)), "Output shape invalid."
    return nume_feature_ret.astype(np.float32), np.array(label_ret).astype(np.int64)

def load_stagn_data(test_size: float = 0.2):
    """Load and transform the S-FFSD dataset for graph models."""
    data_path = "S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - test_size

    if os.path.exists("features.npy"):
        features, labels = np.load("features.npy"), np.load("labels.npy")
    else:
        features, labels = span_data_2d(feat_df)
        np.save("features.npy", features)
        np.save("labels.npy", labels)

    sampled_df = feat_df[feat_df['Labels'] != 2].reset_index(drop=True)
    all_nodes = pd.concat([sampled_df['Source'], sampled_df['Target']]).unique()
    encoder = LabelEncoder().fit(all_nodes)
    encoded_source = encoder.transform(sampled_df['Source'])
    encoded_tgt = encoder.transform(sampled_df['Target'])

    loc_enc = OneHotEncoder()
    loc_feature = np.array(loc_enc.fit_transform(
        sampled_df['Location'].to_numpy()[:, np.newaxis]).todense())
    loc_feature = np.hstack(
        [zscore(sampled_df['Amount'].to_numpy())[:, np.newaxis], loc_feature])

    g = dgl.DGLGraph()
    g.add_edges(encoded_source, encoded_tgt, data={"feat": torch.from_numpy(loc_feature).to(torch.float32)})
    return features, labels, g
