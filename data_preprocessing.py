import os
import numpy as np
import pandas as pd

transaction_df = pd.read_csv('data/ieee-fraud-detection/train_transaction.csv')
identity_df = pd.read_csv('data/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('data/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('data/ieee-fraud-detection/test_identity.csv')

id_cols = ['card1','card2','card3','card4','card5','card6','ProductCD','addr1','addr2','P_emaildomain','R_emaildomain']
cat_cols = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']
train_data_ratio = 0.8

n_train = int(transaction_df.shape[0]*train_data_ratio)
test_ids = transaction_df.TransactionID.values[n_train:]

get_fraud_frac = lambda series: 100 * sum(series)/len(series)
print("Percent fraud for train transactions: {}".format(get_fraud_frac(transaction_df.isFraud[:n_train])))
print("Percent fraud for test transactions: {}".format(get_fraud_frac(transaction_df.isFraud[n_train:])))
print("Percent fraud for all transactions: {}".format(get_fraud_frac(transaction_df.isFraud)))

with open('data/test.csv', 'w') as f:
    f.writelines(map(lambda x: str(x) + "\n", test_ids))

non_feature_cols = ['isFraud', 'TransactionDT'] + id_cols
print(non_feature_cols)

feature_cols = [col for col in transaction_df.columns if col not in non_feature_cols]
print(feature_cols)

features = pd.get_dummies(transaction_df[feature_cols], columns=cat_cols).fillna(0)
features['TransactionAmt'] = features['TransactionAmt'].apply(np.log10)

print(list(features.columns))

features.to_csv('data/features.csv', index=False, header=False)
transaction_df[['TransactionID', 'isFraud']].to_csv('data/tags.csv', index=False)
edge_types = id_cols + list(identity_df.columns)
all_id_cols = ['TransactionID'] + id_cols
full_identity_df = transaction_df[all_id_cols].merge(identity_df, on='TransactionID', how='left')
full_identity_df.head(5)

edges = {}
for etype in edge_types:
    edgelist = full_identity_df[['TransactionID', etype]].dropna()
    edgelist.to_csv('data/relation_{}_edgelist.csv'.format(etype), index=False, header=True)
    edges[etype] = edgelist


import glob

file_list = glob.glob('./data/*edgelist.csv')

edges = ",".join(map(lambda x: x.split("/")[-1], [file for file in file_list if "relation" in file]))

edges_full = ''
for etype in edge_types:
    edges_full += ',data/relation_{}_edgelist.csv'.format(etype)
