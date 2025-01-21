import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


train_data = pd.read_csv("data/ieee-fraud-detection/train_transaction.csv")
test_data = pd.read_csv("data/ieee-fraud-detection/test_transaction.csv")

numeric_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
train_data[numeric_columns] = train_data[numeric_columns].fillna(train_data[numeric_columns].median())

categorical_columns = train_data.select_dtypes(include=['object']).columns
train_data = pd.get_dummies(train_data, columns=categorical_columns, drop_first=True)

X = train_data.drop(['isFraud'], axis=1)
y = train_data['isFraud']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

class FraudDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First layer
        self.relu = nn.ReLU()                # Activation function
        self.fc2 = nn.Linear(128, 64)        # Second layer
        self.fc3 = nn.Linear(64, 1)          # Output layer
        self.sigmoid = nn.Sigmoid()          # Sigmoid for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

input_dim = X_train.shape[1]
model = FraudDetectionNN(input_dim)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
batch_size = 32

# Training loop
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_accuracy = ((val_outputs > 0.5).float() == y_val_tensor).float().mean()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy.item():.4f}")


# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_val_tensor)
    y_pred = (y_pred > 0.5).float()
    y_pred = y_pred.numpy()
    y_val = y_val_tensor.numpy()

    auc = roc_auc_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")



