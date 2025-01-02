import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm
from stagn import STAGNModel
from sklearn.model_selection import train_test_split
from feature_engineering import load_stagn_data


def to_pred(logits: torch.Tensor) -> list:
    """Convert logits to class predictions."""
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()

def stagn_train_2d(
    features,
    labels,
    train_idx,
    test_idx,
    g,
    num_classes: int = 2,
    epochs: int = 18,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    g = g.to(device)

    # Initialize model
    model = STAGNModel(
        time_windows_dim=features.shape[2],
        feat_dim=features.shape[1],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
        g=g,
        device=device
    )
    model.to(device)

    # Prepare data
    features = torch.from_numpy(features).to(device)
    features.transpose_(1, 2)  # Transpose features to match model input format
    labels = torch.from_numpy(labels).to(device)

    # Compute class weights for imbalanced datasets
    unique_labels, counts = torch.unique(labels, return_counts=True)
    weights = (1 / counts) * len(labels) / len(unique_labels)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights)

    # Training loop
    pbar = tqdm(range(epochs), desc="Training Progress")

    for epoch in pbar:
        optimizer.zero_grad()
    
        # Forward pass
        out = model(features, g)
    
        # Compute loss
        loss = loss_func(out[train_idx], labels[train_idx])
    
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
        # Evaluate on training data
        pred = to_pred(out[train_idx])
        true = labels[train_idx].cpu().numpy()
        pred = np.array(pred)
        auc = roc_auc_score(true, pred)
        f1 = f1_score(true, pred, average='macro')
        ap = average_precision_score(true, pred)
    
        # Update progress bar with metrics
        pbar.set_postfix(loss=f"{loss:.4f}", auc=f"{auc:.4f}", f1=f"{f1:.4f}", ap=f"{ap:.4f}")

    # Evaluate on test data
    with torch.no_grad():
        out = model(features, g)
        pred = to_pred(out[test_idx])
        true = labels[test_idx].cpu().numpy()
        pred = np.array(pred)
        print(
            f"Test set | auc: {roc_auc_score(true, pred):.4f}, "
            f"F1: {f1_score(true, pred, average='macro'):.4f}, "
            f"AP: {average_precision_score(true, pred):.4f}"
        )
    return model

def stagn_main(
    features,
    labels,
    test_ratio,
    g,
    mode: str = "2d",
    epochs: int = 18,
    attention_hidden_dim: int = 150,
    lr: float = 0.003,
    device="cpu",
):
    # Split data into training and testing sets
    train_idx, test_idx = train_test_split(
        np.arange(features.shape[0]), test_size=test_ratio, stratify=labels
    )

    # Train the model
    model = stagn_train_2d(
        features,
        labels,
        train_idx,
        test_idx,
        g,
        epochs=epochs,
        attention_hidden_dim=attention_hidden_dim,
        lr=lr,
        device=device
    )

    return model

features, labels, g = load_stagn_data()
model = stagn_main(
    features=features,
    labels=labels,
    test_ratio=0.2,
    g=g,
    epochs=1000,
    attention_hidden_dim=128,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
)