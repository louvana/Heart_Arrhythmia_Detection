import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ECGDataset(Dataset):
    def __init__(self, X, y):
        # PyTorch 1D CNN expects (Batch, Channels, Length). 
        # We add a channel dimension (1) here.
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data(csv_path, test_size=0.2, random_state=42):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Separate features (first 160) and labels (last column)
    X = df.iloc[:, :160].values
    y = df.iloc[:, -1].values
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(y_train)} | Test set: {len(y_test)}")
    
    return ECGDataset(X_train, y_train), ECGDataset(X_test, y_test)