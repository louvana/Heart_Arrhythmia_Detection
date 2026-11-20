import torch.nn as nn

class ECG_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECG_CNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        # Block 3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Classifier
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        x = self.adaptive_pool(x).squeeze(-1) 
        x = self.fc(x)                        
        return x