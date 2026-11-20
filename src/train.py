import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

from data import get_data
from model import ECG_CNN

def train():
    # 1. Config
    # Path to the CSV (going up one level from src/)
    CSV_PATH = "../data/balanced_augmented_ecg_data.csv"
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. Load Data
    train_dataset, test_dataset = get_data(CSV_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Handle Class Imbalance (The model will penalize mistakes on Class 1 more)
    train_labels = train_dataset.y.numpy()
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    # 4. Initialize Model, Loss, Optimizer
    model = ECG_CNN(num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
        test_acc = 100. * test_correct / test_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # 6. Save Model
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/ecg_cnn.pth")
    print("Model saved to models/ecg_cnn.pth")

if __name__ == "__main__":
    train()