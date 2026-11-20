import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import our dataset and model
from data import get_data
from model import ECG_CNN

def evaluate():
    CSV_PATH = "../data/balanced_augmented_ecg_data.csv"
    MODEL_PATH = "../models/ecg_cnn.pth"
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")
    # We only need the test_dataset here
    _, test_dataset = get_data(CSV_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading model...")
    model = ECG_CNN(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Print the Classification Report
    print("\n--- Classification Report ---")
    # Classes are 0, 1, 2, 3
    class_names = ['Normal (0)', 'Class 1', 'Class 2', 'Class 3']
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 2. Plot the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    print("Showing plot.)")
    plt.savefig("../models/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.show()

if __name__ == "__main__":
    evaluate()