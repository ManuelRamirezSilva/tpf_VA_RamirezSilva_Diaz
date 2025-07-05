import copy
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader import class_names
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    return running_loss / len(loader), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = 100 * correct / total
    return running_loss / len(loader), acc, y_true, y_pred

def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, save_path="resnet18_best.pth", device='cuda'):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device= device)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved!")

    print(f"\nTotal training time: {(time.time() - start_time)/60:.2f} min")
    model.load_state_dict(best_model_wts)

    # Evaluation
    print("\nFinal Evaluation:")
    
    # 🔧 Dynamically select only present classes
    present_labels = sorted(list(unique_labels(y_true, y_pred)))
    present_class_names = [class_names[i] for i in present_labels]

    print(classification_report(y_true, y_pred, target_names=present_class_names))

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    # 🔧 Use only up to 20 classes from present ones
    selected_indices = sorted(random.sample(present_labels, min(40, len(present_labels))))
    selected_names = [class_names[i] for i in selected_indices]
    cm_20 = cm[np.ix_([present_labels.index(i) for i in selected_indices],
                      [present_labels.index(i) for i in selected_indices])]

    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_20, display_labels=selected_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap='Blues', colorbar=True)
    plt.title("Confusion Matrix (20 Classes Subset)")
    plt.tight_layout()
    plt.show()