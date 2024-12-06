import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
from dataset import TextureWindowDataset
from model import SimpleCNN
from pretrained_model import get_pretrained_model
from tqdm import tqdm
from augmentations import make_augmentations
from patch import PatchShuffleTransform
import datetime
import sys

# Redirect print output to a log file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def calculate_metrics_per_class(correct_per_class, total_per_class):
    num_classes = len(correct_per_class)
    acc_per_class = []
    for c in range(num_classes):
        if total_per_class[c] > 0:
            acc = 100.0 * correct_per_class[c] / total_per_class[c]
        else:
            acc = 0.0
        acc_per_class.append(acc)
    return acc_per_class

def train_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    correct_per_class = [0]*num_classes
    total_per_class = [0]*num_classes
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        for lbl, pred in zip(labels, predicted):
            total_per_class[lbl.item()] += 1
            if pred.item() == lbl.item():
                correct_per_class[lbl.item()] += 1
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    per_class_acc = calculate_metrics_per_class(correct_per_class, total_per_class)
    return epoch_loss, epoch_acc, per_class_acc

def test_epoch(model, test_loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    correct_per_class = [0]*num_classes
    total_per_class = [0]*num_classes

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for lbl, pred in zip(labels, predicted):
                total_per_class[lbl.item()] += 1
                if pred.item() == lbl.item():
                    correct_per_class[lbl.item()] += 1
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    per_class_acc = calculate_metrics_per_class(correct_per_class, total_per_class)
    return epoch_loss, epoch_acc, per_class_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action='store_true', help="Use pretrained model if set")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of data to use as test set")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--patch", action='store_true', help="Use patch-and-shuffle approach if set")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer")  # Added flag for learning rate
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file name")
    args = parser.parse_args()

    # Configure log file
    if args.log_file:
        log_filename = args.log_file
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"slurm_jobs/outputs/training-metrics-{timestamp}.log"
    
    sys.stdout = Logger(log_filename)

    # Hyperparameters
    batch_size = 32
    lr = args.learning_rate  # Use the learning rate from the flag
    num_classes = 4
    base_size = 128
    test_split = args.test_split
    epochs = args.epochs
    weight_decay = args.weight_decay

    if args.patch:
        print("Using PATCH approach!!")
    else:
        print("Using STANDARD approach")

    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    
    # Paths
    data_dir = "data/texture_windows"
    csv_path = "data/texture_windows-labels.csv"
    save_path = "model.pth"
    
    base_resize_transform = transforms.Resize((base_size, base_size))
    
    full_dataset = TextureWindowDataset(csv_path=csv_path, images_dir=data_dir, transform=base_resize_transform)
    
    # Train/Test split
    test_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Build transforms depending on patch flag
    # If patch is True, we insert PatchShuffleTransform before ToTensor and Normalize.
    if args.patch:
        final_transform = transforms.Compose([
            PatchShuffleTransform(patch_size=16),  # you can change patch_size if needed
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    
    class FinalTransformDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            image, label = self.base_dataset[idx]
            image = self.transform(image)
            return image, label
    
    class AugTransformDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, aug_transform, final_transform):
            self.base_dataset = base_dataset
            self.aug_transform = aug_transform
            self.final_transform = final_transform
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            image, label = self.base_dataset[idx]
            image = self.aug_transform(image)
            image = self.final_transform(image)
            return image, label
    
    aug_list = make_augmentations(base_size=base_size)

    # Training set: original + augmented
    train_original = FinalTransformDataset(train_dataset, final_transform)
    aug_datasets = [AugTransformDataset(train_dataset, aug_t, final_transform) for aug_t in aug_list]
    combined_train_dataset = ConcatDataset([train_original] + aug_datasets)
    
    # Testing set: original + augmented
    test_original = FinalTransformDataset(test_dataset, final_transform)
    aug_datasets_test = [AugTransformDataset(test_dataset, aug_t, final_transform) for aug_t in aug_list]
    combined_test_dataset = ConcatDataset([test_original] + aug_datasets_test)
    
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model selection
    if args.pretrained:
        model = get_pretrained_model(num_classes=num_classes).to(device)
    else:
        model = SimpleCNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        train_loss, train_acc, train_class_acc = train_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        test_loss, test_acc, test_class_acc = test_epoch(model, test_loader, criterion, device, num_classes)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        for c, acc in enumerate(train_class_acc):
            print(f"  Train Class {c} Acc: {acc:.2f}%")
        
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        for c, acc in enumerate(test_class_acc):
            print(f"  Test Class {c} Acc: {acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("Model saved.")

    print("Training complete.")
    print(f"Best Test Acc: {best_test_acc:.2f}%")
