import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
from dataset import TextureWindowDataset
from model import SimpleCNN
from tqdm import tqdm
from augmentations import make_augmentations

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    lr = 0.001
    epochs = 10
    test_split = 0.2
    num_classes = 4
    base_size = 128
    
    # Paths
    data_dir = "data/texture_windows"
    csv_path = "data/texture_windows-labels.csv"
    save_path = "model.pth"
    
    # Base resizing transform (applied to all images before augmentation)
    # No ToTensor/Normalize yet, as we need PIL images for augmentation
    base_resize_transform = transforms.Resize((base_size, base_size))
    
    # Create the dataset with only resizing done here
    full_dataset = TextureWindowDataset(csv_path=csv_path, images_dir=data_dir, transform=base_resize_transform)
    
    # Split dataset into train and test
    test_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create a dataset wrapper to apply final transformations (ToTensor, Normalize) 
    # for sets that don't need augmentation or have already been augmented.
    class FinalTransformDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
            self.final_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            image, label = self.base_dataset[idx]  # image is PIL, already resized
            image = self.final_transform(image)
            return image, label
    
    # Define augmentation datasets
    # We'll produce 16 augmented versions per image, as described
    aug_list = make_augmentations(base_size=base_size)
    
    # Dataset wrapper for augmentation:
    class AugTransformDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, aug_transform):
            self.base_dataset = base_dataset
            self.aug_transform = aug_transform
            # We'll apply aug_transform to the PIL image and then do final transforms.
            self.final_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            image, label = self.base_dataset[idx]  # PIL image, resized
            # Apply augmentation
            image = self.aug_transform(image)
            # Final transform to Tensor and normalize
            image = self.final_transform(image)
            return image, label
    
    # Original training dataset without augmentation, just final transforms
    train_original = FinalTransformDataset(train_dataset)
    
    # Augmented datasets
    aug_datasets = [AugTransformDataset(train_dataset, aug_t) for aug_t in aug_list]
    
    # Combine original training + augmented versions
    combined_train_dataset = ConcatDataset([train_original] + aug_datasets)
    
    # Test dataset: only original images, no augmentation, just final transforms
    test_final = FinalTransformDataset(test_dataset)
    
    # DataLoaders
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_final, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_test_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Save best model based on test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("Model saved.")

    print("Training complete.")
    print(f"Best Test Acc: {best_test_acc:.2f}%")
