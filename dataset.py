import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TextureWindowDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file with image names and labels.
            images_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image"]
        label = int(self.data.iloc[idx]["class"])
        
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
