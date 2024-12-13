import random
import torch
import torchvision.transforms as transforms
from PIL import Image

class PatchShuffleTransform:
    """ 
    A transform that divides the image into patches, shuffles the patches, and then recombines them.
    Assumes a square image for simplicity. 
    """
    def __init__(self, patch_size=16):
        self.patch_size = patch_size
        
    def __call__(self, img):
        # img is a PIL Image. Convert to tensor to manipulate easily.
        img_tensor = transforms.functional.to_tensor(img)  # C, H, W in [0,1]
        C, H, W = img_tensor.shape
        
        # Ensure H and W are divisible by patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image size not divisible by patch size"
        
        # Number of patches along each dimension
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Reshape into (C, num_patches_h, patch_size, num_patches_w, patch_size)
        patches = img_tensor.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        # patches shape: C, num_patches_h, num_patches_w, patch_size, patch_size
        
        # Rearrange to have patches in a list
        # Transpose to get shape: num_patches_h, num_patches_w, C, patch_size, patch_size
        patches = patches.permute(1, 2, 0, 3, 4)  # (num_patches_h, num_patches_w, C, patch_size, patch_size)
        
        # Flatten into a list of patches
        patches_list = patches.reshape(num_patches_h * num_patches_w, C, self.patch_size, self.patch_size)
        
        # Shuffle patches
        patches_list = list(patches_list)
        random.shuffle(patches_list)
        
        # Stack them back
        patches_shuffled = torch.stack(patches_list, dim=0)
        
        # Reshape back into image grid
        patches_shuffled = patches_shuffled.reshape(num_patches_h, num_patches_w, C, self.patch_size, self.patch_size)
        # Move channels back to first dim
        patches_shuffled = patches_shuffled.permute(2, 0, 3, 1, 4)  # C, num_patches_h, patch_size, num_patches_w, patch_size
        # Combine patches along H dimension
        patches_shuffled = patches_shuffled.reshape(C, num_patches_h * self.patch_size, num_patches_w * self.patch_size)
        
        # Convert back to PIL image
        img_shuffled = transforms.functional.to_pil_image(patches_shuffled)
        return img_shuffled