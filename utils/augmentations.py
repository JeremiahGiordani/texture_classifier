import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class IlluminationTransform:
    """
    Applies a linear intensity transform: out = alpha * img + beta
    
    This transform assumes the image is a PIL Image or a torch Tensor in [0,1].
    After the transform, values may need to be clipped. We'll clip to [0,1].
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        # Convert to tensor if not already
        was_pil = False
        if not torch.is_tensor(img):
            was_pil = True
            img = F.to_tensor(img)  # Converts to C,H,W in [0,1]
        
        img = img * self.alpha + (self.beta / 255.0)  # Beta is given as intensity shift, normalize by 255
        img = torch.clamp(img, 0, 1)
        
        if was_pil:
            img = F.to_pil_image(img)
        
        return img

def make_augmentations(base_size=128):
    """
    Create a list of 16 transforms as specified in the paper.
    
    Transform sets:
    - Rotations: angles ∈ {−4,−3,−2,2,3,4}
    - Zooms: scale factors ∈ {0.97,0.98,0.99,1.01,1.02,1.03}
    - Illumination: (alpha/beta) ∈ {(0.9,0), (1.1,0), (1.0,-50), (1.0,50)}

    Note on zoom:
    We'll simulate zoom by resizing the image to a slightly larger or smaller size 
    and then center-cropping back to the base size.
    For a scale factor s, we resize the image to int(s * base_size).
    If s > 1, we get a larger image and center crop down to base_size (zoom in).
    If s < 1, we get a smaller image and then by center-cropping we effectively scale it down (zoom out).

    After the transform, we will still apply the same normalization as original.
    """
    # Rotations
    rotation_angles = [-4, -3, -2, 2, 3, 4]
    rotation_transforms = [transforms.RandomRotation((angle, angle)) for angle in rotation_angles]

    # Zooms
    # We'll assume a symmetric approach: resize, then center crop
    zoom_scales = [0.97, 0.98, 0.99, 1.01, 1.02, 1.03]
    zoom_transforms = []
    for s in zoom_scales:
        new_size = int(base_size * s)
        zoom_t = transforms.Compose([
            transforms.Resize((new_size, new_size)),
            transforms.CenterCrop((base_size, base_size))
        ])
        zoom_transforms.append(zoom_t)

    # Illumination
    # alpha/beta pairs: 0.9/0, 1.1/0, 1.0/-50, 1.0/50
    illum_pairs = [(0.9,0), (1.1,0), (1.0,-50), (1.0,50)]
    illum_transforms = [IlluminationTransform(a,b) for (a,b) in illum_pairs]

    # Combine all
    aug_transforms = rotation_transforms + zoom_transforms + illum_transforms

    return aug_transforms
