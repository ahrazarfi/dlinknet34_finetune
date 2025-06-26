
"""
Dataset with Albumentations + torchvision tensor conversion
"""
import cv2, os, torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def default_aug(height: int = 1024, width: int = 1024):
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, alpha_coef=0.1, p=0.3),
        A.RandomShadow(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=10, border_mode=cv2.BORDER_REFLECT, p=0.7),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

class RoadsDataset(Dataset):
    def __init__(self, id_list, root, aug=None):
        self.ids  = id_list
        self.root = root
        self.aug  = aug or default_aug()

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path  = os.path.join(self.root, f"{id_}_sat.jpg")
        mask_path = os.path.join(self.root, f"{id_}_mask.png")
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = (mask > 127).astype(np.uint8)
        transformed = self.aug(image=image, mask=mask)
        image_t = transformed['image']
        mask_t  = transformed['mask'].unsqueeze(0).float()
        return image_t, mask_t
