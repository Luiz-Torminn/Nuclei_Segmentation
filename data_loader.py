#%%
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import re
import cv2
from pathlib import Path
#%%
class Nuclei_Loader(Dataset):
    def __init__(self, dir_path):
        super().__init__()
        self.image_files = sorted([f'{dir_path}/images/{i}' for i in os.listdir(f'{dir_path}/images')])
        self.mask_files = sorted([f'{dir_path}/masks/{i}' for i in os.listdir(f'{dir_path}/masks')])
        
        # Check if images have their corresponding masks
        self.img_stems = [re.findall(r'[^\simages_|masks_]\w*[^.jpg\s]', Path(self.image_files[i]).stem)[0] for i in range(len(self.image_files)) if i != '.DS_Store']
        self.mask_stems = [re.findall(r'[^\simages_|masks_]\w*[^.jpg\s]', Path(self.mask_files[i]).stem)[0] for i in range(len(self.mask_files)) if i != '.DS_Store']
        
        self.intersection = set(self.img_stems) & set(self.mask_stems)
        
        self.image_files = sorted([self.image_files[i] for i in range(len(self.image_files)) if re.findall(r'[^\simages_|masks_]\w*[^.jpg\s]', Path(self.image_files[i]).stem)[0] in self.intersection] )
        self.mask_files = sorted([self.mask_files[i] for i in range(len(self.mask_files)) if re.findall(r'[^\simages_|masks_]\w*[^.jpg\s]', Path(self.mask_files[i]).stem)[0] in self.intersection] )
        
        # transformation function
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Tranform images
        image = self.transformation(image)
        
        # Open and stablish classifiction for
        mask = cv2.imread(self.mask_files[index], 0)
        mask = self.transformation(mask)
        
        return (image, mask)

# %%
# dir_path = 'data/dataset/test'

# image_files = [f'{dir_path}/images/{i}' for i in os.listdir(f'{dir_path}/images')]
# mask_files = [f'{dir_path}/masks/{i}' for i in os.listdir(f'{dir_path}/masks')]





# # Check if images have their corresponding masks
# img_stems = [re.findall(r'[^\simages_|masks_]\w*[^.jpg\s]', Path(image_files[i]).stem)[0] for i in range(len(image_files)) if i != '.DS_Store']
# mask_stems = [re.findall(r'[^\simages_|masks_]\w*[^.jpg\s]', Path(mask_files[i]).stem)[0] for i in range(len(mask_files)) if i != '.DS_Store']

# intersection = set(img_stems) & set(mask_stems)
        
# image_files = sorted([image_files[i] for i in range(len(image_files)) if re.findall(r'[^\simages_|masks_]\w*[^.jpg\s]', Path(image_files[i]).stem)[0] in intersection] )
# mask_files = sorted([mask_files[i] for i in range(len(mask_files)) if re.findall(r'[^\simages_|masks_]\w*[^.jpg\s]', Path(mask_files[i]).stem)[0] in intersection] )
# %%
