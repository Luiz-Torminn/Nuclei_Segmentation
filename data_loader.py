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
        self.image_files = [f'{dir_path}/images/{i}' for i in os.listdir(f'{dir_path}/image')]
        self.mask_files = [f'{dir_path}/masks/{i}' for i in os.listdir(f'{dir_path}/masks')]
        
        # Check if images have their corresponding masks
        self.img_stems = [re.findall(r'#[^images_|masks_]\w*[^.jpg]', Path(i).stem)[0] for i in sorted(self.image_files) if i != '.DS_Store']
        self.mask_stems = [re.findall(r'#[^images_|masks_]\w*[^.jpg]', Path(i).stem)[0] for i in sorted(self.mask_files) if i != '.DS_Store']
        
        self.intersection = set(self.img_stems) & set(self.mask_stems)
        
        self.image_files = [i for i in self.image_files if re.findall(r'#[^images_|masks_]\w*[^.jpg]', Path(i).stem)[0] in self.img_stems] 
        self.mask_files = [i for i in self.mask_files if re.findall(r'#[^images_|masks_]\w*[^.jpg]', Path(i).stem)[0] in self.mask_stems] 
        
        # transformation function
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image = cv2.imread(sorted(self.image_files[index]))
        image = cv2.cvtcolor(image, 'BGR2RGB')
        
        # Tranform images
        image = self.transformation(image)
        
        # Open and stablish classifiction for
        mask = cv2.imread(sorted(self.mask_files[index]), 0)
        
        return (image, mask)

# %%
