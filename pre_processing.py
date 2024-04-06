#%%
import cv2
import re
import os
import numpy as np
from PIL import Image
from patchify import patchify
from sklearn.model_selection import train_test_split

# %%
def create_dataset_folders():
    FOLDERS = ['train', 'val', 'test']
    
    for folder in FOLDERS:
        if not os.path.exists(f'data/dataset/{folder}'):
            os.mkdir(f'data/dataset/{folder}')
        else:
            print('Files already exist...')
            
    [os.mkdir(f'data/dataset/{folder}/images') for folder in FOLDERS if not os.path.exists(f'data/dataset/{folder}/images')]
    [os.mkdir(f'data/dataset/{folder}/masks') for folder in FOLDERS if not os.path.exists(f'data/dataset/{folder}/masks')]

create_dataset_folders()
 
# %%
def create_patches(file_path):
    img = Image.open(file_path)
    file_name = os.path.split(file_path)[1]
    file_type = re.findall(r'[^nucleus_]\w*[^.tiff]', file_name)[0]
    
    for n in range(img.n_frames):
        img.seek(n)
        image_data = np.asarray(img)
        
        img_patches = patchify(image_data, (512,512), step = 200) #<-- (2, 3, 512, 512)
        
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                patch = img_patches[i,j]
                file_name = f"{file_type}_tile_{n+1}_patch_{(i * 3) + (j+1)}.jpg"
                [cv2.imwrite(f'data/raw_data/{file_type}/{file_name}', patch) if not os.path.exists(f'data/raw_data/{file_type}/{file_name}') else print(f'File {file_name} already exists inside data/raw_data/{file_type}')]

# %%   
create_patches('data/raw_data/nucleus_images.tiff')
create_patches('data/raw_data/nucleus_masks.tiff')

# %%
def divide_data(image_path, masks_path, image_dest, mask_dest):
    images = sorted(os.listdir(image_path))
    masks = sorted(os.listdir(masks_path))
    
    # Divide data into train, validation and test
    train_image, val_image, train_mask, val_mask = train_test_split(images, masks, test_size=0.2, train_size=0.8)
    
    val_image, test_image, val_mask, test_mask = train_test_split(val_image, val_mask, test_size=0.1, train_size=0.9)
    
    # Realocate data to their correspoding dataset assigned files:
    # Train
    [os.rename(f'{image_path}/{i}', f'{image_dest}/train/images/{i}') for i in train_image if not os.path.exists(f'{image_dest}/train/images/{i}')]
    [os.rename(f'{masks_path}/{i}', f'{mask_dest}/train/masks/{i}') for i in train_mask if not os.path.exists(f'{mask_dest}/train/images/{i}')]
    
    # Validation
    [os.rename(f'{image_path}/{i}', f'{image_dest}/val/images/{i}') for i in val_image if not os.path.exists(f'{image_dest}/val/images/{i}')]
    [os.rename(f'{masks_path}/{i}', f'{mask_dest}/val/masks/{i}') for i in val_mask if not os.path.exists(f'{mask_dest}/val/images/{i}')]
    
    # Test
    [os.rename(f'{image_path}/{i}', f'{image_dest}/test/images/{i}') for i in test_image if not os.path.exists(f'{image_dest}/test/images/{i}')]
    [os.rename(f'{masks_path}/{i}', f'{mask_dest}/test/masks/{i}') for i in test_mask if not os.path.exists(f'{mask_dest}/test/images/{i}')]
    
    
 # %%   
divide_data('data/raw_data/images', 'data/raw_data/masks', 'data/dataset', 'data/dataset')

