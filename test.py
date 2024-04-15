#%%
import torch
import torch.utils
from torch.utils.data import DataLoader

from train import * 
from utils import *
from data_loader import *
from model import Unet
from graphs import plot_loss

# %%
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

MODEL_DICT_PATH = 'data/saves/model'

# %%
model = Unet().to(DEVICE)
load_model(model, MODEL_DICT_PATH)

test_data = Nuclei_Loader('data/dataset/test')
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)


# %%
for image, mask in next(iter(test_loader)):
    gpu_image, gpu_mask = image.to(DEVICE), mask.to(DEVICE)
    
    prediction = model(gpu_image)
    prediction = torch.max(prediction, 1)
    print(prediction.shape)
    
    # prediction = torch.transpose(prediction, ())
    
    
    


