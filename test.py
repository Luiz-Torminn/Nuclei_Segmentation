#%%
import torch
import torch.utils
from torch.utils.data import DataLoader

from model import Unet
from utils.train import * 
from utils.utils import *
from data_loader import *
from utils.graphs import *

# %%
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_DICT_PATH = 'data/saves/model/BCE_Loss_50_epochs.pth.tar'

# %%
model = Unet(in_channels=3).to(DEVICE)
# load_model(model, MODEL_DICT_PATH)

test_data = Nuclei_Loader('data/dataset/test')
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# %%
image, mask = next(iter(test_loader))
image = image.to(DEVICE)

# %%
prediction = torch.sigmoid(model(image))
prediction = (prediction > 0.5).float()

# %%
prepare_data = lambda var: var.cpu().numpy().squeeze()

prediction = prepare_data(prediction)
image = prepare_data(image).transpose((1,2,0))
mask = mask.squeeze()

# %%
plot_image_mask(image, mask, prediction, show = True)
# %%
