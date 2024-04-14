#%%
import torch
import torch.utils
from torch.utils.data import DataLoader

from train import * 
from utils import *
from data_loader import *
from model import Unet

#%%
# HYPERPARAMETERS
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

BATCH_SIZE = 3
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_LOAD = True

# %%
train_data = Nuclei_Loader('data/dataset/train')
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data = Nuclei_Loader('data/dataset/val')
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# %%
model = Unet(in_channels=3).to(DEVICE)
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# %%
if MODEL_LOAD:
    try:
        load_model(model=model, save_path=f'data/saves/model/BCE_Loss_50_epochs.pth.tar')
    except FileNotFoundError:
        print('No save file was found...')
# %%
for epoch in range(EPOCHS):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    
    train_loss = train_model(train_data, device = DEVICE, model = model, loss_function = loss, optimizer = optimizer, epochs = EPOCHS, epoch = epoch)
    val_loss, val_accuracy = validate_model(train_data, device = DEVICE, model = model, loss_function = loss)
    
    print(f'''
          \nFor epoch [{epoch}/{EPOCHS}]
            Train Loss: {train_loss}
            Validation Loss: {val_loss}
            Validation Accuracy: {val_accuracy}%
            \n
          ''')
    
    save_model(checkpoint=checkpoint, save_path=f'data/saves/model/BCE_Loss_{epoch}_epochs.pth.tar')
    
# %%