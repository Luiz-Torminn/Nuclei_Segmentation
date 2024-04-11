#%%
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import torch
from torch.utils.data import DataLoader
from train import * 
from data_loader import *
import torch.utils
from utils import *

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
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights = weights).to(DEVICE)
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# %%
if MODEL_LOAD:
    load_model(model=model, save_path=f'data/saves/model/BCE_Loss_50_epochs.pth.tar')

# %%
for epoch in range(EPOCHS):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    
    train_loss = train_model(train_data, device = DEVICE, model = model, loss = loss, optimizer = optimizer, epochs = EPOCHS, epoch = epoch)
    val_loss, val_accuracy = validate_model(train_data, device = DEVICE, model = model, loss = loss, optimizer = optimizer, epochs = EPOCHS, epoch = epoch)
    
    print(f'''
          \nFor epoch [{epoch}/{EPOCHS}]
            Train Loss: {train_loss}
            Validation Loss: {val_loss}
            Validation Accuracy: {val_accuracy}%
            \n
          ''')
    
    save_model(checkpoint=checkpoint, save_path=f'data/saves/model/BCE_Loss_{epoch}_epochs.pth.tar')
    
# %%