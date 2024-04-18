#%%
import torch
import torch.utils
from torch.utils.data import DataLoader

from train import * 
from utils import *
from data_loader import *
from model import Unet
from graphs import plot_loss

#%%
# HYPERPARAMETERS
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_LOAD = True
LOSS_VALUES = {
    'Train Loss':[],
    'Validation Loss':[]
}

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
        print('\nNo save file was found...\n')
        
# %%
def main() -> None:
    for epoch in range(EPOCHS):
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        train_loss = train_model(train_loader, device = DEVICE, model = model, loss_function = loss, optimizer = optimizer, epochs = EPOCHS, epoch = epoch)
        val_loss, val_accuracy = validate_model(val_loader, device = DEVICE, model = model, loss_function = loss)
        
        LOSS_VALUES['Train Loss'].append(train_loss)
        LOSS_VALUES['Validation Loss'].append(val_loss)
        
        print(f'''
              \nFor epoch [{epoch}/{EPOCHS}]
                Train Loss: {train_loss:.4f}
                Validation Loss: {val_loss:.4f}
                Validation Accuracy: {val_accuracy}%
                \n
              ''')
        
        save_model(checkpoint=checkpoint, save_path=f'data/saves/model/BCE_Loss_{EPOCHS}_epochs.pth.tar')
        
    # %%
    plot_loss(LOSS_VALUES, f'data/saves/loss_graphs', EPOCHS)

# %%
if __name__ == '__main__':
    main()