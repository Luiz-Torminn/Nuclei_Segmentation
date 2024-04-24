#%%
import torch
import torch.utils
from torch.utils.data import DataLoader

from model import Unet
from utils.train import * 
from utils.utils import *
from data_loader import *
from utils.graphs import plot_loss
from utils.earlystopping import EarlyStopping

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
es = EarlyStopping(patience=5, min_loss=0.001, verbose=True)
 
# %%
def main() -> None:
    train_process = True
    
    while train_process:
        for epoch in range(EPOCHS):
            train_loss = train_model(train_loader, device = DEVICE, model = model, loss_function = loss, optimizer = optimizer, epochs = EPOCHS, epoch = epoch)
            val_loss, val_accuracy, dice = validate_model(val_loader, device = DEVICE, model = model, loss_function = loss)

            LOSS_VALUES['Train Loss'].append(train_loss)
            LOSS_VALUES['Validation Loss'].append(val_loss)
            
            train_process = es(model=model, current_loss=val_loss)

            print(f'''
                  \nFor epoch [{epoch + 1}/{EPOCHS}]:
                    Train Loss: {train_loss:.5f}
                    Validation Loss: {val_loss:.5f}
                    Validation Accuracy: {val_accuracy:.2f}%
                    Dice Score: {dice:.2f}
                    \n
                  ''')
            
            if not train_process:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                save_model(checkpoint=checkpoint, save_path=f'data/saves/model/BCE_Loss_{EPOCHS}_epochs.pth.tar')
                break
        
    # %%
    plot_loss(LOSS_VALUES, f'data/saves/loss_graphs', EPOCHS)

# %%
if __name__ == '__main__':
    main()