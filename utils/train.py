#%%
import torch

def train_model(dataloader, device:str, model, loss_function, optimizer, epochs:int, epoch:int) -> float:
    model.train()
    
    cumulative_loss = 0.0
    
    for i,(img, mask) in enumerate(dataloader):
        img, mask = img.to(device), mask.to(device)
        output = model(img.float())
        
        loss = loss_function(output, mask)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        cumulative_loss += loss.item()
        
        if i%5 == 0:
            print(f'For EPOCH[{epoch + 1}/{epochs}] - Training step [{i+1}/{len(dataloader)}] --> loss: {loss:.3f}')
    
    avg_loss = (cumulative_loss/len(dataloader))
    
    return avg_loss

def validate_model(dataloader, device:str, model, loss_function) -> tuple[float]:
    model.eval()
    
    cumulative_loss = 0.0
    total_pxl = 0
    correct_pxl = 0   
    dice_score = 0.0
    
    with torch.no_grad():
        for image, mask in dataloader:
            image, mask = image.to(device), mask.to(device)

            output = torch.sigmoid(model(image.float()))

            # Loss
            loss = loss_function(output, mask)
            cumulative_loss += loss.item()

            # Accuracy
            prediction = (output > 0.5).float()
            total_pxl += mask.shape[0] * mask.shape[2] * mask.shape[3] 
            correct_pxl += (prediction == mask).sum()
            
            # Dice Score
            dice_score += (2 * (prediction * mask).sum()) / (total_pxl * 2)
        
    avg_loss = cumulative_loss/len(dataloader)
    avg_accuracy = (correct_pxl/total_pxl)*100
    dice = dice_score/len(dataloader)
        
    return  (avg_loss, avg_accuracy, dice)

#%%
# from model import Unet
# import torch
# from torch.utils.data import DataLoader
# from data_loader import Nuclei_Loader


# DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
# model = Unet(3).to(DEVICE)
# loss = torch.nn.BCELoss()

# data = Nuclei_Loader('data/dataset/val')
# val_data = DataLoader(data, batch_size=3, shuffle=True)

# val_loss, val_acc, dice = validate_model(val_data, DEVICE, model, loss)
# print(val_acc)

# %%
