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
    
    for  image, mask in dataloader:
        image, mask = image.to(device), mask.to(device)
        
        output = model(image.float())
        
        # Loss
        loss = loss_function(output, mask)
        cumulative_loss += loss.item()
        
        # Accuracy
        _, prediction = torch.max(output, 1)
        total_pxl = mask.shape[0] * mask.shape[1]
        correct_pxl += (mask == prediction).sum()
        
    avg_loss = cumulative_loss/len(dataloader)
    avg_accuracy = correct_pxl/total_pxl
        
    return  (avg_loss, avg_accuracy)
        
        
        