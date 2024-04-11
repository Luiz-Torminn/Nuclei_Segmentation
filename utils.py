# Save and Load model and Plot Graphs
#%%
import torch

#%%
def save_model(checkpoint:dict, save_path:str) -> None:
    torch.save(checkpoint, save_path)

#%%
def load_model(model, save_path:str, optimizer:bool = False) -> None:
    states = torch.load(save_path)
    
    model.load_state_dict(states['state_dict'], strict = False)
    
    if optimizer:
        model.load_state_dict(states['optimizer'], strict = False)