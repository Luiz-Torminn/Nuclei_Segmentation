#%%
from torch import nn
import torch.nn.functional as F

# %%
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(self, Conv2D).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)  
        )
        
class Unet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        
        
        
        
    