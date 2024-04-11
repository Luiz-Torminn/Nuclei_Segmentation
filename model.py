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
    
    def forward(self, x):
        return self.conv(x)
    
# %%       
class Unet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, features = [64,128,256,512]) -> None:
        super(self, Unet).__init__()
        
        self.down = nn.ModuleList()
        self.upward = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        
    # Down part of U-Net
        
        
        
        
    