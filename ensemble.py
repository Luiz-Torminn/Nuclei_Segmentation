#%%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TTF

#%%
class AtrousConvs(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, dilation:int) -> None:
        super(AtrousConvs,  self).__init__()
        
        self.dilation_conv = nn.Sequential([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation = dilation,  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ])
    
    def forward(self, x):
        return self.dilation_conv(x)
    
class AtrousEncoder(nn.Module):
    def __init__(self, in_channels:int = 1, out_channels:int = 1, dilation:list[int] = [1, 6, 12, 18]) -> None:
        super(AtrousEncoder).__init__()
        self.in_channels = in_channels
        self.outputs = []
        
        self.conv1 = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        #Encoder unit layers:
        for d in dilation:
            if d == 1:
                self.conv1.append(AtrousConvs(in_channels, out_channels, kernel_size=1, dilation=d))
                
            else:
                self.conv1.append(AtrousConvs(in_channels, out_channels, kernel_size=3, dilation=d))
                
    def forward(self,x):
        [self.outputs.append(i(x)) for i in self.conv1()]
        
        concat = torch.cat(self.outputs, dim=1)
            
            
        
        
            
            
        
    
