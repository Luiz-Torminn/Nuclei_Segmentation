#%%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TTF

# %%
class Conv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(Conv2D, self).__init__()
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
    def __init__(self, in_channels:int = 1, out_channels:int = 1, features = [64,128,256,512]) -> None:
        super(Unet, self).__init__()
        self.in_channels = in_channels
        
        self.downs = nn.ModuleList()
        self.upward = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        
        # Downward part
        for feature in features:
            self.downs.append(Conv2D(in_channels = self.in_channels, out_channels = feature))
            self.in_channels = feature
        
        # Upward part
        for feature in reversed(features):
            self.upward.append(
                nn.ConvTranspose2d(in_channels = feature * 2, out_channels = feature, kernel_size = 2, stride = 2), 
            )
            self.upward.append(Conv2D(in_channels = feature * 2, out_channels = feature))
        
        # Bottleneck
        self.bottleneck = Conv2D(in_channels = features[-1], out_channels = features[-1] * 2)
        
        # Last Conv:
        self.last_conv = nn.Conv2d(in_channels = features[0], out_channels = out_channels, kernel_size = 1, stride = 1)
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            
            skip_connections.append(x)
            
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        for i in range(0, len(self.upward), 2):
            x = self.upward[i](x)
            skip_connection_unit = list(reversed(skip_connections))[i//2]
            
            if x.shape != skip_connection_unit.shape:
                x = TTF.resize(x, size=(skip_connection_unit.shape[2:]))
                
            concat = torch.cat([x, skip_connection_unit],dim=1)
            x = self.upward[i + 1](concat)
        
        x = self.last_conv(x)
        
        return x
        
# %%
# def test():
#     rand_tensor = torch.rand((1,1,572,572))
#     model = Unet(1,1)
#     prediction = model(rand_tensor)
#     print(rand_tensor.shape)
#     print(prediction.shape)

# %%
# if __name__ == '__main__':
#     test()

# %%
