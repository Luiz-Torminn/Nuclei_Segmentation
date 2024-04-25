#%%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TTF

#%%
class Conv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int = 1, dilation:int = 1) -> None:
        super(Conv2D, self).__init__()
        
        self.conv = nn.Sequential([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation = dilation,  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ])
    
    def forward(self, x):
        return self.conv(x)
    
class AtrousEncoder(nn.Module):
    def __init__(self, in_channels:int = 1, out_channels:int = 256, dilation:list[int] = [1, 6, 12, 18]) -> None:
        super(AtrousEncoder, self).__init__()
        self.outputs = []
        self.in_channels = in_channels
        
        self.conv1 = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 1x1 Convulational layer
        self.xconv = Conv2D(in_channels = (out_channels * 4), out_channels = 256)
        
        # Upsample layer
        self.upsample = nn.ConvTranspose2d(in_channels = out_channels, out_channels = 48)
        
        #Encoder unit layers:
        for d in dilation:
            if d == 1:
                self.conv1.append(Conv2D(in_channels, out_channels, kernel_size=1, dilation=d))
                
            else:
                self.conv1.append(Conv2D(in_channels, out_channels, kernel_size=3, dilation=d))
                
    def forward(self,x):
        [self.outputs.append(self.conv[i](x)) for i in range(len(self.conv1()))]
        
        concat = torch.cat(self.outputs, dim=1)
        
        x = self.pool(concat)
        x = self.xconv(x)
        x = self.upsample(x)
        
        return x
        
class Decoder(nn.Module):
    def __init__(self, in_channels:int = 1, out_channels:int = 256, output_classes = 1) -> None:
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        
        self.decoder_output = AtrousEncoder(in_channels = in_channels, out_channels = out_channels)
        
        # First 1x1 conv
        self.first_conv = Conv2D(in_channels = in_channels, out_channels = 48)
        
        # Following covolutions
        self.decoder_conv = nn.Sequential([
            Conv2D(in_channels = 48*2, out_channels = out_channels, kernel_size = 3),
            Conv2D(in_channels = out_channels, out_channels = out_channels),
        ])
        
        # Decoder upsample
        self.decoder_up = self.decoder_conv = nn.Sequential([
            Conv2D(in_channels = out_channels, out_channels = 64),
            Conv2D(in_channels = 64, out_channels = output_classes),
        ])
        
    def forward(self, x):
        x = self.first_conv(x)
        x = torch.cat([x, self.decoder_output], dim = 1)
        x = self.decoder_conv(x)
        x = self.decoder_up(x)
        
        return x
        
            
            
        
        
            
            
        
    
