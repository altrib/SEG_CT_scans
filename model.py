import torch
import torch.nn as nn
# import torchvision.transforms.functional as tf

from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(
            self,in_channels=3,out_channels=1,features = [16, 32, 64, 128] # size reduced for cpu/gpu memory
            ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)


        # down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature
        
        # up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)
            )
            self.ups.append(DoubleConv(feature*2,feature))
        
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.final_conv = DoubleConv(features[0], out_channels)  # Map to classes



    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        x = self.final_conv(x)
        # x = x.view(x.size(0), -1)

        return x
             


def test():
    x = torch.randn(3,1,512,512).to("cuda")
    model = UNET(1,1).to("cuda")
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
    print(summary(model, (1, 512, 512),batch_size=2, device="cuda"))
    




if __name__=="__main__":
    test()
