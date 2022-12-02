import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,kernel_size,in_channel,out_channel):
        super(Block, self).__init__()
        self.kernel_size=kernel_size
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.Conv1=nn.Conv2d(kernel_size=kernel_size,in_channels=in_channel,out_channels=out_channel//2,stride=1,padding=kernel_size//2)
        self.BN=nn.BatchNorm2d(out_channel//2)
        self.pooling=nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv2=nn.Conv2d(kernel_size=1,in_channels=out_channel//2,out_channels=out_channel)

    def forward(self,x):
        x=self.Conv1(x)
        x=self.BN(x)
        x=self.pooling(x)
        x=self.Conv2(x)
        return x



class MultiBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(MultiBlock, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block1=Block(3,in_channel=in_channel,out_channel=out_channel//2)
        self.block2=Block(5,in_channel=in_channel,out_channel=out_channel//4)
        self.block3=Block(7,in_channel=in_channel,out_channel=out_channel//4)

    def forward(self,x):
        y=self.block1(x)
        y=torch.concat((self.block2(x),y),dim=1)
        y=torch.concat((self.block3(x),y),dim=1)
        return y


class multi_scale(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(multi_scale, self).__init__()
        self.net=nn.Sequential(
            MultiBlock(in_channel=in_channel, out_channel=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(kernel_size=5, in_channels=128, out_channels=64, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=out_channel, stride=1, padding=1),
            nn.BatchNorm2d(21),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=(3, 2), stride=(3, 2)),

        )
        self.softmax=nn.Softmax(dim=0)

    def forward(self,x):
        x=self.net(x)
        y=self.softmax(x)
        return y

if __name__=='__main__':
    a=torch.rand(1,1,48,32)
    net=multi_scale(1,21)
    print(net(a).shape)
    #net=Block(3,in_channel=1,out_channel=128)
    #net=Block(3,in_channel=1,out_channel=128)
    #print(net(a))
    #print()
