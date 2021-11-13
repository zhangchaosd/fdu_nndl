'''
x=torch.rand((3,1,128,32))
ret = torch.tensor([[0.1706, 0.1674, 0.1637, 0.1687, 0.1632, 0.1664],
        [0.1711, 0.1691, 0.1655, 0.1685, 0.1608, 0.1649],
        [0.1714, 0.1672, 0.1648, 0.1692, 0.1580, 0.1693]])
'''



import torch
import torch.nn as nn

class HW87(nn.Module):
    def __init__(self):
        super(HW87, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#64*16
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layers2 = nn.Sequential(
            nn.Linear(in_features = 16*32*8, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500,out_features=6),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        ret = self.layers(x)
        ret = ret.reshape(ret.shape[0],-1)
        return self.layers2(ret)

if __name__ == '__main__':
    x=torch.rand((3,1,128,32))
    print(x.shape)
    print(x)
    model = HW87()
    ret = model(x)
    print(ret)
    exit()