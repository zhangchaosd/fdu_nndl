'''
x = torch.rand((3, 3, 32, 64))
ret = torch.rand((3, 4))
'''

import torch
import torch.nn as nn

class HW82(nn.Module):
    def __init__(self):
        super(HW82, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = (4, 3))
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features = 29 * 62, out_features = 4)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)

if __name__ == '__main__':
    x = torch.rand((3, 3, 32, 64))
    model = HW82()
    ret = model(x)
    print('ret:', ret.shape)
    exit()