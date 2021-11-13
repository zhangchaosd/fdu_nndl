'''
x = torch.rand((3, 64, 150))
ret = torch.rand((3, 64, 3))
'''

import torch
import torch.nn as nn

class HW84(nn.Module):
    def __init__(self):
        super(HW84, self).__init__()
        self.fc1 = nn.Linear(in_features=150, out_features=64)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 64, out_features = 3),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        return self.fc2(self.fc1(x))

if __name__ == '__main__':
    x = torch.rand((3, 64, 150))
    model = HW84()
    ret = model(x)
    print('ret:', ret.shape)
    exit()