'''
x = torch.rand((3, 150, 150))
ret = torch.rand((3, 5))
'''

import torch
import torch.nn as nn

class HW83(nn.Module):
    def __init__(self):
        super(HW83, self).__init__()
        self.cnn = nn.Sequential(
            nn.LSTM(input_size = 150, hidden_size = 150, batch_first = True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features = 150, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 32),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 32, out_features = 16),
            nn.ReLU(),
            nn.Linear(in_features = 16, out_features = 8),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features = 150 * 8, out_features = 1024),
            nn.ReLU(),
            nn.Linear(in_features = 1024, out_features = 5),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        out, _ = self.cnn(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc3(out)
        return out

if __name__ == '__main__':
    x = torch.rand((3, 150, 150))
    model = HW83()
    ret = model(x)
    print('ret:', ret.shape)
    exit()