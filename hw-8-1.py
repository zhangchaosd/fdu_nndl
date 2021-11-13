'''
x=torch.rand((3,1,128,32))
ret = torch.tensor([[0.1706, 0.1674, 0.1637, 0.1687, 0.1632, 0.1664],
        [0.1711, 0.1691, 0.1655, 0.1685, 0.1608, 0.1649],
        [0.1714, 0.1672, 0.1648, 0.1692, 0.1580, 0.1693]])
'''

'''
x=torch.rand((3,32,64))
ret = torch.rand((3, 64, 3))
'''

import torch
import torch.nn as nn

class HW81(nn.Module):
    def __init__(self):
        super(HW81, self).__init__()
        self.layer1 = nn.Sequential(
            nn.LSTM(input_size = 64, hidden_size = 32, num_layers = 1, batch_first = True, bidirectional = True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding=1),
            nn.LSTM(input_size = 64, hidden_size = 32, num_layers = 1, batch_first = True, bidirectional = True)
        )
        
        self.layer3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=1)
        )
        self.layers2 = nn.Sequential(
            nn.Linear(in_features=64,out_features=3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out, (h, c) = self.layer1(x)
        print('out: ', out.shape)
        print('h: ', h.shape)
        print('c: ',c.shape)
        out, (h, c) = self.layer2(out)
        print('done')
        ret = self.layer3(out)
        #ret = out.reshape(out.shape[0],-1)
        return self.layers2(ret)

if __name__ == '__main__':
    x=torch.rand((3,32,64))
    print(x.shape)
    model = HW81()
    ret = model(x)
    print('ret:', ret.shape)
    exit()