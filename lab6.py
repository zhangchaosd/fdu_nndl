"""
训练过程使用随机梯度下降作为优化算法，并在LSTM和全连接层之间加入了 dropout 层。输入的训练数据为若干形状为 (6,12) 的张量和 one-hot 表示的标签，测试数据为若干形状为 (6,12) 的张量，对测试数据输出包含 0/1 标签的向量。
本模型为使用单层LSTM和单层全连接层的二分类神经网络，LSTM的输入维度为12维，隐层维度为24维
dropout的drop率p为0.5，全连接层的输入为LSTM输出拼接成的6*24维向量，输出维度为2维

train_data = torch.randn((5,6,12))
train_label = torch.zeros((5,2))
test_data = torch.randn((3,6,12))
epoch_num = 4
ret = torch.randn((3,))
"""
import torch
import torch.nn as nn

def textclassificaton(train_data, train_label, test_data, epoch_num):
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.rnn = nn.Sequential(
                nn.LSTM(input_size=12, hidden_size=24, batch_first=True),
            )
            self.fnn = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Flatten(1, -1),
                nn.Linear(6*24,2),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            x, _ = self.rnn(x)
            output = self.fnn(x)
            return output

    model = Network()
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    for epoch in range(epoch_num):
        y_pred = model(train_data)
        loss = criterion(y_pred, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred = torch.argmax(model(test_data), -1)
    return y_pred

"""
训练过程使用随机梯度下降或AdamW或RMSprop作为优化算法，在GRU之前加入BN层，在GRU和全连接层之间加入了 dropout 层。输入的训练数据为若干形状为 (10,16) 的张量和 one-hot 表示的标签，测试数据为若干形状为 (10,16) 的张量，对测试数据输出包含 0/1 标签的向量。
本模型为使用单层GRU和单层全连接层的二分类神经网络，GRU的输入维度为16维，隐层维度为32维
dropout的drop率p为0.5，全连接层的输入为GRU输出拼接成的10*32维向量，输出维度为2维

The training process uses SGD or AdamW or RMSprop as optimization algorithms, a BN layer is inserted before the GRU and a dropout layer is inserted between the GRU and the fully-connected layer. The input training data is a number of tensors with shapes of (10,16) and labels represented by one-hot; the test data is a number of tensors with shapes of (10,16), and the test data is output with vectors containing 0/1 labels.
This model is a binary neural network using a single layer GRU and a single layer full connection layer. The input dimension of GRU is 16 dimensions and the hidden dimension is 32 dimensions
The drop rate of dropout P is 0.5. The input of the full connection layer is a 10*32 dimensional vector spliced by GRU outputs, and the output dimension is 2 dimensions
'''
'''
train_data = torch.randn((6,10,16))
train_label = torch.randn((6,2))
test_data = torch.randn((3,10,16))
epoch_num = 4
ret = torch.randn((3,))
"""
import torch
import torch.nn as nn

def textc(train_data, train_label, test_data, epoch_num, op='adamw'):
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.bn=nn.Sequential(
                nn.BatchNorm2d(num_features=10)
            )
            self.rnn = nn.Sequential(
                nn.GRU(input_size = 16, hidden_size = 32, num_layers = 3, batch_first=True)
            )
            self.fnn = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Flatten(1, -1),
                nn.Linear(10*32,2),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            x = x.unsqueeze(2)
            x = self.bn(x)
            x = x.squeeze(2)
            x, _ = self.rnn(x)
            output = self.fnn(x)
            return output

    model = Network()
    criterion = torch.nn.CrossEntropyLoss()#torch.nn.MSELoss(reduction="sum")
    if op == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif op=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.RMSprop(model.parameters())


    for epoch in range(epoch_num):
        y_pred = model(train_data)
        loss = criterion(y_pred, train_label)
        optimizer.zero_grad()
        print(loss)
        loss.backward()
        optimizer.step()

    y_pred = torch.argmax(model(test_data), -1)
    return y_pred

if __name__ == '__main__':
    if False:
        train_data = torch.randn((5,6,12))
        train_label = torch.zeros((5,2))
        test_data = torch.randn((3,6,12))
        epoch_num = 4
        ret = textclassificaton(train_data, train_label, test_data, epoch_num)
        print(ret)

    if True:
        train_data = torch.randn((6,10,16))
        train_label = torch.randn((6,2))
        for i in range(6):
            if train_label[i][0]>0:
                train_label[i][0]=1
                train_label[i][1]=0
            else:
                train_label[i][0]=0
                train_label[i][1]=1
        print(train_label)
        test_data = torch.randn((3,10,16))
        epoch_num = 40
        ret = textc(train_data, train_label, test_data, epoch_num, op='pros')
        print(ret)