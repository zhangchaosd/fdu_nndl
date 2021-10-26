#https://a1jkiq3cpx.feishu.cn/docs/doccnQOKmZ3R31UMqL2DaPdHGim

'''
X = torch.tensor([[1,2,3],[4,5,6]])
Y = torch.tensor([[7,8,9],[7,5,3]])
ret = torch.tensor([[0.1000, 0.2000, 0.3000], [0.4000, 0.5000, 0.6000], [0.7000, 0.8000, 0.9000], [0.7000, 0.5000, 0.3000]])
'''
from typing import ForwardRef
import torch
def zy1(X, Y):
    return torch.cat((X, Y), dim=0) / 10



'''
X = torch.tensor([[10,0.2,0.003],[4,5,6]])
Y = torch.tensor([[70,8,0.009],[0.7,0.0005,0.03]])
ret = torch.tensor([[0., 1., 1.], [1., 1., 1.]])
'''
import torch
def zy2(X, Y):
    C = X + Y
    D = X * Y
    ret = C - D
    ret[ret > 0] = 1
    ret[ret < 0] = 0
    return ret

'''
X = torch.tensor([1,2,3])
Y = torch.tensor([[7,8,9],[7,5,3]])
ret = torch.tensor([10.3923,  6.7082])
'''
import torch
def zy3(X, Y):
    out = (Y - X) ** 2
    out2 = torch.sum(out, dim = 1)
    return torch.sqrt(out2)


'''
x = torch.Tensor([[[0.6840, 0.1093],[1.0749, 2.3809],[0.3862, 0.4335]], [[1.7795, 1.0825],[0.2621, 1.0084],[0.4832, 0.5558]]])
ret = torch.Tensor([[[0.0818, 0.1766],[0.0555, 0.1432],[0.0663, 0.1533]], [[0.1013, 0.1704],[0.0852, 0.1404],[0.0932, 0.1542]], [[0.1215, 0.1684],[0.1122, 0.1458],[0.1178, 0.1591]]])
'''
import torch
import torch.nn as nn
'''
这个神经网络包括一个编码器模块、一个解码器模块：
- 编码器模块包含一个嵌入层、一个门控循环单元：
    1）嵌入层由全连接实现，输入维度为input_size(feature_dim)，默认为2， 输出维度为encoder_embedding_dim，默认为8；
    2）门控循环单元的输入维度为embedding_dim，默认为8， 隐藏层维度为hidden_dim，默认为16，层数为n_layers，默认为2；
- 解码器模块包含一个嵌入层、一个门控循环单元以及一个全连接层：
    1）嵌入层由全连接实现，输入维度为input_size(feature_dim)，默认为2， 输出维度为decoder_embedding_dim，默认为8；
    2）门控循环单元的输入维度为embedding_dim，默认为8， 隐藏层维度为hidden_dim，默认为16，层数为n_layers，默认为2；
输入一个三维张量x（其形状为(sequence_len, batch_size, feature_dim))，x输入编码器，经过嵌入层得到x的嵌入向量embedded,再输入进门控循环单元得到输出out以及隐状态hidden
输出为三维张量y(其形状为(target_len(默认为3)，batch_size, feature_dim))，按序通过解码器解码得到y；其中，解码器的第一个输入y为x的最后一个序列，第一个输入隐状态为编码器输出隐状态；
经过嵌入层得到y的嵌入向量embedded，再输入进门控循环单元得到输出y1以及隐状态hidden,y1经过全连接层得到out，输出out以及hidden；以预测的输出out以及hidden作为下一次门控循环单元的输入；
并得到最终长度为target_len的输出序列y;
'''
class EncoderDecoder(nn.Module):
    def __init__(self, feature_dim = 2,  encoder_embedding_dim = 8, decoder_embedding_dim = 8, hidden_dim = 16,  n_layers = 2, target_len = 3):
        super().__init__()
        self.target_len = target_len
        self.out_dim = feature_dim
        self.encoder = self.EncoderGRU(input_size= feature_dim, embedding_dim= encoder_embedding_dim, hidden_dim = hidden_dim, n_layers= n_layers)
        self.decoder = self.DecoderGRU(output_size= feature_dim, embedding_dim=decoder_embedding_dim, hidden_dim = hidden_dim, n_layers= n_layers)


    def forward(self, x):
        batch_size = x.shape[1]
        out = torch.zeros((self.target_len, batch_size, self.out_dim))
        _ , hidden = self.encoder(x)
        de_in = x[-1, :]
        for i in range(self.target_len):
            de_out, hidden = self.decoder(de_in, hidden)
            out[i] = de_out
            de_in = de_out # using predictions as the next input
        return out
    
    class EncoderGRU(nn.Module):
        def __init__(self, input_size = 10, embedding_dim = 8, hidden_dim = 16, n_layers = 2):
            super().__init__()
            self.embedding = nn.Sequential(nn.Linear(input_size, embedding_dim))
            self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = n_layers)
            
        def forward(self, x):
            embedded = self.embedding(x)
            out, hidden = self.gru(embedded)
            return out, hidden
    
    class DecoderGRU(nn.Module):
        def __init__(self, output_size, embedding_dim, hidden_dim, n_layers = 1):
            super( ).__init__()
            self.embedding = nn.Sequential(nn.Linear(output_size, embedding_dim))
            self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim,  num_layers = n_layers)
            self.fc = nn.Linear(hidden_dim, output_size)
        
        def forward(self, y, hidden):
            y = y.unsqueeze(0)
            embedded =self.embedding(y)
            y1, hidden = self.gru(embedded, hidden)
            out = self.fc(y1.squeeze(0))
            return out, hidden

'''
这个神经网络包含一个循环神经网络层和一个全连接层:
- 循环神经网络层：
    门控循环单元输入维度为 input_size(feature_dim)，默认为2， 隐藏层维度为 hidden_dim，默认为8，层数为 n_layers，默认为3；
- 全连接层：
    输入维度为 hidden_dim，输出维度 out_dim 默认为4。
输入一个三维张量x（其形状为(sequence_len, batch_size, feature_dim))，x输入进门控循环单元得到输出 ys 以及隐状态 hidden；
在 ys 的第一维上求平均值，得到 avr；
将 avr 输入全连接层，得到 ret。

This neural network consists of a cyclic neural network layer and a full-connection layer:
- CNN layer:
The input dimension of the gated loop unit is input_size(Feature_DIM), which defaults to 2. The hidden layer dimension is hidden_DIM, which defaults to 8. The number of layers is N_layers, which defaults to 3.
- Full-connection layer:
The input dimension is hidden_dim, and the output dimension out_dim defaults to 4.
Input a three-dimension tensor X (its shape is (sequence_len, batch_size, feature_dim)), x is input into the GRU to obtain the output ys and hidden;
avr is obtained by averaging the first dimension of YS.
avr is input into the full connection layer to obtain ret.
'''

'''
x = torch.Tensor([[[0.6840, 0.1093],[1.0749, 2.3809],[0.3862, 0.4335]], [[1.7795, 1.0825],[0.2621, 1.0084],[0.4832, 0.5558]]])
ret = torch.tensor([[-0.0451, -0.2391, -0.2339, 0.4280], [-0.0340, -0.2333, -0.2181, 0.4198], [-0.0396, -0.2363, -0.2301, 0.4245]])
'''
import torch
import torch.nn as nn
class Net1(nn.Module):
    def __init__(self, feature_dim = 2, hidden_dim = 8, n_layer = 3, out_dim = 4):
        super().__init__()
        self.gru = nn.GRU(input_size = feature_dim, hidden_size = hidden_dim, num_layers = n_layer)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        ys, hidden = self.gru(x)
        avr = torch.mean(ys, dim = 0)
        ret = self.fc(avr)
        return ret



'''
这个神经网络包括一个编码器模块、一个解码器模块：
- 编码器模块包含一个嵌入层、一个门控循环单元：
    1）嵌入层由全连接实现，输入维度为 input_size(feature_dim)，默认为 2， 输出维度为 encoder_embedding_dim，默认为 8；
    2）门控循环单元的输入维度为 embedding_dim，默认为 8， 隐藏层维度为 hidden_dim，默认为 16，层数为 n_layers，默认为 3；
- 解码器模块包含一个嵌入层、一个门控循环单元以及一个全连接层：
    1）嵌入层由全连接实现，输入维度为input_size(feature_dim)，默认为 2， 输出维度为decoder_embedding_dim，默认为 8；
    2）门控循环单元的输入维度为embedding_dim，默认为 8， 隐藏层维度为hidden_dim，默认为 16，层数为n_layers，默认为 2；
输入一个三维张量x（其形状为(sequence_len, batch_size, feature_dim))，x输入编码器，经过嵌入层得到x的嵌入向量embedded,再输入进门控循环单元得到输出out以及隐状态hidden
输出为三维张量y(其形状为(target_len(默认为4)，batch_size, feature_dim))，按序通过解码器解码得到y；其中，解码器的第一个输入y为x的最后一个序列，第一个输入隐状态为编码器输出隐状态；
经过嵌入层得到y的嵌入向量embedded，再输入进门控循环单元得到输出y1以及隐状态hidden,y1经过全连接层得到out，输出out以及hidden；以预测的输出out以及hidden作为下一次门控循环单元的输入；
并得到最终长度为target_len的输出序列y;
The neural network consists of an encoder module and a decoder module:
- Encoder module consists of an embedded layer and a gru:
1) The embedding layer is fully connected, with an input dimension of input_size(feature_DIM), which defaults to 2, and an output dimension of encoder_embedding_DIM, which defaults to 8;
2) The input dimension of the gru is embedding_dim, which defaults to 8, the hidden layer dimension is hidden_dim, which defaults to 16, and the number of layers is n_layers, which defaults to 3;
- The decoder module contains an embed layer, a gated loop unit, and a full connection layer:
1) The embedding layer is fully connected, with input dimension input_size(feature_DIM), which defaults to 2, and output dimension decoder_embedding_DIM, which defaults to 8;
2) The input dimension of the gated loop unit is embedding_DIM, which defaults to 8, the hidden layer dimension is hidden_DIM, which defaults to 16, and the number of layers is n_layers, which defaults to 2;
Input a 3D tensor X (its shape is (sequence_len, batch_size, Feature_DIM)), input X into the encoder, get the embedded vector embedded of X through the embedded layer, and then input the door control loop unit to get the output OUT and hidden state hidden
The output is the 3d tensor Y (whose shape is (target_len(default: 4), batch_size, feature_DIM)), which is decoded by the decoder in sequence to obtain Y; Where, the first input Y of the decoder is the last sequence of X, and the first input implicit state is the encoder output implicit state;
The embedded vector embedded of Y is obtained through the embedded layer, and then input into the door control loop unit to obtain the output Y1 and hidden state. Y1 is obtained through the fully connected layer, and output OUT and hidden. Take the predicted output OUT and hidden as the input of the next gated loop unit;
The output sequence Y with the final length of target_len is obtained.
'''
'''
x = torch.Tensor([[[0.6840, 0.1093],[1.0749, 2.3809],[0.3862, 0.4335]], [[1.7795, 1.0825],[0.2621, 1.0084],[0.4832, 0.5558]], [[0.6840, 0.1093],[1.0749, 2.3809],[0.3862, 0.4335]], [[1.7795, 1.0825],[0.2621, 1.0084],[0.4832, 0.5558]]])
ret = torch.tensor([[[-0.1406, -0.0360], [-0.1417, -0.0322], [-0.1477, -0.0293]], [[-0.1290, -0.0580], [-0.1404, -0.0483], [-0.1416, -0.0473]], [[-0.1343, -0.0565], [-0.1498, -0.0457], [-0.1490, -0.0459]], [[-0.1456, -0.0497], [-0.1612, -0.0401], [-0.1597, -0.0407]]])
'''
import torch
import torch.nn as nn
class Net2(nn.Module):
    def __init__(self, feature_dim = 2, embedding_dim = 8, hidden_dim = 16, n_layer = 3, out_size = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_size = out_size
        self.encoder = self.Encoder(feature_dim,embedding_dim, hidden_dim, n_layer)
        self.decoder = self.Decoder(feature_dim, embedding_dim, hidden_dim, n_layer)

    def forward(self, x):
        _, hidden = self.encoder(x)
        ret = torch.zeros((self.out_size, x.shape[1], x.shape[2]))
        ret[0], hidden = self.decoder(x[-1], hidden)
        for i in range(self.out_size - 1):
            ret[i + 1], hidden = self.decoder(ret[i], hidden)
        return ret

    class Encoder(nn.Module):
        def __init__(self, input_dim = 2, embedding_dim = 8, hidden_dim = 16, n_layers = 3) -> None:
            super().__init__()
            self.fc = nn.Linear(input_dim, embedding_dim)
            self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = n_layers)

        def forward(self, x):
            x = self.fc(x)
            return self.gru(x)

    class Decoder(nn.Module):
        def __init__(self, output_dim = 2, embedding_dim = 8, hidden_dim = 16, n_layers = 3) -> None:
            super().__init__()
            self.fc1 = nn.Linear(output_dim, embedding_dim)
            self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = n_layers)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, hidden): #(batch_size, D)
            out = self.fc1(x)
            out = out.unsqueeze(0)
            y, hidden = self.gru(out, hidden)
            y = y.squeeze(0)
            ret = self.fc2(y)
            return ret, hidden

if __name__ == '__main__':
    #(2,3,2)
    x = torch.Tensor([[[0.6840, 0.1093],[1.0749, 2.3809],[0.3862, 0.4335]], [[1.7795, 1.0825],[0.2621, 1.0084],[0.4832, 0.5558]], [[0.6840, 0.1093],[1.0749, 2.3809],[0.3862, 0.4335]], [[1.7795, 1.0825],[0.2621, 1.0084],[0.4832, 0.5558]]])
    net2 = Net2(feature_dim = x.shape[2])
    ret = net2(x)
    #(3,3,2)
    #ret = torch.Tensor([[[0.0818, 0.1766],[0.0555, 0.1432],[0.0663, 0.1533]], [[0.1013, 0.1704],[0.0852, 0.1404],[0.0932, 0.1542]], [[0.1215, 0.1684],[0.1122, 0.1458],[0.1178, 0.1591]]])
    #print(x.shape)
    print(ret.shape)
    print(ret)