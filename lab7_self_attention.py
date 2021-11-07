#https://a1jkiq3cpx.feishu.cn/docs/doccn0mgaZlm83OOA8mfZqQ4FUh



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
x = torch.Tensor([[[0.6840, 0.1093],[1.7795, 1.0825],[0.6840, 0.1093],[1.7795, 1.0825]],[[1.0749, 2.3809],[0.2621, 1.0084],[1.0749, 2.3809],[0.2621, 1.0084]],[[0.3862, 0.4335],[0.4832, 0.5558],         [0.3862, 0.4335],         [0.4832, 0.5558]]])
ret = torch.tensor([[[-0.0780,  0.2123],
         [-0.0581,  0.2180],
         [-0.0648,  0.2167]],

        [[-0.1318,  0.2591],
         [-0.1155,  0.2653],
         [-0.1184,  0.2654]],

        [[-0.1493,  0.2881],
         [-0.1357,  0.2944],
         [-0.1370,  0.2945]],

        [[-0.1533,  0.3075],
         [-0.1423,  0.3132],
         [-0.1430,  0.3127]]])
'''
import torch
import torch.nn as nn
class Net1(nn.Module):
    def __init__(self, feature_dim = 2, embedding_dim = 8, hidden_dim = 16, n_layer = 3, out_size = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_size = out_size
        self.encoder = self.Encoder(feature_dim, embedding_dim, hidden_dim, n_layer)
        self.decoder = self.Decoder(feature_dim, embedding_dim, hidden_dim, n_layer)

    def forward(self, x):
        _, hidden = self.encoder(x)
        ret = torch.zeros((self.out_size, x.shape[0], x.shape[2]))
        x = torch.transpose(x, 0, 1)
        ret[0], hidden = self.decoder(x[-1], hidden)
        
        for i in range(self.out_size - 1):
            ret[i + 1], hidden = self.decoder(ret[i], hidden)
        return ret

    class Encoder(nn.Module):
        def __init__(self, input_dim = 2, embedding_dim = 8, hidden_dim = 16, n_layers = 3, att_dim = 8) -> None:
            super().__init__()
            self.fc = nn.Linear(input_dim, embedding_dim)
            self.fcq = nn.Linear(embedding_dim,att_dim)
            self.fck = nn.Linear(embedding_dim,att_dim)
            self.fcv = nn.Linear(embedding_dim,att_dim)

            self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = n_layers, batch_first = True)

        def forward(self, x):
            x = self.fc(x)
            q = torch.transpose(self.fcq(x),1,2)
            k = torch.transpose(self.fck(x),1,2)
            v = torch.transpose(self.fcv(x),1,2)
            a = torch.bmm(torch.transpose(k,1,2), q)
            a_ = nn.functional.softmax(a, dim = 0)
            o = torch.bmm(v, a_)
            o = torch.transpose(o,1,2)
            return self.gru(o)

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
    x = torch.Tensor([[[0.6840, 0.1093],[1.7795, 1.0825],[0.6840, 0.1093],[1.7795, 1.0825]],[[1.0749, 2.3809],[0.2621, 1.0084],[1.0749, 2.3809],[0.2621, 1.0084]],[[0.3862, 0.4335],[0.4832, 0.5558],         [0.3862, 0.4335],         [0.4832, 0.5558]]])
    
    #print(x)
    net2 = Net1(feature_dim = x.shape[2])
    #print(x.shape) # 4,3,2  L,N,H
    ret = net2(x)
    #(3,3,2)
    #ret = torch.Tensor([[[0.0818, 0.1766],[0.0555, 0.1432],[0.0663, 0.1533]], [[0.1013, 0.1704],[0.0852, 0.1404],[0.0932, 0.1542]], [[0.1215, 0.1684],[0.1122, 0.1458],[0.1178, 0.1591]]])
    #print(x.shape)
    #print(ret.shape)
    print(ret)
    print('ret')