"""
x = torch.Tensor([[0.3065, 0.1160, 0.4094, 0.0920, 0.2748, 0.3449, 0.4496, 0.7667, 0.0889, 0.3903, 0.0702, 0.4274, 0.7400, 0.1801, 0.8190, 0.5982], [0.2650, 0.2428, 0.3423, 0.4266, 0.1475, 0.5357, 0.4137, 0.8574, 0.8066, 0.2703, 0.5406, 0.0034, 0.0951, 0.0633, 0.7144, 0.0834], [0.4511, 0.0712, 0.6195, 0.5863, 0.8904, 0.3637, 0.4541, 0.2128, 0.4295, 0.5676, 0.1737, 0.6271, 0.1006, 0.2404, 0.6204, 0.6794]])
ret = torch.Tensor([[0.2452, 0.2982, 0.2319, 0.2246], [0.2345, 0.2995, 0.2378, 0.2283], [0.2472, 0.2931, 0.2442, 0.2155]])
"""
import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax

'''
这个2层神经网络包含2个全连接层； 第一个全连接层输入维度是16，输出维度是 mid_dim，默认大小为32，使用 LeakyReLU 激活函数； 第二个全连接层输入维度是 mid_dim，输出维度是 4，使用 Softmax 激活函数； 输入 x 经过两个全连接层后得到输出 ret；
This two-layer neural network contains two full-connection layers. For the first full-connection layer, the input dimension is 16 and the output dimension is mid_dim, with a default size of 32, using the LeakyReLU activation function. The second full-connection layer input dimension is mid_dim, output dimension is 4, using Softmax activation function; Input X passes through two fully connected layers to obtain output ret;
'''
class LinearModel1(nn.Module):
    def __init__(self, mid_dim = 32):
        super(LinearModel1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, mid_dim),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(mid_dim, 4),
            nn.Softmax(dim = -1)
        )

    def forward(self, x):
        ret = self.layers(x)
        return ret

'''
这个5层神经网络包含5个全连接层；
第一个全连接层输入维度是16，输出维度是 mid_dim，默认大小为32，使用 LeakyReLU 激活函数；
第二三四个全连接层输入维度和输出维度全部是 mid_dim，使用LeakyReLU 激活函数；
最后一层全连接层输入维度为 mid_dim，输出维度为 4；

This 5-layer neural network contains 5 full connection layers.
For the first full-connection layer, the input dimension is 16 and the output dimension is mid_dim, with a default size of 32, using the LeakyReLU activation function.
For the second, third, and fourth full-connection layers, the input and output dimensions are all mid_dim, using the LeakyReLU activation function.
The input dimension of the last full-connection layer is mid_dim and the output dimension is 4.
Input X passes through five full-connection layers to get the output ret;

'''
"""
x = torch.Tensor([[0.3065, 0.1160, 0.4094, 0.0920, 0.2748, 0.3449, 0.4496, 0.7667, 0.0889, 0.3903, 0.0702, 0.4274, 0.7400, 0.1801, 0.8190, 0.5982], [0.2650, 0.2428, 0.3423, 0.4266, 0.1475, 0.5357, 0.4137, 0.8574, 0.8066, 0.2703, 0.5406, 0.0034, 0.0951, 0.0633, 0.7144, 0.0834], [0.4511, 0.0712, 0.6195, 0.5863, 0.8904, 0.3637, 0.4541, 0.2128, 0.4295, 0.5676, 0.1737, 0.6271, 0.1006, 0.2404, 0.6204, 0.6794]])
ret = torch.Tensor([[0.2731, 0.2590, 0.2407, 0.2272], [0.2652, 0.2666, 0.2504, 0.2177], [0.2726, 0.2555, 0.2343, 0.2376]])
"""
import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax

class LinearModel2(nn.Module):
    def __init__(self, mid_dim = 32):
        super(LinearModel2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, mid_dim),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(mid_dim, 4),
            nn.Softmax(dim = -1)
        )

    def forward(self, x):
        ret = self.layers(x)
        return ret

'''
这个5层神经网络包含7个全连接层；
第一个全连接层输入维度是16，输出维度是 hid_dim1，使用 LeakyReLU 激活函数；
第二层全连接层输入维度是 hid_dim1，输出维度是 hid_dim2，使用 LeakyReLU 激活函数；
第三层全连接层输入维度是 hid_dim2，输出维度是 hid_dim2，使用 LeakyReLU 激活函数；
第四层全连接层输入维度是 hid_dim2 + hid_dim2，输出维度是 hid_dim3，使用 LeakyReLU 激活函数；
第五层全连接层输入维度是 hid_dim3，输出维度是 4，使用 Softmax 激活函数；
输入 X 经过第一个全连接层得到 out1, 将 out1 分别输入到并联的两组全连接层，经过第二层和第三层神经网络之后得到 out2_1 和 out2_2，将 out2_1 和 out2_2 在最后一维拼接得到 out2；
让 out2 经过第四层和第五层神经网络得到 ret；

This 5-layer neural network contains 7 full-connection layers.
For the first full-connection layer, the input dimension is 16 and the output dimension is hid_dim1, using the LeakyReLU activation function.
For the second full-connection layer, the input dimension is hid_dim1 and the output dimension is hid_dim2. Run the LeakyReLU activation function.
For the third full-connection layer, the input dimension is hid_dim2 and the output dimension is hid_dim2. Use the LeakyReLU activation function.
For the fourth full-connection layer, the input dimension is hid_dim2 + hid_dim2, and the output dimension is hid_dim3. Use the LeakyReLU activation function.
The input dimension of the fifth full-connection layer is hid_dim3, and the output dimension is 4. Softmax activation function is used.
Input X goes through the first fully-connected layer to obtain out1, and out1 is input to two groups of fully connected layers in parallel respectively. out2_1 and out2_2 are obtained after the second and third layer neural networks, and out2 is obtained by splicing out2_1 and out2_2 in the last one dimension.
Let out2 pass through the fourth and fifth layer neural network to get ret.

'''
"""
x = torch.Tensor([[0.3065, 0.1160, 0.4094, 0.0920, 0.2748, 0.3449, 0.4496, 0.7667, 0.0889, 0.3903, 0.0702, 0.4274, 0.7400, 0.1801, 0.8190, 0.5982], [0.2650, 0.2428, 0.3423, 0.4266, 0.1475, 0.5357, 0.4137, 0.8574, 0.8066, 0.2703, 0.5406, 0.0034, 0.0951, 0.0633, 0.7144, 0.0834], [0.4511, 0.0712, 0.6195, 0.5863, 0.8904, 0.3637, 0.4541, 0.2128, 0.4295, 0.5676, 0.1737, 0.6271, 0.1006, 0.2404, 0.6204, 0.6794]])
ret = torch.Tensor([[0.2486, 0.2661, 0.2403, 0.2449], [0.2482, 0.2641, 0.2420, 0.2457], [0.2478, 0.2658, 0.2413, 0.2451]])
"""
import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax

class LinearModel3(nn.Module):
    def __init__(self, hid_dim1 = 32, hid_dim2 = 64, hid_dim3 = 128):
        super(LinearModel3, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(16, hid_dim1),
            nn.LeakyReLU(negative_slope = 0.1)
        )
        self.fc2_1 = nn.Sequential(
            nn.Linear(hid_dim1, hid_dim2),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(hid_dim2, hid_dim2),
            nn.LeakyReLU(negative_slope = 0.1)
        )
        self.fc2_2 = nn.Sequential(
            nn.Linear(hid_dim1, hid_dim2),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(hid_dim2, hid_dim2),
            nn.LeakyReLU(negative_slope = 0.1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hid_dim2 + hid_dim2, hid_dim3),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(hid_dim3, 4),
            nn.Softmax(dim = -1)
        )

    def forward(self, x):
        out1 = self.fc1(x)
        out2_1 = self.fc2_1(out1)
        out2_2 = self.fc2_2(out1)
        out2 = torch.cat((out2_1, out2_2), dim = -1)
        ret = self.fc3(out2)
        return ret




if __name__ == '__main__':
    print('h')
    model = LinearModel3()
    x = torch.Tensor([[0.3065, 0.1160, 0.4094, 0.0920, 0.2748, 0.3449, 0.4496, 0.7667, 0.0889, 0.3903, 0.0702, 0.4274, 0.7400, 0.1801, 0.8190, 0.5982], [0.2650, 0.2428, 0.3423, 0.4266, 0.1475, 0.5357, 0.4137, 0.8574, 0.8066, 0.2703, 0.5406, 0.0034, 0.0951, 0.0633, 0.7144, 0.0834], [0.4511, 0.0712, 0.6195, 0.5863, 0.8904, 0.3637, 0.4541, 0.2128, 0.4295, 0.5676, 0.1737, 0.6271, 0.1006, 0.2404, 0.6204, 0.6794]])
    #print(x)
    ret = model.forward(x)
    print(ret)