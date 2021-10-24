"""
pred = torch.Tensor([1, 2, 3, 4, 5])
label = torch.Tensor([3, 1, 0, 9, 5])
ret = torch.Tensor([39.])

实现平方损失函数；
输入为两个一维 Tensor。

Implement the squared loss function;
Input two one-dimensional Tensors.
"""

import torch
def square_loss(pred, label):
    loss = torch.sum(torch.pow(torch.sub(pred, label), 2))
    return loss.type(torch.float32)

'''
pred = torch.Tensor([1, 2, 3, 4, 5])
label = torch.Tensor([3, 1, 0, 9, 5])
ret = square_loss(pred, label)
print(ret)
print(ret.dtype)
'''


'''
pred = torch.Tensor([[1, 2, 3, 4, 5], [3, 1, 0, 9, 5]])
labels = torch.Tensor([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
ret = torch.Tensor([10.9729])

实现交叉熵损失函数；
输入两个相同大小的二维向量。

Implement the Cross-entropy loss function;
Input two Two-dimensional Tensors.
'''

import torch
def cross_entroy_loss(pred, labels):
    #N, C
    N = pred.shape[0]
    pred = torch.exp(pred)
    sums = torch.sum(pred, dim = 1, keepdim = True)
    pred = pred / sums
    inds = torch.argmax(labels, dim = 1, keepdim = True)
    loss = -torch.sum(torch.log(pred[:,inds]))
    loss = loss / N
    return loss#.type(torch.float32)
'''
pred = torch.Tensor([[1, 2, 3, 4, 5], [3, 1, 0, 9, 5]])
label = torch.Tensor([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
ret = cross_entroy_loss(pred, label)
print(ret)
print(ret.dtype)
'''

################################################

"""
1. 输入两个相同大小的一维张量X和Y
2. 使用最小二乘法计算参数w并返回

1. Input X and Y are one dimension Tensors with same size.
2. Compute the w by least squares method.


X = torch.Tensor([1, 2, 3, 4, 5])
Y = torch.Tensor([3, 5.01, 7, 8.94, 11.05])
ret = torch.Tensor([2.0030, 0.9910])
"""

def regression(X, Y):
    num_train = X.shape[0]
    X = X.reshape((num_train, 1))
    ones = torch.ones((num_train, 1))
    X = torch.cat([X, ones], dim = 1)
    w = torch.linalg.inv(X.T.matmul(X)).matmul(X.T).matmul(Y)
    return w

'''
X = torch.Tensor([1, 2, 3, 4, 5])
Y = torch.Tensor([3, 5.01, 7, 8.94, 11.05])
ret = torch.Tensor([2.0030, 0.9910])
print(ret)
'''

###########################################
'''
输入一个 N * D 的张量 X 和一个 长度为 N 的一维张量 Y 作为训练数据；
每轮训练手动计算参数 w 的梯度，然后更新 w；
若干轮之后，返回 (D + 1) * 2 的权重矩阵 w

Input a tensor X of N * D and a one-dimensional tensor Y of length N as training data;
The gradient of parameter W was manually calculated for each training round, and then update w;
After several rounds, return the weight matrix W of (D + 1) * 2.

X = torch.tensor([[16.8470, 13.0935, 16.4787,  3.6465, 18.2576],
        [ 8.9662,  5.1513,  3.1874,  9.8732, 13.0335],
        [17.5560, 19.9259,  6.0421,  4.2841, 17.1478],
        [ 3.6849, 18.3430, 18.6264, 10.1852, 15.0222],
        [16.6799,  2.7613,  5.1764,  4.3264, 13.0472],
        [ 7.8734, 17.4444,  1.9220, 12.2270, 13.5414],
        [15.3688, 19.5739,  7.9656,  8.7431,  1.3333],
        [18.4246,  6.1349, 19.3323, 18.7937, 12.8167],
        [11.4212,  5.8471,  5.9637,  7.3721,  2.6311],
        [ 0.9324, 17.9416,  7.2731, 10.0915,  0.3300]])
Y = torch.tensor([0,1,0,1,1,0,0,0,1,1])
ret = torch.tensor([[1.0976, 0.9024],
        [1.1009, 0.8991],
        [1.0705, 0.9295],
        [1.0797, 0.9203],
        [1.0773, 0.9227],
        [1.0092, 0.9908]])
'''

def bi_classification(X, Y):
    num_train = X.shape[0]
    X = torch.cat((X, torch.ones((num_train, 1))), dim = 1)
    Y2 = torch.zeros((num_train, 2))
    Y2[:,Y] = 1
    w = torch.ones((X.shape[1], 2))
    epoch = 250
    lr = 0.001
    loss = 0
    for i in range(epoch):
        preds = torch.matmul(X, w)
        pred = preds
        labels = Y2
        pred = torch.exp(pred)
        sums = torch.sum(pred, dim = 1, keepdim = True)
        pred = pred / sums
        inds = torch.argmax(labels, dim = 1, keepdim = True)

        loss = 0
        for j in range(num_train):
            loss = loss - torch.log(pred[j, inds[j]])
        loss = loss / num_train

        for j in range(num_train):
            pred[j, inds[j]] -= 1
        dw = torch.matmul(X.T, pred)

        w = w - dw * lr
    ret = w.detach()
    return ret
'''
X = torch.tensor([[16.8470, 13.0935, 16.4787,  3.6465, 18.2576],
        [ 8.9662,  5.1513,  3.1874,  9.8732, 13.0335],
        [17.5560, 19.9259,  6.0421,  4.2841, 17.1478],
        [ 3.6849, 18.3430, 18.6264, 10.1852, 15.0222],
        [16.6799,  2.7613,  5.1764,  4.3264, 13.0472],
        [ 7.8734, 17.4444,  1.9220, 12.2270, 13.5414],
        [15.3688, 19.5739,  7.9656,  8.7431,  1.3333],
        [18.4246,  6.1349, 19.3323, 18.7937, 12.8167],
        [11.4212,  5.8471,  5.9637,  7.3721,  2.6311],
        [ 0.9324, 17.9416,  7.2731, 10.0915,  0.3300]])
Y = torch.tensor([0,1,0,1,1,0,0,0,1,1])
print(X)
ret = bi_classification(X, Y)
print(ret)
'''

###########################################
'''
输入一个 N * D 的张量 X 和一个 D * C 的张量 Y 作为训练数据；
每轮训练手动计算参数 w 的梯度，然后更新 w；
若干轮之后，返回 (D + 1) * C 的权重矩阵 w

Input a tensor X of N * D and a tensor X of D * C as training data;
The gradient of parameter W was manually calculated for each training round, and then update w;
After several rounds, return the weight matrix W of (D + 1) * C.
'''
'''
X = torch.tensor([[39.1467, 20.5535, 14.9515, 10.6497, 38.8540],
        [37.3535, 20.5351,  2.2542, 27.6599, 12.1164],
        [18.2645,  5.7404, 39.7290, 47.4515, 48.0882],
        [20.0169, 40.3000,  8.8746, 16.8172, 24.7160],
        [19.6337, 29.4148, 35.9423,  4.5859,  1.9753],
        [42.5378, 44.7354,  5.4101, 48.7900, 34.7609],
        [30.5570,  3.9888, 20.1841,  9.1199,  1.7989],
        [37.1038, 41.9657,  8.6687, 45.2733, 14.3551],
        [40.1467, 20.8888,  0.1283,  4.1633, 49.2245],
        [ 2.8266, 47.3158, 47.9091,  4.3411, 20.5988],
        [43.0030, 28.1347, 31.9239, 19.8220, 24.6456],
        [35.6317, 33.4389, 46.1758, 38.6261, 35.8310],
        [25.4405, 41.8322, 37.7248, 32.7324, 47.7321],
        [ 6.0368, 15.3703,  0.0972, 23.0495, 26.4298],
        [28.8952,  6.8817,  2.3038, 10.0482, 21.5332],
        [30.7931, 32.1851, 45.1780, 22.6372,  5.4868],
        [35.5153, 44.1605, 46.7706, 18.4849, 48.5605],
        [41.5659,  7.7858, 34.4243,  4.5082,  3.3732],
        [ 8.5829, 20.3806, 30.4702, 27.9592, 37.9304],
        [ 2.0725,  0.8398,  6.4065, 47.2383, 45.0296]])
Y = torch.tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.]])
ret = torch.tensor([[-0.0520,  0.0749, -0.0229],
        [ 0.1003, -0.1084,  0.0082],
        [-0.0073,  0.0066,  0.0007],
        [-0.0826,  0.0734,  0.0091],
        [ 0.0102, -0.0450,  0.0347],
        [ 0.0003,  0.0037, -0.0040]])
'''

def multi_classification(X, Y):
    num_train = X.shape[0]
    C = Y.shape[1]
    X = torch.cat((X, torch.ones((num_train, 1))), dim = 1)
    w = torch.zeros((X.shape[1], C))
    epoch = 100
    lr = 0.0001
    loss = 0
    for i in range(epoch):
        preds = torch.matmul(X, w)
        pred = preds
        pred = torch.exp(pred)
        sums = torch.sum(pred, dim = 1, keepdim = True)
        pred = pred / sums
        inds = torch.argmax(Y, dim = 1, keepdim = True)

        loss = 0
        for j in range(num_train):
            loss = loss - torch.log(pred[j, inds[j]])
        loss = loss / num_train

        for j in range(num_train):
            pred[j, inds[j]] -= 1
        dw = torch.matmul(X.T, pred)

        w = w - dw * lr
    ret = w.detach()
    return ret


X = torch.tensor([[39.1467, 20.5535, 14.9515, 10.6497, 38.8540],
        [37.3535, 20.5351,  2.2542, 27.6599, 12.1164],
        [18.2645,  5.7404, 39.7290, 47.4515, 48.0882],
        [20.0169, 40.3000,  8.8746, 16.8172, 24.7160],
        [19.6337, 29.4148, 35.9423,  4.5859,  1.9753],
        [42.5378, 44.7354,  5.4101, 48.7900, 34.7609],
        [30.5570,  3.9888, 20.1841,  9.1199,  1.7989],
        [37.1038, 41.9657,  8.6687, 45.2733, 14.3551],
        [40.1467, 20.8888,  0.1283,  4.1633, 49.2245],
        [ 2.8266, 47.3158, 47.9091,  4.3411, 20.5988],
        [43.0030, 28.1347, 31.9239, 19.8220, 24.6456],
        [35.6317, 33.4389, 46.1758, 38.6261, 35.8310],
        [25.4405, 41.8322, 37.7248, 32.7324, 47.7321],
        [ 6.0368, 15.3703,  0.0972, 23.0495, 26.4298],
        [28.8952,  6.8817,  2.3038, 10.0482, 21.5332],
        [30.7931, 32.1851, 45.1780, 22.6372,  5.4868],
        [35.5153, 44.1605, 46.7706, 18.4849, 48.5605],
        [41.5659,  7.7858, 34.4243,  4.5082,  3.3732],
        [ 8.5829, 20.3806, 30.4702, 27.9592, 37.9304],
        [ 2.0725,  0.8398,  6.4065, 47.2383, 45.0296]])
Y = torch.tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.]])
ret = multi_classification(X, Y)
print(ret)
