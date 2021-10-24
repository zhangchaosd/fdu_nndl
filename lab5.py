'''
X = torch.tensor([[1,2,3],[4,5,6]])
Y = torch.tensor([[7,8,9],[7,5,3]])
ret = torch.tensor([[0.1000, 0.2000, 0.3000], [0.4000, 0.5000, 0.6000], [0.7000, 0.8000, 0.9000], [0.7000, 0.5000, 0.3000]])
'''
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

if __name__ == '__main__':
    X = torch.tensor([1,2,3])
    Y = torch.tensor([[7,8,9],[7,5,3]])
    ret = zy3(X, Y)
    print(ret)