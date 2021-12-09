#神经网络搭建
#author:chenyy
import numpy as np
def sigmoid(x): # 定义激活函数
    return 1/(1+np.exp(-x))
def fun(x): # 激活函数
    return x
def init_network(): # 权重与偏置
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['B1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['B2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['B3']= np.array([0.1,0.2])
    return network
def forward(network,x): # 前向传播
    w1,w2,w3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['B1'],network['B2'],network['B3']
    z1 = sigmoid(np.dot(x,w1)+b1)
    z2 = sigmoid(np.dot(z1,w2)+b2)
    z3 = fun(np.dot(z2,w3)+b3)
    return z3
if __name__=="__main__":
    network = init_network()
    x = np.array([1.0,0.5])
    y = forward(network,x)
    print(y)