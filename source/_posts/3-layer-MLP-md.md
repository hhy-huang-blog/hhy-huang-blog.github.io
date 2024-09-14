---
title: 三层感知机-step by step
date: 2021-11-09 23:12:33
tags: 机器学习
---

## 实现内容：
1. 实现一个三层感知机
2. 对手写数字数据集进行分类
3. 绘制损失值变化曲线
4. 完成kaggle MNIST手写数字分类任务，根据给定的超参数训练模型，完成表格的填写

## 实现

数据集使用手写数字集。并且40%作测试集，60%做训练集。


```py
import matplotlib.pyplot as plt
%matplotlib inline
from time import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(load_digits()['data'], load_digits()['target'], test_size = 0.4, random_state = 32)
```

接下来是数据预处理，神经网络的训练方法一般是基于梯度的优化算法，如梯度下降，为了让这类算法能更好的优化神经网络，我们往往需要对数据集进行归一化，这里我们选择对数据进行标准化。

减去均值可以让数据以0为中心，除以标准差可以让数据缩放到一个较小的范围内。这样可以使得梯度的下降方向更多样，同时缩小梯度的数量级，让学习变得稳定。  

首先需要对训练集进行标准化，针对每个特征求出其均值和标准差，然后用训练集的每个样本减去均值除以标准差，就得到了新的训练集。然后用测试集的每个样本，减去训练集的均值，除以训练集的标准差，完成对测试集的标准化。
```py
trainY_mat = np.zeros((len(trainY), 10))
trainY_mat[np.arange(0, len(trainY), 1), trainY] = 1

testY_mat = np.zeros((len(testY), 10))
testY_mat[np.arange(0, len(testY), 1), testY] = 1
```
下面是参数的初始化。
```py
def initialize(h, K):
    '''
    参数初始化
    
    Parameters
    ----------
    h: int: 隐藏层单元个数
    
    K: int: 输出层单元个数
    
    Returns
    ----------
    parameters: dict，参数，键是"W1", "b1", "W2", "b2"
    
    '''
    np.random.seed(32)
    W_1 = np.random.normal(size = (trainX.shape[1], h)) * 0.01
    b_1 = np.zeros((1, h))
    
    np.random.seed(32)
    W_2 = np.random.normal(size = (h, K)) * 0.01
    b_2 = np.zeros((1, K))
    
    parameters = {'W1': W_1, 'b1': b_1, 'W2': W_2, 'b2': b_2}
    
    return parameters
```
向前传播，这里具体指的就是依据公式向前计算值。

这里有一点要注意，矩阵的点乘是使用```np.dot()```进行的，否则py会默认为元素乘。
```py
def linear_combination(X, W, b):
    '''
    计算Z，Z = XW + b
    
    Parameters
    ----------
    X: np.ndarray, shape = (n, m)，输入的数据
    
    W: np.ndarray, shape = (m, h)，权重
    
    b: np.ndarray, shape = (1, h)，偏置
    
    Returns
    ----------
    Z: np.ndarray, shape = (n, h)，线性组合后的值
    
    '''
    
    # Z = XW + b
    # YOUR CODE HERE
    Z = np.dot(X,W) + b
    
    return Z
```
每一线性层的输出都要经过一个activate，隐藏层的activate为ReLu。
```py
def ReLU(X):
    '''
    ReLU激活函数
    
    Parameters
    ----------
    X: np.ndarray，待激活的矩阵
    
    Returns
    ----------
    activations: np.ndarray, 激活后的矩阵
    
    '''
    
    # YOUR CODE HERE
    X[X < 0] = 0
    activations = X
    
    return activations
```
输出层要经过softmax找到每一个label的概率大小。这里值得注意的是，O矩阵的求和是对每一行的各个元素求和，而不是对所有元素求和，所以要有```axis=1```，对行进行sum操作，并保持维度。

前一个```my_softmax(O)```会导致对于较小值的output，会导致分母为0的情况，所以要对其进行一些处理，让O的每一个元素减去该行的最大值，这样能保证取exp后至少一个元素为1，所以不会出现NaN的情况。
```py
def my_softmax(O):
    '''
    softmax激活
    '''
    # YOUR CODE HERE
    return np.exp(O) / np.sum(np.exp(O), axis = 1, keepdims = True)

def softmax(O):
    '''
    softmax激活函数
    
    Parameters
    ----------
    O: np.ndarray，待激活的矩阵
    
    Returns
    ----------
    activations: np.ndarray, 激活后的矩阵
    
    '''
    
    # YOUR CODE HEER
    O = O - np.max(O, axis=1, keepdims=True)
    activations = my_softmax(O)
    return activations
```
接下来是实现损失函数，交叉熵损失函数：
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathrm{loss}&space;=&space;-&space;\frac{1}{n}&space;\sum_n&space;\sum^{K}_{k=1}&space;y_k&space;\log{(\hat{y_k})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathrm{loss}&space;=&space;-&space;\frac{1}{n}&space;\sum_n&space;\sum^{K}_{k=1}&space;y_k&space;\log{(\hat{y_k})}" title="\mathrm{loss} = - \frac{1}{n} \sum_n \sum^{K}_{k=1} y_k \log{(\hat{y_k})}" /></a>
这里又会出一个问题，交叉熵损失函数中，我们需要对softmax的激活值取对数，也就是log\haty，这就要求我们的激活值全都是大于0的数，不能等于0，但是我们实现的softmax在有些时候确实会输出0。这就使得在计算loss的时候会出现问题，解决这个问题的方法是log softmax。所谓log softmax，就是将交叉熵中的对数运算与softmax结合起来，避开为0的情况。

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\log{\frac{\exp{(O_i)}}{\sum_K&space;\exp{(O_k)}}}&space;&=&space;\log{\frac{\exp{(O_i&space;-&space;\mathrm{max}(O))}}{\sum_K&space;\exp{(O_k&space;-&space;\mathrm{max}(O))}}}\\&space;&=&space;O_i&space;-&space;\mathrm{max}(O)&space;-&space;\log{\sum_K&space;\exp{(O_k&space;-&space;\mathrm{max}(O))}}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\log{\frac{\exp{(O_i)}}{\sum_K&space;\exp{(O_k)}}}&space;&=&space;\log{\frac{\exp{(O_i&space;-&space;\mathrm{max}(O))}}{\sum_K&space;\exp{(O_k&space;-&space;\mathrm{max}(O))}}}\\&space;&=&space;O_i&space;-&space;\mathrm{max}(O)&space;-&space;\log{\sum_K&space;\exp{(O_k&space;-&space;\mathrm{max}(O))}}&space;\end{aligned}" title="\begin{aligned} \log{\frac{\exp{(O_i)}}{\sum_K \exp{(O_k)}}} &= \log{\frac{\exp{(O_i - \mathrm{max}(O))}}{\sum_K \exp{(O_k - \mathrm{max}(O))}}}\\ &= O_i - \mathrm{max}(O) - \log{\sum_K \exp{(O_k - \mathrm{max}(O))}} \end{aligned}" /></a>

这样我们再计算loss的时候就可以把输出层的输出直接放到log softmax中计算，不用先激活，再取对数了。
```py
def log_softmax(x):
    '''
    log softmax
    
    Parameters
    ----------
    x: np.ndarray，待激活的矩阵
    
    Returns
    ----------
    log_activations: np.ndarray, 激活后取了对数的矩阵
    
    '''
    # YOUR CODE HERE
    log_activations = x - np.max(x) - np.log( np.sum(np.exp(x - np.max(x)), axis = 1, keepdims = True) )
    
    return log_activations
```
然后编写`cross_entropy_with_softmax`。函数内容不再赘述。
```py
def cross_entropy_with_softmax(y_true, O):
    '''
    求解交叉熵损失函数，这里需要使用log softmax，所以参数分别是真值和未经softmax激活的输出值

    Parameters
    ----------
    y_true: np.ndarray，shape = (n, K), 真值
    
    O: np.ndarray, shape = (n, K)，softmax激活前的输出层的输出值
    
    Returns
    ----------
    loss: float, 平均的交叉熵损失值
    
    '''
    
    # 平均交叉熵损失
    # YOUR CODE HERE
    loss = - 1/len(y_true) * np.sum(np.sum(y_true * log_softmax(O)))    # 这里是元素乘
    
    return loss
```
正是因为softmax激活与交叉熵损失会有这样的问题，所以在很多深度学习框架中，交叉熵损失函数就直接带有了激活的功能，所以我们在实现前向传播计算的时候，就不要加softmax激活函数了。
```py
def forward(X, parameters):
    '''
    前向传播，从输入一直到输出层softmax激活前的值
    
    Parameters
    ----------
    X: np.ndarray, shape = (n, m)，输入的数据
    
    parameters: dict，参数
    
    Returns
    ----------
    O: np.ndarray, shape = (n, K)，softmax激活前的输出层的输出值
    
    '''
    # 输入层到隐藏层
    # YOUR CODE HERE
    Z = np.dot(X, parameters['W1']) + parameters['b1']
    
    # 隐藏层的激活
    # YOUR CODE HERE
    H = ReLU(Z)
    
    # 隐藏层到输出层
    # YOUR CODE HERE
    O = np.dot(H, parameters['W2']) + parameters['b2']

    return O
```
下面是反向传播，也是本篇blog的重点。首先是偏导的推导，细节不再赘述，使用链式求导法则认真推导即可。

forward公式：最后一层的输出，使用softmax函数激活，得到神经网络计算出的各类的概率值。
<a href="https://www.codecogs.com/eqnedit.php?latex=Z&space;=&space;XW_1&space;&plus;&space;b_1\\&space;H_1&space;=&space;\mathrm{ReLU}(Z)\\&space;O&space;=&space;H_1&space;W_2&space;&plus;&space;b_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;XW_1&space;&plus;&space;b_1\\&space;H_1&space;=&space;\mathrm{ReLU}(Z)\\&space;O&space;=&space;H_1&space;W_2&space;&plus;&space;b_2" title="Z = XW_1 + b_1\\ H_1 = \mathrm{ReLU}(Z)\\ O = H_1 W_2 + b_2" /></a>

损失函数对参数W_2和b_2的偏导数：
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\frac{\partial&space;loss}{\partial&space;W_2}&space;&&space;=&space;\frac{\partial&space;\mathrm{loss}}{\partial&space;\hat{y}}&space;\frac{\partial&space;\hat{y}}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;W_2}\\&space;&&space;=&space;\frac{\partial&space;loss}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;W_2}\\&space;&&space;=&space;\frac{1}{n}&space;(\hat{y}&space;-&space;y)&space;\frac{\partial&space;O}{\partial&space;W_2}\\&space;&&space;=&space;\frac{1}{n}&space;[{H_1}^\mathrm{T}&space;(\hat{y}&space;-&space;y)]&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\frac{\partial&space;loss}{\partial&space;W_2}&space;&&space;=&space;\frac{\partial&space;\mathrm{loss}}{\partial&space;\hat{y}}&space;\frac{\partial&space;\hat{y}}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;W_2}\\&space;&&space;=&space;\frac{\partial&space;loss}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;W_2}\\&space;&&space;=&space;\frac{1}{n}&space;(\hat{y}&space;-&space;y)&space;\frac{\partial&space;O}{\partial&space;W_2}\\&space;&&space;=&space;\frac{1}{n}&space;[{H_1}^\mathrm{T}&space;(\hat{y}&space;-&space;y)]&space;\end{aligned}" title="\begin{aligned} \frac{\partial loss}{\partial W_2} & = \frac{\partial \mathrm{loss}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial O} \frac{\partial O}{\partial W_2}\\ & = \frac{\partial loss}{\partial O} \frac{\partial O}{\partial W_2}\\ & = \frac{1}{n} (\hat{y} - y) \frac{\partial O}{\partial W_2}\\ & = \frac{1}{n} [{H_1}^\mathrm{T} (\hat{y} - y)] \end{aligned}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\frac{\partial&space;loss}{\partial&space;b_2}&space;&&space;=&space;\frac{\partial&space;\mathrm{loss}}{\partial&space;\hat{y}}&space;\frac{\partial&space;\hat{y}}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;b_2}\\&space;&&space;=&space;\frac{\partial&space;loss}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;b_2}\\&space;&&space;=&space;\frac{1}{n}&space;(\hat{y}&space;-&space;y)&space;\frac{\partial&space;O}{\partial&space;b_2}\\&space;&&space;=&space;\frac{1}{n}&space;\sum^n_{i=1}&space;(\hat{y_i}&space;-&space;y_i)&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\frac{\partial&space;loss}{\partial&space;b_2}&space;&&space;=&space;\frac{\partial&space;\mathrm{loss}}{\partial&space;\hat{y}}&space;\frac{\partial&space;\hat{y}}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;b_2}\\&space;&&space;=&space;\frac{\partial&space;loss}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;b_2}\\&space;&&space;=&space;\frac{1}{n}&space;(\hat{y}&space;-&space;y)&space;\frac{\partial&space;O}{\partial&space;b_2}\\&space;&&space;=&space;\frac{1}{n}&space;\sum^n_{i=1}&space;(\hat{y_i}&space;-&space;y_i)&space;\end{aligned}" title="\begin{aligned} \frac{\partial loss}{\partial b_2} & = \frac{\partial \mathrm{loss}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial O} \frac{\partial O}{\partial b_2}\\ & = \frac{\partial loss}{\partial O} \frac{\partial O}{\partial b_2}\\ & = \frac{1}{n} (\hat{y} - y) \frac{\partial O}{\partial b_2}\\ & = \frac{1}{n} \sum^n_{i=1} (\hat{y_i} - y_i) \end{aligned}" /></a>

求得loss对W_1和b_1的偏导数：
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\frac{\partial&space;loss}{\partial&space;W_1}&space;&&space;=&space;\frac{\partial&space;\mathrm{loss}}{\partial&space;\hat{y}}&space;\frac{\partial&space;\hat{y}}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;H_1}&space;\frac{\partial&space;H_1}{\partial&space;Z}&space;\frac{\partial&space;Z}{\partial&space;W_1}\\&space;&&space;=&space;\frac{\partial&space;loss}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;H_1}&space;\frac{\partial&space;H_1}{\partial&space;Z}&space;\frac{\partial&space;Z}{\partial&space;W_1}\\&space;&&space;=&space;\frac{1}{n}&space;{X}^\mathrm{T}&space;[(\hat{y}&space;-&space;y)&space;{W_2}^\mathrm{T}&space;\frac{\partial&space;H_1}{\partial&space;Z}]\\&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\frac{\partial&space;loss}{\partial&space;W_1}&space;&&space;=&space;\frac{\partial&space;\mathrm{loss}}{\partial&space;\hat{y}}&space;\frac{\partial&space;\hat{y}}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;H_1}&space;\frac{\partial&space;H_1}{\partial&space;Z}&space;\frac{\partial&space;Z}{\partial&space;W_1}\\&space;&&space;=&space;\frac{\partial&space;loss}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;H_1}&space;\frac{\partial&space;H_1}{\partial&space;Z}&space;\frac{\partial&space;Z}{\partial&space;W_1}\\&space;&&space;=&space;\frac{1}{n}&space;{X}^\mathrm{T}&space;[(\hat{y}&space;-&space;y)&space;{W_2}^\mathrm{T}&space;\frac{\partial&space;H_1}{\partial&space;Z}]\\&space;\end{aligned}" title="\begin{aligned} \frac{\partial loss}{\partial W_1} & = \frac{\partial \mathrm{loss}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial O} \frac{\partial O}{\partial H_1} \frac{\partial H_1}{\partial Z} \frac{\partial Z}{\partial W_1}\\ & = \frac{\partial loss}{\partial O} \frac{\partial O}{\partial H_1} \frac{\partial H_1}{\partial Z} \frac{\partial Z}{\partial W_1}\\ & = \frac{1}{n} {X}^\mathrm{T} [(\hat{y} - y) {W_2}^\mathrm{T} \frac{\partial H_1}{\partial Z}]\\ \end{aligned}" /></a>

ReLu的偏导数：

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\mathrm{ReLU(x)}}{\partial&space;x}&space;=&space;\begin{cases}&space;0&space;&&space;\text{if&space;}&space;x&space;<&space;0\\&space;1&space;&&space;\text{if&space;}&space;x&space;\geq&space;0&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\mathrm{ReLU(x)}}{\partial&space;x}&space;=&space;\begin{cases}&space;0&space;&&space;\text{if&space;}&space;x&space;<&space;0\\&space;1&space;&&space;\text{if&space;}&space;x&space;\geq&space;0&space;\end{cases}" title="\frac{\partial \mathrm{ReLU(x)}}{\partial x} = \begin{cases} 0 & \text{if } x < 0\\ 1 & \text{if } x \geq 0 \end{cases}" /></a>

从而：

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;loss}{\partial&space;{W_1}_{ij}}&space;=&space;\begin{cases}&space;0&space;&&space;\text{if&space;}&space;{Z}_{ij}&space;<&space;0\\&space;\frac{1}{n}&space;{X}^\mathrm{T}&space;(\hat{y}&space;-&space;y)&space;{W_2}^\mathrm{T}&space;&&space;\text{if&space;}&space;{Z}_{ij}&space;\geq&space;0&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;loss}{\partial&space;{W_1}_{ij}}&space;=&space;\begin{cases}&space;0&space;&&space;\text{if&space;}&space;{Z}_{ij}&space;<&space;0\\&space;\frac{1}{n}&space;{X}^\mathrm{T}&space;(\hat{y}&space;-&space;y)&space;{W_2}^\mathrm{T}&space;&&space;\text{if&space;}&space;{Z}_{ij}&space;\geq&space;0&space;\end{cases}" title="\frac{\partial loss}{\partial {W_1}_{ij}} = \begin{cases} 0 & \text{if } {Z}_{ij} < 0\\ \frac{1}{n} {X}^\mathrm{T} (\hat{y} - y) {W_2}^\mathrm{T} & \text{if } {Z}_{ij} \geq 0 \end{cases}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\frac{\partial&space;loss}{\partial&space;b_1}&space;&&space;=&space;\frac{\partial&space;\mathrm{loss}}{\partial&space;\hat{y}}&space;\frac{\partial&space;\hat{y}}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;H_1}&space;\frac{\partial&space;H_1}{\partial&space;Z}&space;\frac{\partial&space;Z}{\partial&space;b_1}\\&space;&&space;=&space;\frac{\partial&space;loss}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;H_1}&space;\frac{\partial&space;H_1}{\partial&space;Z}&space;\frac{\partial&space;Z}{\partial&space;b_1}\\&space;&&space;=&space;\frac{1}{n}&space;(\hat{y}&space;-&space;y)&space;{W_2}^\mathrm{T}&space;\frac{\partial&space;H_1}{\partial&space;Z}\\&space;&&space;=&space;\begin{cases}&space;0&space;&\text{if&space;}&space;{Z}_{ij}&space;<&space;0\\&space;\frac{1}{n}&space;\sum_n&space;(\hat{y}&space;-&space;y)&space;{W_2}^\mathrm{T}&space;&\text{if&space;}&space;{Z}_{ij}&space;\geq&space;0&space;\end{cases}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\frac{\partial&space;loss}{\partial&space;b_1}&space;&&space;=&space;\frac{\partial&space;\mathrm{loss}}{\partial&space;\hat{y}}&space;\frac{\partial&space;\hat{y}}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;H_1}&space;\frac{\partial&space;H_1}{\partial&space;Z}&space;\frac{\partial&space;Z}{\partial&space;b_1}\\&space;&&space;=&space;\frac{\partial&space;loss}{\partial&space;O}&space;\frac{\partial&space;O}{\partial&space;H_1}&space;\frac{\partial&space;H_1}{\partial&space;Z}&space;\frac{\partial&space;Z}{\partial&space;b_1}\\&space;&&space;=&space;\frac{1}{n}&space;(\hat{y}&space;-&space;y)&space;{W_2}^\mathrm{T}&space;\frac{\partial&space;H_1}{\partial&space;Z}\\&space;&&space;=&space;\begin{cases}&space;0&space;&\text{if&space;}&space;{Z}_{ij}&space;<&space;0\\&space;\frac{1}{n}&space;\sum_n&space;(\hat{y}&space;-&space;y)&space;{W_2}^\mathrm{T}&space;&\text{if&space;}&space;{Z}_{ij}&space;\geq&space;0&space;\end{cases}&space;\end{aligned}" title="\begin{aligned} \frac{\partial loss}{\partial b_1} & = \frac{\partial \mathrm{loss}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial O} \frac{\partial O}{\partial H_1} \frac{\partial H_1}{\partial Z} \frac{\partial Z}{\partial b_1}\\ & = \frac{\partial loss}{\partial O} \frac{\partial O}{\partial H_1} \frac{\partial H_1}{\partial Z} \frac{\partial Z}{\partial b_1}\\ & = \frac{1}{n} (\hat{y} - y) {W_2}^\mathrm{T} \frac{\partial H_1}{\partial Z}\\ & = \begin{cases} 0 &\text{if } {Z}_{ij} < 0\\ \frac{1}{n} \sum_n (\hat{y} - y) {W_2}^\mathrm{T} &\text{if } {Z}_{ij} \geq 0 \end{cases} \end{aligned}" /></a>

描述完公式后下面来用代码实现，首先dW2和db2的代码是很显然的。对于dW1，这里涉及到的ReLu的偏导数，很显然如果hidden层的值小于零对应ReLu为0时，定义其偏导为0，那么如何确定dW1中的那些值是由该定义得到的呢。如果我们眼光狭窄只分析dW2公式的最后结果必然很难分析出来，因为最终的dW2是(hidden, output)维度的，而relu_regard是(n, hidden)维度的，直接对它们进行关联显然不现实。那么需要追根溯源，深入了解这个dW2的来由。

在dW2分段函数的前一步，它的结果是XT点积后面的一堆，其中H对Z的偏导其实就是ReLu的偏导，是在这里决定了dW2的值，再来分析一下维度1/n可以broadcast不用管，后面是```(n, input)T · [(n, output) · (hidden, output)T * (n, hidden)]```这样的维度关系。这里尤其要注意最后一个运算，我一开始卡在这里好久，因为这里涉及到了元素乘，```(n, hidden) * (n, hidden)```，这里决定了哪个计算位置的值来自于ReLu的0，元素乘后再与X的转置计算。

db2同理。

```py
def compute_gradient(y_true, y_pred, H, Z, X, parameters):
    '''
    计算梯度
    
    Parameters
    ----------
    y_true: np.ndarray，shape = (n, K), 真值
    
    y_pred: np.ndarray, shape = (n, K)，softmax激活后的输出层的输出值
    
    H: np.ndarray, shape = (n, h)，隐藏层激活后的值
    
    Z: np.ndarray, shape = (n, h), 隐藏层激活前的值
    
    X: np.ndarray, shape = (n, m)，输入的原始数据
    
    parameters: dict，参数
    
    Returns
    ----------
    grads: dict, 梯度
    
    '''
    
    # 计算W2的梯度
    # YOUR CODE HERE
    dW2 = (1/len(y_true)) * np.dot(np.transpose(H), (y_pred - y_true))
    
    # 计算b2的梯度
    # YOUR CODE HERE
    db2 = (1/len(y_true)) * np.sum(y_pred - y_true, axis=0)
    
    # 计算ReLU的梯度
    relu_grad = Z.copy()
    relu_grad[relu_grad >= 0] = 1
    relu_grad[relu_grad < 0] = 0
    
    # 计算W1的梯度
    # YOUR CODE HERE
    dW1 = 1/len(y_true) * np.dot(np.transpose(X), np.dot((y_pred - y_true), np.transpose(parameters['W2'])) * relu_grad  )
    # 计算b1的梯度
    # YOUR CODE HERE
    db1 = 1/len(y_true) * np.sum(np.dot((y_pred - y_true), np.transpose(parameters['W2'])) * relu_grad, axis=0)
    
    grads = {'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}
    
    return grads
```
梯度下降，反向传播，参数更新。
```py
def update(parameters, grads, learning_rate):
    '''
    参数更新
    
    Parameters
    ----------
    parameters: dict，参数
    
    grads: dict, 梯度
    
    learning_rate: float, 学习率
    
    '''
    parameters['W2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
```
```py
def backward(y_true, y_pred, H, Z, X, parameters, learning_rate):
    '''
    计算梯度，参数更新
    
    Parameters
    ----------
    y_true: np.ndarray，shape = (n, K), 真值
    
    y_pred: np.ndarray, shape = (n, K)，softmax激活后的输出层的输出值
    
    H: np.ndarray, shape = (n, h)，隐藏层激活后的值
    
    Z: np.ndarray, shape = (n, h), 隐藏层激活前的值
    
    X: np.ndarray, shape = (n, m)，输入的原始数据
    
    parameters: dict，参数
    
    learning_rate: float, 学习率
    
    '''
    # 计算梯度
    # YOUR CODE HERE
    grads = compute_gradient(y_true, y_pred, H, Z, X, parameters)
    
    # 更新参数
    # YOUR CODE HERE
    update(parameters, grads, learning_rate)
```
训练。
```py
def train(trainX, trainY, testX, testY, parameters, epochs, learning_rate = 0.01, verbose = False):
    '''
    训练
    
    Parameters
    ----------
    Parameters
    ----------
    trainX: np.ndarray, shape = (n, m), 训练集
    
    trainY: np.ndarray, shape = (n, K), 训练集标记
    
    testX: np.ndarray, shape = (n_test, m)，测试集
    
    testY: np.ndarray, shape = (n_test, K)，测试集的标记
    
    parameters: dict，参数
    
    epochs: int, 要迭代的轮数
    
    learning_rate: float, default 0.01，学习率
    
    verbose: boolean, default False，是否打印损失值
    
    Returns
    ----------
    training_loss_list: list(float)，每迭代一次之后，训练集上的损失值
    
    testing_loss_list: list(float)，每迭代一次之后，测试集上的损失值
    
    '''
    # 存储损失值
    training_loss_list = []
    testing_loss_list = []
    
    for i in range(epochs):
        
        # 这里要计算出Z和H，因为后面反向传播计算梯度的时候需要这两个矩阵
        Z = linear_combination(trainX, parameters['W1'], parameters['b1'])
        H = ReLU(Z)
        train_O = linear_combination(H, parameters['W2'], parameters['b2'])
        train_y_pred = softmax(train_O)
        training_loss = cross_entropy_with_softmax(trainY, train_O)
        
        test_O = forward(testX, parameters)
        testing_loss = cross_entropy_with_softmax(testY, test_O)
        
        if verbose == True:
            print('epoch %s, training loss:%s'%(i + 1, training_loss))
            print('epoch %s, testing loss:%s'%(i + 1, testing_loss))
            print()
        
        training_loss_list.append(training_loss)
        testing_loss_list.append(testing_loss)
        
        backward(trainY, train_y_pred, H, Z, trainX, parameters, learning_rate)
    return training_loss_list, testing_loss_list
```
绘制loss随epoch的变化曲线。
```py
def plot_loss_curve(training_loss_list, testing_loss_list):
    '''
    绘制损失值变化曲线
    
    Parameters
    ----------
    training_loss_list: list(float)，每迭代一次之后，训练集上的损失值
    
    testing_loss_list: list(float)，每迭代一次之后，测试集上的损失值
    
    '''
    plt.figure(figsize = (10, 6))
    plt.plot(training_loss_list, label = 'training loss')
    plt.plot(testing_loss_list, label = 'testing loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
```
预测
```py
def predict(X, parameters):
    '''
    预测，调用forward函数完成神经网络对输入X的计算，然后完成类别的划分，取每行最大的那个数的下标作为标记
    
    Parameters
    ----------
    X: np.ndarray, shape = (n, m), 训练集
    
    parameters: dict，参数
    
    Returns
    ----------
    prediction: np.ndarray, shape = (n, 1)，预测的标记
    
    '''
    # 用forward函数得到softmax激活前的值
    # YOUR CODE HERE
    O = forward(X, parameters)
    
    # 计算softmax激活后的值
    # YOUR CODE HERE
    y_pred = softmax(O)
    
    # 取每行最大的元素对应的下标
    # YOUR CODE HERE
    prediction = np.argmax(y_pred, axis=1)
    
    return prediction
```
训练一个不算特别优秀的3-layer-perceptron。
```py
from sklearn.metrics import accuracy_score
start_time = time()

h = 50
K = 10
parameters = initialize(h, K)
training_loss_list, testing_loss_list = train(trainX, trainY_mat, testX, testY_mat, parameters, 1000, 0.03, False)

end_time = time()
print('training time: %s s'%(end_time - start_time))
prediction = predict(testX, parameters)
print(accuracy_score(prediction, testY))
plot_loss_curve(training_loss_list, testing_loss_list)
``` 
到这里就结束了，其实不算复杂，数值计算的细节比较重要，以前经常用pytorch来写BP、RNN之类的，但是很少从底层去实现过，这还是一个简单的感知机模型，较复杂的基础模型涉及到的内容可能更复杂。只能说我企图学会吧。