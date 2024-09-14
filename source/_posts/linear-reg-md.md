---
title: 线性回归
date: 2021-09-23 08:02:40
tags: 机器学习
---

线性模型是一种有监督的学习，每个样本都对应有标签。根据我们预测的结果是否为连续值，分为了线性回归和对数几率回归（分类）。前提是输入和输出之前有线性相关关系。

## 回归任务

基本形式：

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;W&space;^T&space;x&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;W&space;^T&space;x&space;&plus;&space;b" title="f(x) = W ^T x + b" /></a>

其中x是样本属性的线性组合，是一个向量。W是对每一个属性的权值。模型的可解释性强（白箱模型）。

对于模型好坏的评估，这里选择的loss function为平方误差。


<a href="https://www.codecogs.com/eqnedit.php?latex=L(W,&space;b)&space;=&space;\sum_{i&space;=&space;1}^{100}(y_i&space;-&space;(W&space;x_i&space;&plus;&space;b))^2f(x)&space;=&space;W&space;^T&space;x&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(W,&space;b)&space;=&space;\sum_{i&space;=&space;1}^{100}(y_i&space;-&space;(W&space;x_i&space;&plus;&space;b))^2f(x)&space;=&space;W&space;^T&space;x&space;&plus;&space;b" title="L(W, b) = \sum_{i = 1}^{100}(y_i - (W x_i + b))^2f(x) = W ^T x + b" /></a>


下面需要找到一组W，b使得上述的损失函数可以达到最小。方法有两种。一种是最小二乘法，它只对线性回归适用。函数分别对W和b求偏导，让偏导数为0，得到闭式解（closed-form）。

另一种方式是梯度下降（Gradient Descent）。前提是有损失函数，且函数对参数可微。对于W这一个参数的更新来说，它的变化与它当前值所在的位置以及函数梯度相关。

<a href="https://www.codecogs.com/eqnedit.php?latex=W^{'}&space;=&space;W^{0}&space;-&space;lr&space;\frac{dL}{dW}&space;|_{W&space;=&space;W^0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{'}&space;=&space;W^{0}&space;-&space;lr&space;\frac{dL}{dW}&space;|_{W&space;=&space;W^0}" title="W^{'} = W^{0} - lr \frac{dL}{dW} |_{W = W^0}" /></a>

其中lr为学习率。面临的问题是很容易收敛到一个极小值点，因此训练的效果也与初值位置的选择相关。

同理、对于W、b两个参数来讲，就是两个参数同时变化。这是可以将它们的梯度用一个向量来描述、叫做这个函数的梯度。

<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown&space;L" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown&space;L" title="\bigtriangledown L" /></a>

值得注意的是，微分为0的点不一定为极小值点，也可能会有鞍点（saddle point）的存在。另外由于一般会设置一个阈值，loss较低时便停止训练，这样也会导致会在平缓下降的位置处停止训练。但是线性回归由于线性的特性所以不会出现这样的问题。

## 二分类任务

现在思考如何将上述线性回归的结果与类别标签联系起来。可以使用单位阶跃函数或者Sigmoid来进行一个映射。

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;Sigmoid(W&space;^T&space;x&space;&plus;&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;Sigmoid(W&space;^T&space;x&space;&plus;&space;b)" title="f(x) = Sigmoid(W ^T x + b)" /></a>

这样将输出映射到了一个0-1的区间。

对于模型的好坏，首先对于一组W、b产生训练数据的概率为一个似然估计：（x3不属于C1类别）

<a href="https://www.codecogs.com/eqnedit.php?latex=L(W,&space;b)&space;=&space;f_{W,b}(x1)&space;f_{W,b}(x2)(1-&space;f_{W,b}(x3))..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(W,&space;b)&space;=&space;f_{W,b}(x1)&space;f_{W,b}(x2)(1-&space;f_{W,b}(x3))..." title="L(W, b) = f_{W,b}(x1) f_{W,b}(x2)(1- f_{W,b}(x3))..." /></a>

那么L(W, b)取值最大是的参数是我们想要的。为了方便优化，我们给上述函数取一负对数。同时对于展开后的每一项进行一个补齐，标签值正确的系数为1，否则系数为0。

<a href="https://www.codecogs.com/eqnedit.php?latex=-lnf_{W,b}(x1)&space;=&space;-(1*ln(fx1)&space;&plus;&space;0*ln(1-f(x1)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-lnf_{W,b}(x1)&space;=&space;-(1*ln(fx1)&space;&plus;&space;0*ln(1-f(x1)))" title="-lnf_{W,b}(x1) = -(1*ln(fx1) + 0*ln(1-f(x1)))" /></a>

这样就能得到同一化的公式：

<a href="https://www.codecogs.com/eqnedit.php?latex=L(W,b)&space;=&space;\sum_{n}-&space;\left&space;[&space;y_n&space;lnf_{W,b}(x_n)&space;&plus;&space;(1-y_n)ln(1-f_{W,b}(x_n))&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(W,b)&space;=&space;\sum_{n}-&space;\left&space;[&space;y_n&space;lnf_{W,b}(x_n)&space;&plus;&space;(1-y_n)ln(1-f_{W,b}(x_n))&space;\right&space;]" title="L(W,b) = \sum_{n}- \left [ y_n lnf_{W,b}(x_n) + (1-y_n)ln(1-f_{W,b}(x_n)) \right ]" /></a>

这里面其实蕴含了p、q两个分布：

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x&space;=&space;1)&space;=&space;y_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x&space;=&space;1)&space;=&space;y_n" title="p(x = 1) = y_n" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x&space;=&space;0)&space;-&space;1&space;-&space;y_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x&space;=&space;0)&space;-&space;1&space;-&space;y_n" title="p(x = 0) - 1 - y_n" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=q(x&space;=&space;1)&space;=&space;f(x_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q(x&space;=&space;1)&space;=&space;f(x_n)" title="q(x = 1) = f(x_n)" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=q(x&space;=&space;0)&space;=&space;1&space;-&space;f(x_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q(x&space;=&space;0)&space;=&space;1&space;-&space;f(x_n)" title="q(x = 0) = 1 - f(x_n)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=H(p,q)&space;=&space;-\sum_{x}p(x)ln(q(X))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H(p,q)&space;=&space;-\sum_{x}p(x)ln(q(X))" title="H(p,q) = -\sum_{x}p(x)ln(q(X))" /></a>


两个分布越接近，交叉熵越小，也就是效果越好。
