---
title: 模型的评估与选择
date: 2021-09-16 08:33:09
tags: 机器学习
---

## 评估方法

通常将包含m样本的数据集划分为训练集和测试集。具体的拆分方法：

### 留出法

直接将数据集划分为两个互斥的集合，分别作为测试集和训练集。

在训练集中训练出模型后，用测试集来评估测试误差，作为对泛化误差的估计。

注意：

数据集的划分要尽可能保持数据分布的一致性（分层采样）。

一般若干次随机划分、重复进行实验评估后取平均值。

比例2:1～4:1。

### 交叉验证法

将数据集分层采样划分为k个大小相似的互斥子集，每次用k-1个子集的并集作为训练集，余下的子集作为测试集，最终返回k个测试结果的均值，k最常用的取值是10，叫做10折交叉验证。

其中将数据集划分为k个子集时为了减小因样本划分不同而引入的差别，k折交叉验证通常随机使用不同的划分重复p次，最终结果为p次k折交叉验证结果的均值。

尤其地，当数据集包含m个样本，令k = m，则得到留一法。它不受随机样本划分的影响（只有一种）。结果较为准确。但是数据大的时候开销难以忍受。

### 自助法(Bootstrapping)

给定数据集$D$，对它采样生成D'，即每次从随机从D挑选一个样本放入D'，然后再放回D中。重复m次，得到包含m个样本的数据集D'。

再将D' 最为训练集，D/D' 作为测试集（0.37m个样本）。

数据集小时很有用。但是改变了初始数据集的分布，会引入估计偏差。

## 调参与验证集

参数分为超参和普通参数。

超参一般为神经网络的层数，神经元个数等等。普通参数一般为权重系数等。

验证集合时从训练集中再次划分为训练集和验证集，用来选择超参，选择出最优的超参再放到测试集去跑。一般测试集的结果会低一些。


## 性能度量

在回归任务中，性能度量通常使用均方误差（MSE）。

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=E(f;D)&space;=&space;\frac{1}{m}\sum^{m}){i&space;=&space;1}(f(x_i)&space;-&space;y_i)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(f;D)&space;=&space;\frac{1}{m}\sum^{m}){i&space;=&space;1}(f(x_i)&space;-&space;y_i)^2" title="E(f;D) = \frac{1}{m}\sum^{m}){i = 1}(f(x_i) - y_i)^2" /></a>
</center>

在分类任务中，经常用到错误率和精度。

分类错误率：
<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=E(f;D)&space;=&space;\frac{1}{m}\sum^{m}{i&space;=&space;1}\Pi&space;(f(x_i)&space;\neq&space;y_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(f;D)&space;=&space;\frac{1}{m}\sum^{m}{i&space;=&space;1}\Pi&space;(f(x_i)&space;\neq&space;y_i)" title="E(f;D) = \frac{1}{m}\sum^{m}{i = 1}\Pi (f(x_i) \neq y_i)" /></a>
</center
>
精度：

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=1&space;-&space;E(f;D)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1&space;-&space;E(f;D)" title="1 - E(f;D)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=E(f;D)&space;=&space;\frac{1}{m}\sum^{m}{i&space;=&space;1}\Pi&space;(f(x_i)&space;=&space;y_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(f;D)&space;=&space;\frac{1}{m}\sum^{m}{i&space;=&space;1}\Pi&space;(f(x_i)&space;=&space;y_i)" title="E(f;D) = \frac{1}{m}\sum^{m}{i = 1}\Pi (f(x_i) = y_i)" /></a>
</center>

关于混淆矩阵，也是分类任务，对于真实类别为正例时，若预测为正例则为True Positive否则为False Negative。对于真实类别为负例时，若预测为正例则为False Positive否则为True Negative。

于是给出查准率（Precision）：

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;=&space;\frac{TP}{TP&space;&plus;&space;FP}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;=&space;\frac{TP}{TP&space;&plus;&space;FP}" title="P = \frac{TP}{TP + FP}" /></a>
</center>

与召回率（Recall）：

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=R&space;=&space;\frac{TP}{TP&space;&plus;&space;FN}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R&space;=&space;\frac{TP}{TP&space;&plus;&space;FN}" title="R = \frac{TP}{TP + FN}" /></a>
</center>

由于二者经常变化矛盾，于是提出P-R曲线。

除了PR曲线，也可以使用另外两种度量：

F1度量：

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=F1&space;=&space;\frac{2\times&space;P&space;\times&space;R}{P&space;&plus;&space;R}&space;=&space;\frac{2&space;\times&space;TP}{m&space;&plus;&space;TP&space;-&space;TN}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F1&space;=&space;\frac{2\times&space;P&space;\times&space;R}{P&space;&plus;&space;R}&space;=&space;\frac{2&space;\times&space;TP}{m&space;&plus;&space;TP&space;-&space;TN}" title="F1 = \frac{2\times P \times R}{P + R} = \frac{2 \times TP}{m + TP - TN}" /></a>
</center>

F_{\beta}度量:

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=F_{\beta}&space;=&space;\frac{(1&space;&plus;&space;{\beta}^2&space;\times&space;P&space;\times&space;R)}{(\beta&space;^2&space;\times&space;P)&space;&plus;&space;R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{\beta}&space;=&space;\frac{(1&space;&plus;&space;{\beta}^2&space;\times&space;P&space;\times&space;R)}{(\beta&space;^2&space;\times&space;P)&space;&plus;&space;R}" title="F_{\beta} = \frac{(1 + {\beta}^2 \times P \times R)}{(\beta ^2 \times P) + R}" /></a>
</center>

另外，也可以使用宏平均，算出多个二分类混淆矩阵上计算出平均macro-P与macro-R，再去计算F1。也存在微平均，不过是先计算矩阵元素的平均值再去计算P与R。

为了衡量犯下不同错误的不同的代价，提出了代价敏感错误率。cost01为将0预测为1，cost10为将1预测为0。