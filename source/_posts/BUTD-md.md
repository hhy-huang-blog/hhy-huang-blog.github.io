---
title: BUTD
date: 2021-07-27 18:31:24
tags: CV
---

# Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering 总结

论文地址: <a href = 'https://arxiv.org/pdf/1707.07998.pdf'>链接</a>

pytorch代码: <a href = 'https://github.com/ezeli/BUTD_model'>链接</a>

这个文章2018年的，主要是提出了以一个使用Bottom-Up和Top-Down Attention模型相结合来解决Image Captioning和VQA的问题。

先说VQA啊....一开始不清楚这是个啥，后来查了一下就是将图片和问题作为输入，然后组织出一条人类语言作为输出，用关总的话来说就是看图说话，当时还有点懵，现在直到他bb了个啥了，就这啊。

其中Bottom-Up Attention指的是使用Faster R-CNN来对图像特征进行提取，Faster R-CNN没有采用滑动窗口（或者说Grid）来获取特征，而是采取采用提取Region Proposal的方法，少去了很多无用的feature。而且普通CNN用于对大样本的分类网路往往会很复杂。相当于对图像信息的Encode。

Top-Down Attention指的是对上面网络得到的特征V进行权重的划分。它分为两个Layer，按顺序先是Top-Down Attention LSTM，然后是Language LSTM。一开始不明白VQA看这个部分的模型是懵的，Language LSTM存在的意义无从得知，查过后它的目的就很明确了，就是生成那段人类语言的输出，这里做的应该类似于NLP里的机器翻译，也是对序列的操作。它的输入序列即：

<a target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{t}^{2}&space;=&space;[\hat{v}_t,&space;h^{1}_{t}]" title="x_{t}^{2} = [\hat{v}_t, h^{1}_{t}]" /></a>

前者为Faster R-CNN输出的feature，后者为第一个LSTM中每一个status的隐藏输出，然后再说这个LSTM的输入：

<a target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{1}^{2}&space;=&space;[h^{2}_{t&space;-&space;1},&space;\bar{v},&space;W_e&space;\Pi&space;_t]" title="x_{1}^{2} = [h^{2}_{t - 1}, \bar{v}, W_e \Pi _t]" /></a>

所以它的输入序列包括Language LSTM的前一个status的隐藏输出，特征的均值，以及embedding matrix 和 one-hot向量的乘积，这个乘积我猜测它表达的结果是一个embedding vector，每一个status都会有一个特殊的one-hot vector，因此每个status的输入也是一个特殊的embedding vector。

BaseLines方面它主要和使用Res NetCNN而非Bottom-Up的ResNet在captioning和VQA分别进行了比较，都取得了进步。