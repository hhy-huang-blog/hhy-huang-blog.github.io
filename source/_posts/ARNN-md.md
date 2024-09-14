---
title: ARNN复现反思
date: 2022-04-26 22:37:56
tags: NLP的一些收获
---

因为找遍了一二三四作，都没有能得到An Attentional Recurrent Neural Networkfor Personalized Next Location Recommendation这篇论文的代码，一作没反应，二三四都让我找一作...麻了，所以硬下头皮准备复现。

其实任务量还好，最幸运的是这篇论文的模型架构与另外一篇DeepMove的模型十分相似，都是先embedding序列后，对序列元素进行attention的思路，不过也是有很多不同的。

## 思路
这篇论文的思路很清晰，先将check in序列处理好，得到用户的历史轨迹，这个历史轨迹包括loc、time和word的序列，分别把它们用相应的维度embedding后，在第三个维度拼接起来得到tensor：x(batch_size, num, dim)，这样这个轨迹序列的元素就融合了地点、时间、语义上的含义。

之后对于历史轨迹的每一个loc，与该地点的所有neighbors求相似度，然后加权进行一次attention，得到targer，也就是与之最相似的loc向量，结果为ck。

然后将x和ck同样在第三个维度拼接起来得到一个新的tensor，让每一个位置的元素融合入与其它loc的转移关系，然后将它pack后输入LSTM，取出最后一个hidden state，融合入user的embedding，最后用softmax得到next poi的概率分布。

其中的loc的neighbors得到的方法是使用基于meta path的随机游走模型得到的，将历史轨迹序列构成图，我这里的操作其实和pageRank的处理方法类似，搞一个邻接表，然后严格按照原路径的类型进行游走，将访问到的loc纳入path，也就是起点loc的neighbor。具体做法详见上篇blog。

## 遇到的问题
（1）在随机游走时，太慢了，虽然现在也不算快，但是一开始参照一位githuber的pageRank代码改造，是用dict代替多维list，连每一步的带权乘法都要自己用for循环写，很慢。后来想着可以把它搞成矩阵，然后转化为tensor，既能调用torh里的乘法，还可以放到GPU上运算，所以就这么做了，真的有很大的改进，但确实也不算快。

（2）不会写attention，第一次复现嘛，一开始很傻，从tensor中一条一条数据遍历，然后找到对应的loc_id，再通过loc_id找到对应的neighbors的序列，然后再对neighbors embedding...太繁琐了，导致一个batch就要两三分钟。

所以，我思考了一下，可不可以把neighbors的embedding也做成一个tensor，然后让二者去运算，这样是可以调用torch.matmul，方便加速的。但是问题在于不同的loc邻居的数量也不同，所以我采用的办法是取游走获得的path中出现次数最多的前n个邻居作为loc的neighbors，这样维度就统一了。

假设batch_size为128，n是10，dim是100，那么一开始的loc_neighbor_emb就是(128, 464, 100, 10)，原本的loc_emb是(128, 464, 100)，为了方便相乘，unsqueeze一下为(128, 464, 1, 100)，这样二者的batch就统一了，为128*464，因为torch.matmul规定四维tensor的运算前两维为batch，前者转置一下，相乘后softmax就是相似度的矩阵了，大小为(128, 464, 1, 10)。这个大小一看就很对，对于每个loc都有10个neighbors对应的weight。最后再将其和neighbors的embedding相乘，得到最终的结果。

然后用这样的方法再尝试，果然快了很多，一个batch就20s左右。时间估计都用在attention前loc_neighbor_emb的构建上了。

（3）每个epoch的最后一个batch不足batch_size，一开始我还想着continue过去，但是想想会影响到测试集到验证的，所以就查了查，发现有解决办法，在DataLoader中设置参数drop_last = True，其实也是drop掉了，不过在一开始就去掉了，不会产生影响。

## 总结
总结一下，最困难的部分也就是核心模块的编写和随机游走的编写了，对于框架代码的编写其实没有涉足过多，毕竟是拿别人的代码改动的，下次有机会还是尝试一下自己从0开始，体会应该更深刻一些吧。另外就是一些torch的函数，了解的还是太少，还是应该多分析分析别人的代码。

## 最后
代码：[ARNN-master](https://github.com/hhy-huang/ARNN-master)

我学识鄙陋，有问题一定要告诉我！
