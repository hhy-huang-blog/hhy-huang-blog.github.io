---
title: STGN-Baseline代码修改
date: 2021-08-06 20:39:44
tags: NLP的一些收获
---
# Where to Go Next: A Spatio-Temporal Gated Network for Next POI Recommendation

最近要给一个模型跑baseline，需要拿STGN做对比，所以就拿这个的代码进行修改。

STGN是一个基于时间和距离差的门结构的RNN推荐系统，模型不算复杂。

数据处理方面也还是很轻松的，困难的在这个模型的训练没有设置早停，也没有计算NDCG。本来以为工作量不大，结果发现这是tensorflow写的。我一开始的思路是在train的Session里创建两个Graph或者用一个Graph但是要修改模型参数，其中一个用于train，另外一个用于predict。搜索了很多，好像没这么干的....后来问了学长，我感觉自己像个傻子，直接在训练用的graph里进行预测不就可以，一开始我的顾虑是二者input的shape不同，因为一个batch_size是10，一个是1。后来我联想到了训练时padding的方法，由于predict的是一个poi的下一个poi，所以在其余9个item的位置补0即可。发现真的可以，效果也不错，看来也没完全傻。