---
title: BLEU
date: 2021-08-17 18:23:27
tags: NLP的一些收获
---

# 一种机器翻译的评估方法 BLEU

<a href='https://aclanthology.org/P02-1040.pdf'>论文链接 BLEU: a Method for Automatic Evaluation of Machine Translation</a>

首先给出一组reference和candidate：

Candidate1：It is a guide to action which ensures that the military always obeys the commands of the party.

Candidate2：It is to insure the troops forever hearing the activity guidebook that party direct.

Reference1：It is a guide to action that ensures that the military will forever heed Party commands.

Reference2：It is the guiding principle which guarantees the military forces always being under the command of the Party.

Reference3：It is the practical guide for the army always to heed the directions of the party.

任务是对两个候选案例进行评估。

论文基于词交集和ngram短语交集设计了如下算法进行评估：

<center>
<a target="_blank"><img src="https://latex.codecogs.com/gif.latex?BLEU&space;=&space;BP\cdot&space;exp(\sum^{N}_{n&space;=&space;1}\omega_n&space;logP_n&space;)" title="BLEU = BP\cdot exp(\sum^{N}_{n = 1}\omega_n logP_n )" /></a>
</center>


其中：
<center>
<a target="_blank"><img src="https://latex.codecogs.com/gif.latex?BP&space;=&space;\left\{\begin{matrix}&space;1&space;&&space;if&space;~c&space;>&space;r\\&space;e^{(1-\frac{r}{c})}&space;&&space;if&space;~c&space;\leq&space;r&space;\end{matrix}\right." title="BP = \left\{\begin{matrix} 1 & if ~c > r\\ e^{(1-\frac{r}{c})} & if ~c \leq r \end{matrix}\right." /></a>
</center>

可以看出这个算法由两部分组成，BP和改进后的ngram。

## Modified Ngram

首先说明这个指标是针对于词的出现次数进行评估的。

它也可以分为两部分，因为它显然是一个值与该位置权重的乘积的累加，也就是wn和log(Pn)，这里的n其实就是ngram的n。指的是词中word的个数。wn就是针对于不同的n的权重。

在计算ngram的值时，引入Min和Max。其中Max表示某一个词在n个reference中的出现次数的最大值，Min表示的是某一个词在Candidate中出现的次数和Max中的最小值。然后再求Min与候选词数和的比值。因此可以看出这个指标是对过长的词句有明显的处罚，因为当Candidate中的word没有在reference中出现时，Min的值必为0而候选词的数目会增加。

## BP

Brevity Penalty补足Ngram的缺陷，对过短的candidate进行处罚，其中c是candidate中每一个句子的长度，r是refernce中最接近c的长度。当c越短时，这个系数当然会越小，指标值越小，很合理。

## Merge

将二者整合在一起就是BLEU，取值范围[0,1]，越大越好。

