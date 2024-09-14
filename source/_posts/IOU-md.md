---
title: IOU
date: 2021-07-27 16:48:46
tags: CV
---

关于这个IOU，之前没接触过CV更没接触过物体检测，这个指标就不太清楚。

这东西叫Intersection-over-union，它干啥的呢，物体检测是要定位出物体的bounding box的，在识别出bounding box的同时还得识别出物体的类别。为了评价这个bounding box的精度，就提出了IOU。

它定义了两个bounding box的重叠度，表示出来就是：

<a target="_blank"><img src="https://latex.codecogs.com/gif.latex?IOU&space;=&space;\frac{A\cap&space;B}{A&space;\cup&space;B}" title="IOU = \frac{A\cap B}{A \cup B}" /></a>

<a target="_blank"><img src="https://latex.codecogs.com/gif.latex?IOU&space;=&space;\frac{S_I}{S_A&space;&plus;&space;S_B&space;&plus;&space;S_I}" title="IOU = \frac{S_I}{S_A + S_B + S_I}" /></a>

就直观起来了。