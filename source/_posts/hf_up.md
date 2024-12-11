---
title: Huggingface upload models
date: 2024-12-11 22:51:33
tags: notes
---

问题就是总是传不上去，说认证有问题，尽管我加了ssh，而且也在terminal登陆了hf的token，都不行。

然后发现是git clone的问题，不用https的url了，改成用ssh链接，就可以了。
