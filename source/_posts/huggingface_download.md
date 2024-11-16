---
title: Huggingface download model
date: 2024-11-16 23:43:34
tags: notes
---

棘手了好久的问题终于解决了, 手里目前有几台服务器, 经常用好用的几台没有设置代理,导致很多大模型的参数下载很麻烦, hf访问不到, modelscope又经常没有bin或者safetensor格式的模型文件, 本地下载下来再rsync到服务器又太费时间了, 尤其像7B以上的模型。

然后就发现了有gitee, 然后在程序一开始改一下hf的环境变量就可以了, 境内的网络也可以丝滑访问。

```python
import os
os.environ["HF_HOME"] = "~/.cache/gitee-ai"
os.environ["HF_ENDPOINT"] = "https://hf-api.gitee.com"
```
