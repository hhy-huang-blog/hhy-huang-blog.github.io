---
title: BLIP-2
date: 2024-09-14 02:32:10
tags: vison-LLM
---
## 解决的问题
端到端训练视觉语言模型需要大尺度模型及大规模数据，该过程成本大，本文提出方法基于现有高质量视觉模型及语言大模型进行联合训练，为减少计算量及防止遗忘，作者对预训练模型进行frozen，为了将两任务对齐，作者提出Querying Transformer (Q- Former) 预训练，如图1，其将有用视觉特征传递至LLM输出目标文本。
## 两阶段训练
- 图像编码器frozen，进行学习视觉语言表征
- 使用frozen LLM进行学习视觉到文本生成
## 模型架构
![](/images/blip2_1.jpg)
Q-Former包括两个贡共享self-attention层的transformer子模块：图像transformer（Q-Former左半部分）与frozen image encoder相互作用提取视觉特征；文本transformer（Q-Former右半部分）可作为文本编码器，也可作为文本解码器。
可学习query embedding作为图像transformer输入，通过self-attention层相互作用，通过cross-attention层与frozen图像特征相互作用，query同时通过self-attention层与文本相互作用。根据预训练任务，作者使用不同self-attention mask控制query-text之间交互；作者使用BERTbase初始化Q-Former，cross-attention层进行随机初始化；
### 图像文本对比学习目标（ITC）
<img src="/images/blip2_2.jpg" width="200" height="600"></img>
ITC学习对齐图像表征与文本表征，通过比较成对与非成对的图像-文本相似度实现；计算过程如下：
计算image transformer输出query表征Z ZZ（与可学习query长度相同）与text transformer输出文本表征t中【CLS】token相似性，选取最大值作为图像文本对相似度，为防止信息泄露，作者使用单模态self-attention mask，query与text不能互相可见，防止从文本直接学习；由于image encoder进行frozen，显存释放，可以使用batch负样本而不用像BLIP中使用队列。
### 基于图像文本生成（ITG）
<img src="/images/blip2_3.jpg" width="200" height="600"></img>
ITG根据输入图像训练Q-Former生成文本，由于Q-Former不允许image encoder与text token直接交互，文本生成所需信息通过query进行提取，通过self-attention进行传递至text token，因此query需要捕获文本相关所有信息，作者使用多模态因果self-attention mask控制query-text交互，query无法获取text token，当前text token 可获取所有query及其之前text token。作者将【CLS】token替换为【DEC】token 作为解码任务标记；
### 图文匹配（ITM）
<img src="/images/blip2_4.jpg" width="200" height="600"></img>
ITM为了学习精细化图像文本匹配，作者使用bi-dirention self-atttention mask，所有query与text相互可见，**因此输出的query embedding Z捕获多模态信息**，Z通过二类线性分类器获取logit，logit均值为匹配得分，作者使用《Align before Fuse》中难例负样本挖掘策略创建负样本对。
难例负样本挖掘策略：
当负样本的图像文本对有相同的语义但在细粒度细节上不同，那么该样本是难样本。作者通过对比相似度寻找batch内的 hard negatives。对于一个batch中的每一幅图像，作者根据对比相似性分布从相同的batch中抽取一个负文本，其中与图像更相似的文本有更高的可能被采样。同样的，作者还为每个文本采样一个hard negative图像。

**给每个text采样negative image代码：**
```python
# select a negative image for each text
behavior_embeds_neg = []
behavior_neighbors_embeds_neg = []
for b in range(bs):
    neg_idx = torch.multinomial(weights_t2i[b], 1).item() # randomly sample as negative samples
	behavior_embeds_neg.append(behavior_embeds_world[neg_idx])
   behavior_neighbors_embeds_neg.append(behavior_neighbors_embeds_world[neg_idx])
	behavior_embeds_neg = torch.stack(behavior_embeds_neg, dim=0) # [batch_size, 1, 768]
	behavior_neighbors_embeds_neg = torch.stack(behavior_neighbors_embeds_neg, dim=0)
```
torch.multinomial指的是从当前的概率分布中进行随机采样，也就是说从ITC构建的相似矩阵中的similarity softmax后的结果作为每个样本被采样的概率分布，然后进行采样，这样会优先采集难负样本。
**给每个image采样negative text代码：**
```python
# select a negative text for each image
text_ids_neg = []
text_atts_neg = []
for b in range(bs):
	neg_idx = torch.multinomial(weights_i2t[b], 1).item()
	text_ids_neg.append(text_input_ids_world[neg_idx])
	text_atts_neg.append(text_attention_mask_world[neg_idx])
text_ids_neg = torch.stack(text_ids_neg, dim=0) # [batch_size, 512]
text_atts_neg = torch.stack(text_atts_neg, dim=0) # [batch_size, 512]
```
策略是一样的。
## 从大规模语言模型学习视觉到语言生成
![[Pasted image 20240829171330.png]]
作者将Q-Former与LLM相连，后去LLM的语言生成能力。FC层映射输出的query embedding Z至LLM的text embedding；基于LLM Q-Former提取到的视觉表征作为soft visual prompt，由于Q-Former已经预训练用于提取对文本有用的视觉表征，减轻LLM学习视觉-文本对齐的负担。
作者实验两种LLM，decoder-based LLM以及encoder-decoder-based LLM。
- 对于decoder-based LLM，作者使用language modeling loss进行预训练，frozen LLM进行文本生成；
- 对于encoder-decoder-based LLM，使用prefix language modeling loss预训练，将text分为两部分，text前半部分与视觉表征concat输入LLM编码器，后半部分作为LLM解码器的生成目标。
## 模型预训练
### 预训练数据
BLIP-2使用与BLIP相同数据，129M图片，包括COCO、Visual Genome、CC3M、CC12M、SBU，其中115M来自 LAION400M，使用CapFilt对网图进行生成caption，具体步骤如下：
1、使用$BLIP_{Large}$生成10个caption；
2、生成10个caption+原始web caption通过CLIP ViT-L/14模型与对应图像进行相似度排序；
3、选取top2作为该图的caption，以此作为训练数据；
### 预训练图像编码器与LLM
- 视觉编码器：ViT-L/14 from CLIP、ViT-G/14 from EVA-CLIP  移除ViT最后一层，使用倒数第二层特征
- LLM模型：
	- decoder-only llm： 无监督训练的OPT
	- encoder-decoder llm：基于SFT的Flan T5
### 两阶段训练
- 第一阶段：从冻结的图像编码器中引导视觉语言表示学习。 通过以上三个目标训练Q-former
- 第二阶段：让LLM进行答案生成，freeze LLM，tuning Q-former和Projetor，进行生成式训练。
## 结论
BLIP-2是一种通用且计算高效的视觉语言预训练方案，使用frozen 预训练图像编码器及LLM，在多个视觉语言任务达到SOTA，也证明了其在零样本instructed image-to-text生成能力。
