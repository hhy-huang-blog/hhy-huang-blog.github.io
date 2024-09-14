---
title: TransE的理解与实现
date: 2021-07-14 00:48:07
tags: NLP的一些收获
---

（依附于博主yuanwyue代码https://blog.csdn.net/shunaoxi2313/article/details/89766467）

## 理解如下图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210401002822969.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjA1Mjg4Ng==,size_16,color_FFFFFF,t_70#pic_center)![在这里插入图片描述](https://img-blog.csdnimg.cn/20210401002903543.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjA1Mjg4Ng==,size_16,color_FFFFFF,t_70#pic_center)
## 附上那位博主的代码
自己加了一些有没有的注释帮助理解

```py
import codecs
import random
import math
import numpy as np
import copy
import time

entity2id = {}
relation2id = {}


def data_loader(file):
    file1 = file + "train.txt"  # entity entity relation
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        # 逐行读取
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            # 删除空白字符，并以'\t'为界划分为list
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            # 将符合标准的数据以dict形式写入entity2id
            entity2id[line[0]] = line[1]

        # 同上
        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    # 创建集合与list
    entity_set = set()
    relation_set = set()
    triple_list = []

    # codecs.open自动转码，打开test
    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            # 三元组,triple[0]和triple[1]均为entity,在entity2id中均能找到对应的编码写入_
            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            # triple list是三元素id的list
            triple_list.append([h_, t_, r_])

            # 三个集合均为编码id的集合
            entity_set.add(h_)
            entity_set.add(t_)
            relation_set.add(r_)

    return entity_set, relation_set, triple_list


# loss function的两个d
def distanceL2(h, r, t):
    # 为方便求梯度，去掉sqrt
    return np.sum(np.square(h + r - t))


def distanceL1(h, r, t):
    return np.sum(np.fabs(h + r - t))


# transE类
class TransE:
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1 = L1

        self.loss = 0

    # 初始化每个向量（随机这里是均匀分布）
    def emb_initialize(self):
        relation_dict = {}
        entity_dict = {}

        # 读取relation_set的每一行
        for relation in self.relation:
            # 均匀分布
            # array(50),-6/sqrt(50) - 6/sqrt(50)
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            # 每一个relation encode
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        # 向量赋给relation和entity
        self.relation = relation_dict
        self.entity = entity_dict

    def train(self, epochs):
        # 设置batch size
        nbatches = 400
        batch_size = len(self.triple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0

            # SGD
            for k in range(nbatches):
                # Sbatch:list(batch_size)   随机不重复抽样
                Sbatch = random.sample(self.triple_list, batch_size)
                Tbatch = []

                # negative sampling
                for triple in Sbatch:
                    # 每个triple选3个负样例
                    # for i in range(3):
                    corrupted_triple = self.Corrupt(triple)
                    # Tbatch中存有正例和负例
                    if (triple, corrupted_triple) not in Tbatch:
                        Tbatch.append((triple, corrupted_triple))
                self.update_embeddings(Tbatch)

            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", self.loss)

            # 保存临时结果
            if epoch % 20 == 0:
                with codecs.open("entity_temp", "w") as f_e:
                    for e in self.entity.keys():
                        f_e.write(e + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("relation_temp", "w") as f_r:
                    for r in self.relation.keys():
                        f_r.write(r + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")

        print("写入文件...")
        with codecs.open("entity_50dim_batch400", "w") as f1:
            for e in self.entity.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open("relation50dim_batch400", "w") as f2:
            for r in self.relation.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        print("写入完成")

    # change head entity or tail entity
    def Corrupt(self, triple):
        corrupted_triple = copy.deepcopy(triple)
        # 随机替换head\tail
        seed = random.random()
        if seed > 0.5:
            # 替换head
            rand_head = triple[0]
            while rand_head == triple[0]:
                rand_head = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[0] = rand_head

        else:
            # 替换tail
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[1] = rand_tail
        return corrupted_triple

    def update_embeddings(self, Tbatch):
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)

        for triple, corrupted_triple in Tbatch:
            # 取copy里的vector累积更新
            h_correct_update = copy_entity[triple[0]]
            t_correct_update = copy_entity[triple[1]]
            relation_update = copy_relation[triple[2]]

            h_corrupt_update = copy_entity[corrupted_triple[0]]
            t_corrupt_update = copy_entity[corrupted_triple[1]]

            # 取原始的vector计算梯度
            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[1]]
            relation = self.relation[triple[2]]

            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[1]]

            if self.L1:
                dist_correct = distanceL1(h_correct, relation, t_correct)
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            # delta d < -m 时跳出
            if err > 0:
                self.loss += err

                grad_pos = 2 * (h_correct + relation - t_correct)
                grad_neg = 2 * (h_corrupt + relation - t_corrupt)

                if self.L1:
                    for i in range(len(grad_pos)):
                        if grad_pos[i] > 0:
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if grad_neg[i] > 0:
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                # head系数为正，减梯度；tail系数为负，加梯度
                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos

                # corrupt项整体为负，因此符号与correct相反
                if triple[0] == corrupted_triple[0]:  # 若替换的是尾实体，则头实体更新两次
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg

                elif triple[1] == corrupted_triple[1]:  # 若替换的是头实体，则尾实体更新两次
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg

                # relation更新两次
                relation_update -= self.learning_rate * grad_pos
                relation_update -= (-1) * self.learning_rate * grad_neg

        # batch norm
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i])
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])

        # 达到批量更新的目的
        self.entity = copy_entity
        self.relation = copy_relation

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)


if __name__ == '__main__':
    file1 = "./FB15k/"
    entity_set, relation_set, triple_list = data_loader(file1)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.01, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=1001)

```