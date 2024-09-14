---
title: Adaboost
date: 2021-12-21 01:27:40
tags: 机器学习
---

## 关于Adaboost

Adaboost算法是针对二分类问题提出的集成学习算法，是boosting类算法最著名的代表。当一个学习器的学习的正确率仅比随机猜测的正确率略高，那么就称它是弱学习器，当一个学习期的学习的正确率很高，那么就称它是强学习器。而且发现弱学习器算法要容易得多，这样就需要将弱学习器提升为强学习器。Adaboost的做法是首先选择一个弱学习器，然后进行多轮的训练，但是每一轮训练过后，都要根据当前的错误率去调整训练样本的权重，让预测正确的样本权重降低，预测错误的样本权重增加，从而达到每次训练都是针对上一次预测结果较差的部分进行的，从而训练出一个较强的学习器。

## 实现

首先展示理论算法描述。

<img src="https://raw.githubusercontent.com/hhy-huang/Image/main/IMG_ED06002BA2CA-1.jpeg">

按照上述步骤开始用代码实现。

首先是初始化样本权重，初始他们的权重都是相等的，都是样本数分之一。代码如下所示。
```python
n_train, n_test = len(X_train), len(X_test)
W = np.ones(n_train) / n_train  # 样本权重初始化
```
然后在规定的训练轮数下，在相应的样本权重下对弱分类器进行训练。
```python
Weak_clf.fit(X_train, Y_train, sample_weight=W)
```
然后在测试集进行预测，并且计算出不正确的样本数。
```python
 # 预测不正确的样本数，计算精度
miss = [int(x) for x in (pred_train_i != Y_train)]
# 在当前权重下计算错误率
miss_w = np.dot(W, miss)
```
下面就是根据预测的结果，对预测正确的样本权重进行削弱，对预测错误的样本权重进行加强，从而对样本对权重进行更新，用于下一次学习器的训练，公式代码如下所示。
```python
# 计算alpha
alpha = 0.5 * np.log(float(1 - miss_w) / float(miss_w + 0.01))
# 权重的系数
factor = [x if x == 1 else -1 for x in miss]
# 更新样本权重
W = np.multiply(W, np.exp([float(x) * alpha for x in factor]))
W = W / sum(W)  # normalization
```
最终输出的H(x)要对每个也测结果乘alhpa然后加入到结果的列表中。
```python
# predict
pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
pred_test = pred_test + np.multiply(alpha, pred_test_i)
```
最后对于列表中大于0 的值认为它预测为标签值1，小于0的值认为它预测为比标签值0。
```python
pred_test = (pred_test > 0) * 1
# pred = (pred > 0) * 1
return pred_test
```
从而完成了Adaboost的一次训练过程。
下面阐述一下我自己实现的十折交叉验证，这里使用的是sklearn的KFold，来对数据集进行十次划分，让每次划分的十个部分轮流做测试集，在同一弱学习器下训练十次，最终预测结果指标是这十次的求和取平均，代码如下所示。
```python
weak_clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
# 十折交叉验证
    acc = []
    pre = []
    rec = []
    f1 = []
    Data = data.copy()
    kf = KFold(n_splits=10, shuffle=True, random_state=0)  # 10折
    for train_index, test_index in tqdm(kf.split(Data)):  # 将数据划分为10折
        train_data = Data[train_index]  # 选取的训练集数据下标
        test_data = Data[test_index]  # 选取的测试集数据下标
        x_train = train_data[:, :8]
        y_train = train_data[:, 8]
        x_test = test_data[:, :8]
        y_test = test_data[:, 8]
        scaler = StandardScaler()  # 标准化转换
        scaler.fit(x_train)  # 训练标准化对象
        x_train = scaler.transform(x_train)
        scaler.fit(x_test)  # 训练标准化对象
        x_test = scaler.transform(x_test)

        pred_test = my_adaboost(weak_clf, x_train, x_test, y_train, y_test, epoch)
        acc.append(accuracy_score(y_test, pred_test))
        pre.append(precision_score(y_test, pred_test))
        rec.append(recall_score(y_test, pred_test))
        f1.append(f1_score(y_test, pred_test))

    # 计算测试集的精度，查准率，查全率，F1
    print("My Adaboost outcome in test set with {} epoch:".format(epoch))
    print("ACC:", sum(acc) / 10)
    print("PRE: ", sum(pre) / 10)
    print("REC: ", sum(rec) / 10)
    print("F1: ", sum(f1) / 10)
```

整体的代码如下：
```python
# 自己实现的adaboost
def my_adaboost(Weak_clf, X_train, X_test, Y_train, Y_test, Epoch):
    """    :param Weak_clf:    :param X_train:    :param X_test:    :param Y_train:    :param Y_test:    :param Epoch:    :return:    """
    n_train, n_test = len(X_train), len(X_test)
    W = np.ones(n_train) / n_train  # 样本权重初始化
    # W = np.ones(n) / n
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    # pred = [np.zeros(n)]
    for i in range(Epoch):
        # 用特定权重训练分类器
        Weak_clf.fit(X_train, Y_train, sample_weight=W)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)
        # pred_i = cross_val_predict(Weak_clf, X, Y, cv=10)
        # 预测不正确的样本数，计算精度
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # 在当前权重下计算错误率
        miss_w = np.dot(W, miss)
        # 计算alpha
        alpha = 0.5 * np.log(float(1 - miss_w) / float(miss_w + 0.01))
        # 权重的系数
        factor = [x if x == 1 else -1 for x in miss]
        # 更新样本权重
        W = np.multiply(W, np.exp([float(x) * alpha for x in factor]))
        W = W / sum(W)  # normalization
        # predict
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        # pred_i = [1 if x == 1 else -1 for x in pred_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_test = pred_test + np.multiply(alpha, pred_test_i)
        pred_train = pred_train + np.multiply(alpha, pred_train_i)
        # pred = pred + np.multiply(alpha, pred_i)
    pred_train = (pred_train > 0) * 1
    pred_test = (pred_test > 0) * 1
    # pred = (pred > 0) * 1
    return pred_test
```