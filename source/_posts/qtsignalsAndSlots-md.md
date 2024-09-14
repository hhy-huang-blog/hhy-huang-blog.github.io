---
title: QT中的Signal和Slots
date: 2021-07-20 00:18:23
tags: QT
---

## 自定义信号和槽函数的同时重载

记录一下，省得一周的小学期又白上.....

Signal和Slots相当交互中一个事件触发的条件与结果。而在QT中，条件与结果常常是某一class的对象中的某个函数，所以使用```connection(arg1, arg2, arg3, arg4)```来连接二者，一旦arg1对象下的arg2操作实现，则arg3对象下的arg4操作进行。

通常分别建立两个head和cpp来实现这两个对象的构建。

<img src="https://raw.githubusercontent.com/hhy-huang/Image/main/1.png"  height="240" width="189">

像这样，其中hello.h是声明的signal类，world.h是声明的slots类。分别如下：

hello.h
```cpp
#ifndef HELLO_H
#define HELLO_H

#include <QObject>
#include <QString>

class hello : public QObject
{
    Q_OBJECT
public:
    explicit hello(QObject *parent = nullptr);

signals:
    void CallWorld();
    void CallWorld(QString mood);   //overload
public slots:

};

#endif // HELLO_H
```

world.h
```cpp
#ifndef WORLD_H
#define WORLD_H

#include <QObject>
#include <QString>

class world : public QObject
{
    Q_OBJECT
public:
    explicit world(QObject *parent = nullptr);

signals:


public slots:
    void ReceiveHello();
    void ReceiveHello(QString mood);    //overload
};

#endif // WORLD_H
```
同时也要在主类中去声明这俩对象。
```cpp
private:
    Ui::SignalAndSlot *ui;
    hello *he;
    world *wo;
    void Ready();
```

下面的Ready函数是用来触发条件的。里面一般是调用signal的函数。当然这个函数的实现应该是在cpp中。像这样：

```cpp
void SignalAndSlot::Ready()
{
    //emit he->CallWorld();
    emit he->CallWorld("happy");
}
```
至于这个emit，很迷，是个发射signal指令？不懂。

然后就可以在这个cpp文件中进行connection了。注意这里需要重载，重载的操作使用的是函数指针来让connect明确要调用的是哪一个函数。

```cpp
SignalAndSlot::SignalAndSlot(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::SignalAndSlot)
{
    ui->setupUi(this);
    this->he = new hello(this);
    this->wo = new world(this);
    //connect(he, &hello::CallWorld, wo, &world::ReceiveHello);
    //overload
    void(hello::*helloSignal)(QString) = &hello::CallWorld;
    void(world::*worldSlots)(QString) = &world::ReceiveHello;
    connect(he, helloSignal, wo, worldSlots);
    Ready();
}
```
另外别忘了include头文件。connect完了这个worldSlots就处于一个待触发的状态了。一旦Ready()触发，它就被调用。

至于```word.cpp```和```hello.cpp```，当然是用来写signal和slots函数了，光在.h声明了也得实现呢。

还有，一直没懂这个```main.cpp```以外的cpp文件是咋执行的，仔细一想鹊食，其它cpp都是对构造函数的定义，然后main中直接去调用它们的头文件就可以去调用某个类中的函数。原来是这样，我是笨蛋。

总结一下就是基本都是用类对象及其构造函数操作的，找到点感觉了。