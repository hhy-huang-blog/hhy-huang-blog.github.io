---
title: 贝格方法计算椭圆周长
date: 2021-06-12 21:13:39
tags: 数值计算
---

### 椭圆周长定积分公式
由于椭圆的周长可以看作是很多$\Delta x$与$\Delta y$直角边构成的斜边的和。因此就是

<a target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{dx^2&plus;dy^2}" title="\sqrt{dx^2+dy^2}" /></a>

，此处为了简化直接用参数方程替换，就是


<a target="_blank"><img src="https://latex.codecogs.com/gif.latex?4\times&space;\int_{0}^{&space;\frac{\pi}{2}}&space;\sqrt{a^2&space;sin\theta&space;&plus;&space;b^2&space;cos\theta}&space;d\theta" title="4\times \int_{0}^{ \frac{\pi}{2}} \sqrt{a^2 sin\theta + b^2 cos\theta} d\theta" /></a>

### 龙贝格积分法Matlab代码
```Matlab
function Romberg(fun,a,b,tol)
M = 1;      %每次的步数
k = 0;      %积分表的行
h = b - a;  %最大步长
tol1 = 1;
R = zeros(10,10); %分配矩阵大小
R(1,1) = h*(feval(fun,a) + feval(fun,b))/2; %第一个值
while tol1 >= tol
    k = k + 1;
    h = h/2;
    tmp = 0;
    %一列中上下行的关系
    for i = 1:M
        tmp = tmp + fun(a + h*(2*i - 1));
    end
    R(k+1,1) = R(k,1)/2 + h*tmp;
    %更新步数
    M = 2*M;
    %构造在同一行中，左右列元素的关系
    for m = 1:min(k,3)
        R(k + 1,m + 1) = R(k+1,m)+(R(k+1,m)-R(k,m))/(4^m-1);
    end
    %计算第四列的龙贝格的误差
    tol1=abs(R(k,min(k,4))-R(k+1,min(k,4)));
end
q = R(k+1, 4)
R
```

### 命令
此处针对a = 20，b = 10的椭圆方程而言。
```
>> a = 0;
>> b = pi/2;
>> f = @(x)4*sqrt(400.*sin(x).*sin(x)+100.*cos(x).*cos(x));
>> tol = 1e-4;
>> Romberg(f,a,b,tol);
````