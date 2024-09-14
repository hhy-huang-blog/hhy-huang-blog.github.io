---
title: Warcraft III 守望者的烦恼 
date: 2021-11-07 00:19:37
tags: ACM
---

头脑并不发达的warden最近在思考一个问题，她的闪烁技能是可以升级的，k级的闪烁技能最多可以向前移动k个监狱，一共有n个监狱要视察，她从入口进去，一路上有n个监狱，而且不会往回走，当然她并不用每个监狱都视察，但是她最后一定要到第n个监狱里去，因为监狱的出口在那里，但是她并不一定要到第1个监狱。
守望者warden现在想知道，她在拥有k级闪烁技能时视察n个监狱一共有多少种方案？

## Input
第一行是闪烁技能的等级 k (1≤k≤10)
第二行是监狱的个数 n (1≤n≤231−1)

## Output
由于方案个数会很多，所以输出它 mod 7777777后的结果就行了

## EX
2
4

5

## 思路
这道题的状态转移很好找到达第n个监狱的方案数位到达n-1,n-2....n-k个监狱的方案数之和，至于为什么到n-k，因为监狱编号再小是无法通过一步直接到n的，也需要先到n-1~n-k中的其中一个再到n，考虑进来就重复了，他们只需要考虑在到达第n-1~n-k个监狱就行了。

但是有个问题是这个数太大了，直接递归恐怕爆了。所以这里引入了矩阵乘法解决递推问题。依据是：

<a href="https://www.codecogs.com/eqnedit.php?latex=[a_k,&space;a_{k-1},&space;a_{k&space;-&space;2},a_{k-3}&space;]*&space;\begin{bmatrix}&space;1&space;&1&space;&0&space;&0&space;\\&space;1&space;&0&space;&1&space;&0&space;\\&space;1&space;&0&space;&0&space;&1&space;\\&space;1&space;&0&space;&0&space;&0&space;\end{bmatrix}&space;=&space;[a_{k&plus;1},&space;a_{k-1},&space;a_{k&space;-&space;2},a_{k-3}&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[a_k,&space;a_{k-1},&space;a_{k&space;-&space;2},a_{k-3}&space;]*&space;\begin{bmatrix}&space;1&space;&1&space;&0&space;&0&space;\\&space;1&space;&0&space;&1&space;&0&space;\\&space;1&space;&0&space;&0&space;&1&space;\\&space;1&space;&0&space;&0&space;&0&space;\end{bmatrix}&space;=&space;[a_{k&plus;1},&space;a_{k-1},&space;a_{k&space;-&space;2},a_{k-3}&space;]" title="[a_k, a_{k-1}, a_{k - 2},a_{k-3} ]* \begin{bmatrix} 1 &1 &0 &0 \\ 1 &0 &1 &0 \\ 1 &0 &0 &1 \\ 1 &0 &0 &0 \end{bmatrix} = [a_{k+1}, a_{k-1}, a_{k - 2},a_{k-3} ]" /></a>

然后再将得到的结果乘这个矩阵，就可以得到an，也就是想要的结果。

矩阵乘法好写，但是多个矩阵相乘这里直接用的快速幂。


## Code
```cpp
#include<cstdio>
#include<iostream>
#include<cmath>
#include<string>
using namespace std;

#define mod 7777777

int k;
long long n;

struct Matrix{
    int r,c;
    long long M[11][11];
    void init(int r, int c){
        this->r = r;
        this->c = c;
        for(int i = 0;i < r;i++){
            for(int j = 0;j < c;j++){
                M[i][j] = 0;
            }
        }
    }

    Matrix operator*(Matrix& B) const{      //重载矩阵乘法
        Matrix A = *this;
        Matrix C;
        C.init(A.r,B.c);

        for(int i = 0;i <C.r;i++){
            for(int j = 0;j < C.c;j++){
                for(int q = 0;q < A.c;q++){
                    C.M[i][j] = (C.M[i][j] + A.M[i][q] * B.M[q][j]) % mod;
                }
            }
        }
        return C;
    }

    Matrix Q_pow(long long p){            //矩阵快速幂
        Matrix tmp = *this;
        Matrix ans;
        ans.init(this->r, this->r);
        for(int i = 0;i < this->r;i++){     //单位矩阵
            ans.M[i][i] = 1;
        }
        while(p){
            if(p & 1){
                ans = ans * tmp;
            }
            tmp = tmp * tmp;
            p >>= 1;
        }
        return ans;
    }
};


int main()
{
    cin>>k;
    cin>>n;

    Matrix factor;
    Matrix base;
    factor.init(k,k);
    for(int i = 0;i < k;i++){
        factor.M[i][0] = 1;
    }
    for(int i = 1;i <= k;i++){
        factor.M[i - 1][i] = 1;
    }
    factor = factor.Q_pow(n);
    base.init(1,k);
    base.M[0][0] = 1;
    base = base * factor;
    cout<<base.M[0][0]<<endl;
}

```