---
title: 加分二叉树
date: 2021-11-07 12:15:41
tags: ACM
---

设一个n个节点的二叉树tree的中序遍历为（l,2,3,…,n），其中数字1,2,3,…,n为节点编号。每个节点都有一个分数（均为正整数），记第i个节点的分数为di，tree及它的每个子树都有一个加分，任一棵子树subtree（也包含tree本身）的加分计算方法如下：

　　subtree的左子树的加分× subtree的右子树的加分＋subtree的根的分数

　　若某个子树为空，规定其加分为1，叶子的加分就是叶节点本身的分数。不考虑它的空子树。

　　试求一棵符合中序遍历为（1,2,3,…,n）且加分最高的二叉树tree。要求输出；

　　（1）tree的最高加分

　　（2）tree的前序遍历

## Input
第 1 行：一个整数 n (n＜30)， 为节点个数。

第 2 行 ：n 个用空格隔开的整数，为每个节点的分数（分数 ＜100）。

## Output
第 1 行：一个整数，为最高加分（结果不会超过4,000,000,000）。

第 2 行 ：n 个用空格隔开的整数，为该树的前序遍历。

若存在多种前序遍历均为最高加分，则输出字典序最小的前序遍历

## EX
5
5 7 1 2 10

145
3 1 2 4 5

## 思路
问题的本质在于寻找到一个最优的二叉树，评估标准是根节点的加分，而跟节点的加分又与它的两个son的加分有关，所以递归方程很好寻找。

可以把中序遍历的序列分为几层，在最外一层寻找最优的根节点，然后在根结点的左右分别找下一层的根结点，每一层的每一个可能根节点的加分都要递归到最后才能算出，所以就能写出来了。

最后要输出前序遍历的序列。这需要用一个变量存下每一层的根结点，然后输出的时候先输出每一层的根结点然后左右层的根结点...以此类推。


## Code
```cpp
#include<iostream>
#include<cstdio>
using namespace std;

int n,a[40],root[40][40];
long long dp[40][40];

long long dfs(int L,int R){
    if(L>R) return 1;       //因为要左右子节点相乘，所以如果是空则给1

    if(dp[L][R]) return dp[L][R];  //关键一步，记忆化，以前走过那就直接返回结果

    long long maxn = 0;
    for(int i=L;i<R;i++){       //递归地寻找加分最大地那个父节点
        long long t = dfs(L,i-1) * dfs(i+1,R) + a[i];   //寻找左侧和右侧地最大父节点对应的加分
        if(t > maxn){
            maxn = t;
            root[L][R] = i;//L-R的最父节点为i
        }
    }
    return dp[L][R] = maxn;
}


void dg(int L,int R){           //前序遍历
    if(L>R)
    {
        return ;
    }
    cout<<root[L][R]<<" ";
    dg(L,root[L][R]-1);
    dg(root[L][R]+1,R);
}

int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        dp[i][i] = a[i];
        root[i][i] = i;
    }
    cout<<dfs(1,n)<<endl;
    dg(1,n);

    return 0;
}
```