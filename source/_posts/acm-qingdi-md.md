---
title:  清帝之惑之顺治
date: 2021-11-07 19:22:27
tags: ACM
---
## Problem
顺治喜欢滑雪，这并不奇怪， 因为滑雪的确很刺激。可是为了获得速度，滑的区域必须向下倾斜，而且当你滑到坡底，你不得不再次走上坡或者等待太监们来载你。顺治想知道载一个区域中最长的滑坡。

　　区域由一个二维数组给出。数组的每个数字代表点的高度。下面是一个例子：
```
　　 1 2 3 4 5
　　16 17 18 19 6
　　15 24 25 20 7
　　14 23 22 21 8
　　13 12 11 10 9
```

　　顺治可以从某个点滑向上下左右相邻四个点之一，当且仅当高度减小。在上面的例子中，一条可滑行的滑坡为24-17-16-1。当然25-24-23-...-3-2-1更长。事实上，这是最长的一条。

## Input
输入的第一行表示区域的行数 R 和列数 C (1≤R,C≤500) 。下面是 R 行，每行有 C 个整数，代表高度 h,0≤h<103 。

## Output
输出最长区域的长度。

## EX
```
5 5
1 2 3 4 5
16 17 18 19 6
15 24 25 20 7
14 23 22 21 8
13 12 11 10 9
```
```
25
```

## 思路
很典型的深搜题，rc并不算大，O(n3)不会爆，所以直接dfs每个点的最长序列。注意要记忆化每个点的长度（在dfs内记忆化效果更优，因为能保存下更多点的mem值）。

## Code
```cpp
#include<iostream>
#include<cstdio>
using namespace std;

int ans = 0;
int r,c;
int height[502][502];
int mem[502][502] = {0};
int changex[4] = {-1, 1, 0, 0};
int changey[4] = {0, 0, -1, 1};

int dfs(int x, int y){
    if(mem[x][y]){
        return mem[x][y];
    }
    int maxx = 0;
    for(int i = 0;i < 4;i++){
        int xx = x + changex[i];
        int yy = y + changey[i];
        int hh = 0;
        if(height[xx][yy] < height[x][y]){
            hh = dfs(xx,yy);
        }
        hh++;
        if(hh > maxx){
            maxx = hh;
        }
    }
    mem[x][y] = maxx;
    return maxx;
}

int main(){
    cin>>r>>c;
    for(int i = 0;i <= r + 1;i++){
        for(int j = 0;j <= c + 1;j++){
            height[i][j] = 0x3f3f3f3f;
            mem[i][j] = 0;
        }
    }
    for(int i = 1;i <= r;i++){
        for(int j = 1;j <= c;j++){
            cin>>height[i][j];
        }
    }
    for(int i = 1;i <= r;i++){
        for(int j = 1;j <= c;j++){
            int h = dfs(i,j);
            if(h > ans){
                ans = h;
            }
        }
    }
    cout<<ans<<endl;
    return 0;
}
```
