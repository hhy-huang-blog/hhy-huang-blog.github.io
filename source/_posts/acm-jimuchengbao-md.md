---
title: 积木城堡
date: 2021-11-06 22:10:40
tags: ACM
---

XC的儿子小XC最喜欢玩的游戏用积木垒漂亮的城堡。城堡是用一些立方体的积木垒成的，城堡的每一层是一块积木。小XC是一个比他爸爸XC还聪明的孩子，他发现垒城堡的时候，如果下面的积木比上面的积木大，那么城堡便不容易倒。所以他在垒城堡的时候总是遵循这样的规则。
小XC想把自己垒的城堡送给幼儿园里漂亮的女孩子们，这样可以增加他的好感度。为了公平起见，他决定把送给每个女孩子一样高的城堡，这样可以避免女孩子们为了获得更漂亮的城堡而引起争执。可是他发现自己在垒城堡的时候并没有预先考虑到这一点。所以他现在要改造城堡。由于他没有多余的积木了，他灵机一动，想出了一个巧妙的改造方案。他决定从每一个城堡中挪去一些积木，使得最终每座城堡都一样高。为了使他的城堡更雄伟，他觉得应该使最后的城堡都尽可能的高。
任务：
请你帮助小XC编一个程序，根据他垒的所有城堡的信息，决定应该移去哪些积木才能获得最佳的效果。

## Input
第一行是一个整数 N (N≤100)， 表示一共有几座城堡。以下 N 行每行是一系列非负整数，用一个空格分隔，按从下往上的顺序依次给出一座城堡中所有积木的棱长。用-1结束。一座城堡中的积木不超过100块，每块积木的棱长不超过100。

## Output
一个整数，表示最后城堡的最大可能的高度。如果找不到合适的方案，则输出 0 。

## EX
2
2 1 –1
3 2 1 –1

3

## 思路
思路就是每次输入一个城堡的高度信息就多一次01背包类型的dp，得到它所能得到的所有高度，保存在dp[i][j]中，这个其实是用一维数组进行dp比较方便，i表示第几个城堡，与dp关系不大，j表示堡垒的最大高度（类比背包size）。
然后对i组堡垒进行取高度的交集，交集中最大的值就是结果。

## Code
```cpp
#include<cstdio>

#include<iostream>
#include<cmath>
#include<string>

using namespace std;

int dp[105][10005],len[105][105],cnt[105];

//len记录每个城堡每块积木的长度，cnt记录每个城堡的积木数
int n,maxh;

bool judge(int x)//判断高度x是否所有城堡都可到达
{
    for(int i=1;i<=n;i++)
        if(dp[i][x]!=x)
            return false;
    return true;
}



int main()
{
    for(int i = 0;i < 105;i++){
        for(int j = 0;j < 10005;j++){
            dp[i][j] = 0;
        }
    }

    cin>>n;
    maxh=0;

    for(int i=1;i<=n;i++) {
        int sum = 0, tlen;
        cnt[i] = 0;
        cin >> tlen;
        while (tlen > 0) {
            cnt[i]++;
            len[i][cnt[i]] = tlen;
            sum += tlen;
            cin >> tlen;
        }
        if (sum > maxh)
        {
            maxh = sum;
        }
        for (int j = 1; j <= cnt[i]; j++){
            for (int k = maxh; k >= len[i][j]; k--) {
                dp[i][k] = max(dp[i][k], dp[i][k - len[i][j]] + len[i][j]);
            }
        }
    }
    for(int i=maxh;i>=0;i--)
    {
        if(judge(i))
        {
            cout<<i<<endl;
            break;
        }
    }
}
```