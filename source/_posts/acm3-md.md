---
title: 二分查找
date: 2021-07-16 01:10:40
tags: ACM
---

# 剑指 Offer 53 - I. 在排序数组中查找数字 I

统计一个数字在排序数组中出现的次数。

## 示例 1:

    输入: nums = [5,7,7,8,8,10], target = 8
    输出: 2

## 示例 2:

    输入: nums = [5,7,7,8,8,10], target = 6
    输出: 0

0 <= 数组长度 <= 50000

## 思路

啊看上去是个遍历水题数据也不大，但是看了评论才知道如果面试官出这道题其实是要考察二分查找的，所以手法还是要专业。所以就先二分查找到一个target，然后向前向后遍历计数即可。

## 代码

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int ans = 0;
        int n = nums.size();
        if(n == 0){
            return 0;
        }
        int l = 0;
        int r = n - 1;
        while(l < r) {
            int mid = (l + r) / 2;
            if(nums[mid] == target) {
                l = mid;
                r = mid;
                break;
            }
            else if(nums[mid] < target) {
                l = mid + 1;
            }
            else {
                r = mid;
            }
        }
        for(int i = l;i >= 0 && nums[i] == target;i--) {
            ans++;
        }
        for(int i = l + 1;i < n && nums[i] == target;i++) {
            ans++;
        }
        return ans;
    }
};
```