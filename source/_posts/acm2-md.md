---
title: 双指针不重复搜索
date: 2021-07-14 23:25:19
tags: ACM
---
# Leetcode 15.三数之和

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

## Example：
1

    输入：nums = [-1,0,1,2,-1,-4]
    输出：[[-1,-1,2],[-1,0,1]]

2

    输入：nums = []
    输出：[]

3

    输入：nums = [0]
    输出：[]

## 思路

固定三元组的一个元素，然后在剩余的元素中选出两个符合要求的元素。在这个搜索过程中，为了避免重复，先对nums进行一个sort，对于i（第一个元素）之后的元素进行双指针搜索，通过x+y+z的结果与0的大小关系来决定是left右移还是right左移。

在避免重复方面，首先对于后两个元素的搜索，left值的重复会导致重复，right值的重复也会导致重复，因此让left或right继续移动直到没有重复，即可避免。然后是第一个元素，同样的方法即可。

再有注意特殊情况，大小小于3的数组和最小值都>0的数组，输出为空。

## 代码
```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        if(nums.size() < 3){//少于三个
            return ans;
        }
        sort(nums.begin(),nums.end());
        if(nums[0] > 0) {//最小的都>0
            return ans;
        }
        int i = 0;
        while(i < nums.size()) {
            if(nums[i] > 0) {//不会存在该解
                break;
            }
            //双指针
            int left = i + 1;
            int right = nums.size() - 1;
            while(left < right) {
                long long y = static_cast<long long>(nums[i]);
                long long x = static_cast<long long>(nums[left]);
                long long z = static_cast<long long>(nums[right]);

                if(x + y > 0 - z){
                    right--;
                } //z要减小
                else if(x + y < 0 - z){
                    left++;
                }//x要增加
                else{
                    ans.push_back({nums[i], nums[left], nums[right]});
                    //处理特殊情况，出现相同的left或right值要跳过
                    while(left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while(left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                }
            }
            //处理特殊情况，出现相同的i的值
            while(i + 1 < nums.size() && nums[i] == nums[i + 1]) {
                i++;
            }
            i++;
        }
        return ans;
    }
};
```