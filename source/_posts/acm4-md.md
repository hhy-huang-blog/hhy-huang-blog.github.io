---
title: unordered_map
date: 2021-07-18 00:56:32
tags: ACM
---

# Leetcode 10.02. 变位词组

编写一种方法，对字符串数组进行排序，将所有变位词组合在一起。变位词是指字母相同，但排列不同的字符串。

## 实例

    输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
    输出:
    [
    ["ate","eat","tea"],
    ["nat","tan"],
    ["bat"]
    ]

## 思路

我直接暴力😭。

这里学到了C++11里面的unordered_map，其实是一个内部使用hash表结构的关联容器。基本上就是用于快速检索。
它通过key来寻找对应的value，内部无序。
对于iterator，itr.first是key，itr.second是value。

还有一个，今天才知道sort还可以对string内的字符排序...

代码：

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> ans;
        unordered_map<string, vector<string>> hash;
        for(int i = 0; i < strs.size();i++) {
            string tmp = strs[i];
            sort(tmp.begin(), tmp.end());
            hash[tmp].push_back(strs[i]);
        }
        for(auto itr = hash.begin();itr != hash.end();itr++){
            ans.push_back(itr->second);
        }
        return ans;
    }
};
```

