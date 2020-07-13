//常数时间插入、删除和获取随机元素
//https://leetcode-cn.com/problems/insert-delete-getrandom-o1/

using System;
using System.Collections.Generic;

namespace leetcode
{
    public class RandomizedSet
    {

        /** Initialize your data structure here. */
        public RandomizedSet()
        {

        }
        private Dictionary<int, int> indexDict = new Dictionary<int, int>();
        private List<int> dataList = new List<int>();
        private Random random = new Random();

        /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
        public bool Insert(int val)
        {
            if (indexDict.ContainsKey(val))
            {
                return false;
            }
            indexDict.Add(val, indexDict.Count);
            dataList.Add(val);
            return true;
        }

        /** Removes a value from the set. Returns true if the set contained the specified element. */
        public bool Remove(int val)
        {
            if (!indexDict.ContainsKey(val))
            {
                return false;
            }
            var last = dataList.Count - 1;
            var rmIndex = indexDict[val];
            if (last != rmIndex)
            {
                //与最后一个交换
                var lastVal = dataList[last];
                dataList[rmIndex] = lastVal;
                indexDict[lastVal] = rmIndex;
            }
            indexDict.Remove(val);
            dataList.RemoveAt(dataList.Count - 1);
            return true;
        }

        /** Get a random element from the set. */
        public int GetRandom()
        {
            return dataList[random.Next(dataList.Count)];
        }
    }
}