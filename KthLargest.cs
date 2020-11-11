using System.Collections.Generic;
using System.Linq;

namespace leetcode
{

    #region 703. 数据流中的第 K 大元素
    //https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/
    public class KthLargest
    {
        private int _count;
        private int _k;
        private SortedDictionary<int, int> dict = new SortedDictionary<int, int>();
        public KthLargest(int k, int[] nums)
        {
            _k = k;
            foreach (var n in nums)
            {
                Add(n);
            }
        }

        public int Add(int val)
        {
            if (_count == _k)
            {
                var head = dict.First();
                if (head.Key >= val)
                {
                    return head.Key;
                }
            }
            if (dict.TryGetValue(val, out var count))
            {
                count++;
            }
            else
            {
                count = 1;
            }
            dict[val] = count;
            if (_count == _k)
            {
                var head = dict.First();
                if (head.Value == 1)
                {
                    dict.Remove(head.Key);
                }
                else
                {
                    dict[head.Key] = head.Value - 1;
                }
            }
            else
            {
                _count++;
            }
            return dict.First().Key;
        }
    }
    #endregion
}