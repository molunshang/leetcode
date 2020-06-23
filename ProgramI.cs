using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 32. 最长有效括号

        //https://leetcode-cn.com/problems/longest-valid-parentheses/
        public int LongestValidParentheses(string s)
        {
            bool IsValid(string str)
            {
                var left = 0;
                foreach (var ch in str)
                {
                    if (ch == ')')
                    {
                        if (left <= 0)
                        {
                            return false;
                        }

                        left--;
                    }
                    else
                    {
                        left++;
                    }
                }

                return left == 0;
            }

            if (string.IsNullOrEmpty(s) || s.Length < 2)
            {
                return 0;
            }

            for (int i = s.Length; i >= 2; i--)
            {
                if ((i & 1) == 1)
                {
                    continue;
                }

                for (int j = 0; j < s.Length - i + 1; j++)
                {
                    var subStr = s.Substring(j, i);
                    if (IsValid(subStr))
                    {
                        return i;
                    }
                }
            }

            return 0;
        }

        public int LongestValidParenthesesByStack(string s)
        {
            if (string.IsNullOrEmpty(s) || s.Length < 2)
            {
                return 0;
            }

            var max = 0;
            var stack = new Stack<int>();
            stack.Push(-1);
            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                if (ch == '(')
                {
                    stack.Push(i);
                }
                else
                {
                    stack.Pop();
                    if (stack.Count <= 0)
                    {
                        stack.Push(i);
                    }
                    else
                    {
                        max = Math.Max(max, i - stack.Peek());
                    }
                }
            }

            return max;
        }

        #endregion

        #region 347. 前 K 个高频元素

        //https://leetcode-cn.com/problems/top-k-frequent-elements/
        public int[] TopKFrequent(int[] nums, int k)
        {
            var len = 0;
            var dict = new Dictionary<int, int>();
            foreach (var num in nums)
            {
                if (!dict.TryGetValue(num, out var count))
                {
                    count = 1;
                }
                else
                {
                    count++;
                }

                dict[num] = count;
                len = Math.Max(len, count);
            }

            var buckets = new IList<int>[len + 1];
            foreach (var kv in dict)
            {
                if (buckets[kv.Value] == null)
                {
                    buckets[kv.Value] = new List<int>();
                }

                buckets[kv.Value].Add(kv.Key);
            }

            var result = new int[k];
            for (int i = buckets.Length - 1; i >= 0 && k > 0; i--)
            {
                if (buckets[i] == null)
                {
                    continue;
                }

                for (int j = 0; j < buckets[i].Count && k > 0; j++, k--)
                {
                    result[k - 1] = buckets[i][j];
                }
            }

            return result;
        }

        #endregion

        #region 378. 有序矩阵中第K小的元素

        //https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/
        //二分法
        public int KthSmallest(int[][] matrix, int k)
        {
            int CountNum(int target)
            {
                int i = matrix.Length - 1, j = 0, cnt = 0;
                while (i >= 0 && j < matrix.Length)
                {
                    if (matrix[i][j] <= target)
                    {
                        cnt = cnt + i + 1;
                        j++;
                    }
                    else
                    {
                        i--;
                    }
                }

                return cnt;
            }

            var n = matrix.Length;
            int left = matrix[0][0], right = matrix[n - 1][n - 1];
            while (left < right)
            {
                var mid = (left + right) / 2;
                var count = CountNum(mid);
                if (count < k)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid;
                }
            }

            return left;
        }

        //优先队列
        // public int kthSmallest(int[][] matrix, int k) {
        //     PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
        //         @Override
        //         public int compare(Integer o1, Integer o2) {
        //         return o2 - o1;
        //     }
        //     });
        //     for (int i = 0; i < matrix.length; i++) {
        //         for (int j = 0; j < matrix[i].length; j++) {
        //             if (queue.size() < k) {
        //                 queue.offer(matrix[i][j]);
        //             } else {
        //                 if (queue.peek() <= matrix[i][j]) {
        //                     break;
        //                 }
        //                 queue.offer(matrix[i][j]);
        //                 queue.poll();
        //             }
        //         }
        //     }
        //     return queue.peek();
        // }

        #endregion

        #region 395. 至少有K个重复字符的最长子串

        //https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/
        public int LongestSubstring(string s, int k)
        {
            var res = 0;
            var bucket = new int[26];
            var len = s.Length;
            for (int l = len; l > 0; l--)
            {
                for (int i = 0; i < s.Length - l + 1; i++)
                {
                    var str = s.Substring(i, l);
                    foreach (var ch in str)
                    {
                        bucket[ch - 'a']++;
                    }

                    if (bucket.Where(n => n > 0).All(n => n >= k))
                    {
                        return l;
                    }

                    Array.Clear(bucket, 0, bucket.Length);
                }
            }

            return res;
        }

        int LongestSubstringPart(string s, int k, int left, int right)
        {
            var len = right - left + 1;
            if (len < k)
            {
                return 0;
            }

            var bucket = new int[26];
            for (int i = left; i <= right; i++)
            {
                bucket[s[i] - 'a']++;
            }

            while (len >= k && bucket[s[left] - 'a'] < k)
            {
                left++;
                len--;
            }

            while (len >= k && bucket[s[right] - 'a'] < k)
            {
                right--;
                len--;
            }

            if (len < k)
            {
                return 0;
            }

            for (int i = left; i <= right; i++)
            {
                if (bucket[s[i] - 'a'] < k)
                {
                    return Math.Max(LongestSubstringPart(s, k, left, i - 1), LongestSubstringPart(s, k, i + 1, right));
                }
            }

            return len;
        }

        public int LongestSubstringByPart(string s, int k)
        {
            return k < 2 ? s.Length : LongestSubstringPart(s, k, 0, s.Length - 1);
        }

        #endregion
    }
}