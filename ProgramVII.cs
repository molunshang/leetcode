using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 304. 二维区域和检索 - 矩阵不可变

        //https://leetcode-cn.com/problems/range-sum-query-2d-immutable/
        class NumMatrix
        {
            private int[,] prefixSums;
            private int[][] matrix;

            public NumMatrix(int[][] matrix)
            {
                this.matrix = matrix;
                if (matrix.Length <= 0 || matrix[0].Length <= 0)
                {
                    return;
                }

                int m = matrix.Length, n = matrix[0].Length;
                prefixSums = new int[m, n];
                for (int i = 0; i < m; i++)
                {
                    var arr = matrix[i];
                    prefixSums[i, 0] = arr[0];
                    for (int j = 1; j < n; j++)
                    {
                        prefixSums[i, j] = prefixSums[i, j - 1] + arr[j];
                    }
                }
            }

            public int SumRegion(int row1, int col1, int row2, int col2)
            {
                if (prefixSums == null)
                {
                    return 0;
                }

                var sum = 0;
                for (int i = row1; i <= row2; i++)
                {
                    sum += prefixSums[i, col2] - prefixSums[i, col1] + matrix[i][col1];
                }

                return sum;
            }
        }

        #endregion

        #region 354. 俄罗斯套娃信封问题

        //https://leetcode-cn.com/problems/russian-doll-envelopes/
        public int MaxEnvelopes(int[][] envelopes)
        {
            if (envelopes.Length < 2)
            {
                return envelopes.Length;
            }

            Array.Sort(envelopes, Comparer<int[]>.Create((a, b) => a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]));
            var dp = new int[envelopes.Length];
            dp[0] = 1;
            var ans = 1;
            for (int i = 1; i < dp.Length; i++)
            {
                var count = 0;
                var big = envelopes[i];
                for (int j = 0; j < i; j++)
                {
                    if (envelopes[j][0] < big[0] && envelopes[j][1] < big[1])
                    {
                        count = Math.Max(count, dp[j]);
                    }
                }

                dp[i] = count + 1;
                ans = Math.Max(ans, dp[i]);
            }

            return ans;
        }

        #endregion

        #region 503. 下一个更大元素 II

        //https://leetcode-cn.com/problems/next-greater-element-ii/
        public int[] NextGreaterElements(int[] nums)
        {
            var res = new int[nums.Length];
            Array.Fill(res, -1);
            var stack = new Stack<int>();
            for (int i = 0, n = nums.Length; i < nums.Length * 2 - 1; i++)
            {
                var num = nums[i % n];
                while (stack.TryPeek(out var j) && nums[j] < num)
                {
                    res[stack.Pop()] = num;
                }

                stack.Push(i % n);
            }

            return res;
        }

        #endregion

        #region 132. 分割回文串 II

        //https://leetcode-cn.com/problems/palindrome-partitioning-ii/
        public int MinCut(string s)
        {
            var dp = new bool[s.Length, s.Length];
            for (int l = 1; l <= s.Length; l++)
            {
                for (int i = 0, j = i + l - 1; j < s.Length; i++, j++)
                {
                    switch (l)
                    {
                        case 1:
                            dp[i, j] = true;
                            break;
                        case 2:
                            dp[i, j] = s[i] == s[j];
                            break;
                        default:
                            dp[i, j] = s[i] == s[j] && dp[i + 1, j - 1];
                            break;
                    }
                }
            }

            var ans = new int[s.Length];
            //判断以i结尾的字符串分割次数
            for (int i = 0; i < s.Length; i++)
            {
                if (dp[0, i])
                {
                    ans[i] = 0;
                }
                else
                {
                    ans[i] = int.MaxValue;
                    for (int j = 0; j < i; j++)
                    {
                        if (dp[j + 1, i])
                        {
                            ans[i] = Math.Min(ans[i], ans[j] + 1);
                        }
                    }
                }
            }

            return ans[s.Length - 1];
        }

        #endregion

        #region 1047. 删除字符串中的所有相邻重复项

        //https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/
        public string RemoveDuplicates(string s)
        {
            var stack = new Stack<char>();
            foreach (var ch in s)
            {
                if (stack.TryPeek(out var h) && h == ch)
                {
                    stack.Pop();
                    continue;
                }

                stack.Push(ch);
            }

            var chars = new char[stack.Count];
            for (int i = chars.Length - 1; i >= 0; i--)
            {
                chars[i] = stack.Pop();
            }

            return new string(chars);
        }

        #endregion

        #region 331. 验证二叉树的前序序列化

        //https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/
        public bool IsValidSerialization(string preorder)
        {
            if (!preorder.EndsWith("#"))
            {
                return false;
            }

            var arr = preorder.Split(',');
            var stack = new Stack<string>();
            for (int i = 0; i < arr.Length; i++)
            {
                var str = arr[i];
                if (str == "#")
                {
                    if (stack.TryPop(out _))
                    {
                        //最后一个栈能弹出说明格式有问题
                        if (i == arr.Length - 1)
                        {
                            return false;
                        }
                    }
                    else if (i != arr.Length - 1) //最后一个栈应该已经空了
                    {
                        return false;
                    }
                }
                else
                {
                    stack.Push(str);
                }
            }

            return stack.Count <= 0;
        }

        public bool IsValidSerializationByLeetcode(string preorder)
        {
            if (!preorder.EndsWith("#"))
            {
                return false;
            }

            var slots = 1;
            for (int i = 0; i < preorder.Length; i++)
            {
                if (slots == 0)
                {
                    return false;
                }

                if (preorder[i] == ',')
                {
                    continue;
                }

                if (preorder[i] == '#')
                {
                    slots--;
                }
                else
                {
                    while (i < preorder.Length - 1 && char.IsDigit(preorder[i + 1]))
                    {
                        i++;
                    }

                    slots++;
                }
            }

            return slots == 0;
        }

        #endregion

        #region 456. 132 模式
        //https://leetcode-cn.com/problems/132-pattern/
        //leetcode题解，单调栈
        public bool Find132pattern(int[] nums)
        {
            var stack = new Stack<int>();
            var num2 = int.MinValue;
            for (int i = nums.Length - 1; i >= 0; i--)
            {
                if (nums[i] < num2)
                {
                    return true;
                }
                while (stack.TryPeek(out var h) && nums[i] > h)
                {
                    num2 = stack.Pop();
                }
                if (nums[i] > num2)
                {
                    stack.Push(nums[i]);
                }
            }
            return false;
        }
        #endregion
    }
}