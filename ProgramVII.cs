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
    }
}