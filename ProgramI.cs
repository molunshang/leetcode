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
        //todo complete
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
        #endregion

        #region 42. 接雨水
        //https://leetcode-cn.com/problems/trapping-rain-water/
        public int Trap(int[] height)
        {
            //暴力解
            var res = 0;
            for (int i = 1; i < height.Length - 1; i++)
            {
                var h = height[i];
                int left = h, right = h;
                for (int l = i - 1; l >= 0; l--)
                {
                    left = Math.Max(left, height[l]);
                }
                for (int r = i + 1; r < height.Length; r++)
                {
                    right = Math.Max(right, height[r]);
                }
                res += Math.Min(left, right) - h;
            }
            return res;
        }

        public int TrapI(int[] height)
        {
            //todo 待完成
            int res = 0, sum = 0;
            for (int i = 1, j = 0; i < height.Length; i++)
            {
                if (height[j] < height[i])
                {
                    res += height[j] * (i - j + 1);
                    res -= sum;
                    sum = 0;
                    j = i;
                }
                else
                {
                    sum += height[i];
                }
            }
            return res - sum;
        }
        #endregion
    }
}