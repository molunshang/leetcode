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
    }
}