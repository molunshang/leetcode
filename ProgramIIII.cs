using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 面试题 16.16. 部分排序
        //https://leetcode-cn.com/problems/sub-sort-lcci/
        public int[] SubSort(int[] array)
        {
            if (array.Length <= 0)
            {
                return new[] { -1, -1 };
            }
            //1 5 3 7
            int min = int.MaxValue, max = int.MinValue;
            int left = -1, right = -1;
            for (int i = 0, j = array.Length - 1; i < array.Length; i++, j--)
            {
                if (array[i] < max)
                {
                    right = i;
                }
                else
                {
                    max = array[i];
                }
                if (array[j] > min)
                {
                    left = j;
                }
                else
                {
                    min = array[j];
                }

            }
            return new[] { left, right };
        }
        #endregion
    }
}