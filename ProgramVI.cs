﻿using System;
using System.Collections.Generic;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 452. 用最少数量的箭引爆气球

        //https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/
        public int FindMinArrowShots(int[][] points)
        {
            if (points.Length <= 0)
            {
                return 0;
            }

            //排序求相交区间
            Array.Sort(points, Comparer<int[]>.Create((a, b) =>
            {
                if (a[1] == b[1])
                {
                    return 0;
                }

                return a[1] > b[1] ? 1 : -1;
            }));
            var end = points[0][1];
            var count = 1;
            foreach (var ints in points)
            {
                if (ints[0] > end)
                {
                    end = ints[1];
                    count++;
                }
            }

            return count;
        }

        #endregion

        #region 164. 最大间距

        //https://leetcode-cn.com/problems/maximum-gap/
        public int MaximumGap(int[] nums)
        {
            if (nums.Length < 2)
            {
                return 0;
            }

            Array.Sort(nums);
            var res = 0;
            for (int i = 1; i < nums.Length; i++)
            {
                res = Math.Max(res, nums[i] - nums[i - 1]);
            }

            return res;
        }

        #endregion

        #region 493. 翻转对

        //https://leetcode-cn.com/problems/reverse-pairs/
        public int ReversePairsII(int[] nums)
        {
            if (nums.Length < 2)
            {
                return 0;
            }

            var sorted = new int[nums.Length];

            int MergeSortPair(int l, int r)
            {
                if (l >= r)
                {
                    return 0;
                }

                var m = (l + r) / 2;
                var lcount = MergeSortPair(l, m);
                var rcount = MergeSortPair(m + 1, r);
                var count = lcount + rcount;

                for (int j = l, k = m + 1; j <= m; j++)
                {
                    while (k <= r && nums[j] > 2L * nums[k])
                    {
                        k++;
                    }

                    count += k - m - 1;
                }

                int i = 0, lp = l, rp = m + 1;
                while (lp <= m && rp <= r)
                {
                    if (nums[lp] <= nums[rp])
                    {
                        sorted[i++] = nums[lp++];
                    }
                    else
                    {
                        sorted[i++] = nums[rp++];
                    }
                }

                while (lp <= m)
                {
                    sorted[i++] = nums[lp++];
                }

                while (rp <= r)
                {
                    sorted[i++] = nums[rp++];
                }

                Array.Copy(sorted, 0, nums, l, r - l + 1);
                return count;
            }

            return MergeSortPair(0, nums.Length - 1);
        }

        #endregion

        #region 面试题 10.03. 搜索旋转数组

        //https://leetcode-cn.com/problems/search-rotate-array-lcci/
        public int SearchI(int[] arr, int target)
        {
            int l = 0, r = arr.Length - 1;
            while (l <= r)
            {
                if (arr[l] == target)
                {
                    return l;
                }

                var m = (l + r) / 2;
                if (arr[m] == target)
                {
                    r = m - 1;
                }
                else if (arr[l] < arr[m])
                {
                    //[l,m]有序
                    if (arr[l] < target && target < arr[m])
                    {
                        r = m - 1;
                    }
                    else
                    {
                        l = m + 1;
                    }
                }
                else if (arr[l] > arr[m])
                {
                    //[m,r]有序
                    if (target > arr[m] && target <= arr[r])
                    {
                        l = m + 1;
                    }
                    else
                    {
                        r = m - 1;
                    }
                }
                else
                {
                    //无法判断哪部分有序
                    l++;
                }
            }

            return l >= arr.Length || arr[l] != target ? -1 : l;
        }

        #endregion

        #region 274. H 指数

        //https://leetcode-cn.com/problems/h-index/
        public int HIndex(int[] citations)
        {
            if (citations.Length <= 0)
            {
                return 0;
            }

            Array.Sort(citations);
            for (int i = 0; i < citations.Length; i++)
            {
                if (citations[i] >= citations.Length - i)
                {
                    return citations.Length - i;
                }
            }

            return citations.Length;
        }

        public int HIndexOn(int[] citations)
        {
            var refs = new int[citations.Length + 1];
            for (int i = 0; i < citations.Length; i++)
            {
                refs[Math.Min(citations[i], citations.Length)]++;
            }

            var h = citations.Length;
            for (int i = refs[h]; i < h; i += refs[h])
            {
                h--;
            }

            return h;
        }

        #endregion

        #region 321. 拼接最大数

        //https://leetcode-cn.com/problems/create-maximum-number/
        //递归+记忆化
        public int[] MaxNumberByReMem(int[] nums1, int[] nums2, int k)
        {
            int[] Max(int[] arr1, int[] arr2)
            {
                if (arr1.Length > arr2.Length)
                {
                    return arr1;
                }

                if (arr1.Length < arr2.Length)
                {
                    return arr2;
                }

                for (int i = 0; i < arr1.Length; i++)
                {
                    if (arr1[i] > arr2[i])
                    {
                        return arr1;
                    }

                    if (arr1[i] < arr2[i])
                    {
                        return arr2;
                    }
                }

                return arr1;
            }

            var cache = new Dictionary<string, int[]>();

            int[] Dfs(int i, int j, int count)
            {
                if (count <= 0)
                {
                    return new int[0];
                }

                var key = i + "," + j + "," + count;
                if (cache.TryGetValue(key, out var result))
                {
                    return result;
                }

                result = new int[0];
                for (int i1 = i; i1 < nums1.Length; i1++)
                {
                    var tmp = Dfs(i1 + 1, j, count - 1);
                    var tmp1 = new int[tmp.Length + 1];
                    tmp1[0] = nums1[i1];
                    Array.Copy(tmp, 0, tmp1, 1, tmp.Length);
                    result = Max(result, tmp1);
                }

                for (int j1 = j; j1 < nums2.Length; j1++)
                {
                    var tmp = Dfs(i, j1 + 1, count - 1);
                    var tmp2 = new int[tmp.Length + 1];
                    tmp2[0] = nums2[j1];
                    Array.Copy(tmp, 0, tmp2, 1, tmp.Length);
                    result = Max(result, tmp2);
                }

                cache[key] = result;
                return result;
            }

            return Dfs(0, 0, k);
        }

        public int[] MaxNumber(int[] nums1, int[] nums2, int k)
        {
            int[] GetMaxSeq(int[] arr, int n)
            {
                var seq = new int[n];
                int index = -1, leave = arr.Length - n;
                foreach (var num in arr)
                {
                    while (index > -1 && seq[index] < num && leave > 0)//leave表示可丢弃个数，必须取够n个数，当leave==0时必须取数，无论大小
                    {
                        index--;
                        leave--;
                    }

                    if (index < seq.Length - 1)
                    {
                        seq[++index] = num;
                    }
                    else
                    {
                        leave--;
                    }
                }

                return seq;
            }

            int Compare(int[] arr1, int[] arr2, int i1, int i2)
            {
                while (i1 < arr1.Length && i2 < arr2.Length)
                {
                    var diff = arr1[i1] - arr2[i2];
                    if (diff != 0)
                    {
                        return diff;
                    }

                    i1++;
                    i2++;
                }

                return (arr1.Length - i1) - (arr2.Length - i2);
            }

            int[] Merge(int[] arr1, int[] arr2)
            {
                if (arr1.Length <= 0)
                {
                    return arr2;
                }

                if (arr2.Length <= 0)
                {
                    return arr1;
                }

                int len = arr1.Length + arr2.Length, i1 = 0, i2 = 0;
                var sorted = new int[len];
                for (int i = 0; i < len; i++)
                {
                    if (Compare(arr1, arr2, i1, i2) >= 0)
                    {
                        sorted[i] = arr1[i1++];
                    }
                    else
                    {
                        sorted[i] = arr2[i2++];
                    }
                }

                return sorted;
            }

            var res = new int[0];
            //k=nums1取数+nums2取数 min:从nums1可取的最小个数(极端情况下nums2中所有数字全被选择)，max:从nums2中可取的最大个数(极端情况下nums1中所有数字全被选择)
            int min = Math.Max(0, k - nums2.Length), max = Math.Min(k, nums1.Length);
            for (int i = min; i <= max; i++)
            {
                var seq1 = GetMaxSeq(nums1, i);
                var seq2 = GetMaxSeq(nums2, k - i);
                var merge = Merge(seq1, seq2);
                if (Compare(merge, res, 0, 0) > 0)
                {
                    res = merge;
                }
            }

            return res;
        }

        #endregion
    }
}