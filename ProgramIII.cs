using System;
using System.Collections.Generic;
using System.Linq;

namespace leetcode
{
    partial class Program
    {
        #region 315. 计算右侧小于当前元素的个数

        //https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/

        int BinaryInsert(int[] nums, int target, int len)
        {
            if (len <= 0)
            {
                nums[0] = target;
                return 0;
            }

            int s = 0, e = len - 1;
            while (s < e)
            {
                var m = (s + e) / 2;
                if (nums[m] >= target)
                {
                    e = m - 1;
                }
                else
                {
                    s = m + 1;
                }
            }

            int index = nums[s] >= target ? s : s + 1;
            if (index < len)
            {
                Array.Copy(nums, index, nums, index + 1, len - index);
            }

            nums[index] = target;
            return index;
        }

        //todo 归并排序性能优化
        public IList<int> CountSmaller(int[] nums)
        {
            var result = new int[nums.Length];
            var copy = new int[nums.Length];
            for (int i = nums.Length - 1, l = 0; i >= 0; i--, l++)
            {
                result[i] = BinaryInsert(copy, nums[i], l);
            }

            return result;
        }

        #endregion

        #region 174. 地下城游戏

        //https://leetcode-cn.com/problems/dungeon-game/
        void CalculateMinimumHP(int x, int y, int[][] dungeon, int sum, int live, ref int result)
        {
            if (live >= result)
            {
                return;
            }

            var res = sum;
            if (sum <= 0)
            {
                res = Math.Abs(sum) + 1;
                live += res;
                res = 1;
            }

            if (x >= dungeon.Length || y >= dungeon[0].Length)
            {
                if ((x == dungeon.Length && y == dungeon[0].Length - 1) ||
                    (x == dungeon.Length - 1 && y == dungeon[0].Length))
                {
                    result = Math.Min(result, live);
                }

                return;
            }

            res += dungeon[x][y];
            CalculateMinimumHP(x + 1, y, dungeon, res, live, ref result);
            CalculateMinimumHP(x, y + 1, dungeon, res, live, ref result);
        }

        int CalculateMinimumHP(int x, int y, int[][] dungeon, int[,] cache)
        {
            if (x == dungeon.Length - 1 && y == dungeon[0].Length - 1)
            {
                return Math.Max(1, 1 - dungeon[x][y]);
            }

            if (cache[x, y] != 0)
            {
                return cache[x, y];
            }

            var num = dungeon[x][y];
            int res;
            if (x == dungeon.Length - 1)
            {
                res = Math.Max(CalculateMinimumHP(x, y + 1, dungeon, cache) - num, 1);
            }
            else if (y == dungeon[0].Length - 1)
            {
                res = Math.Max(CalculateMinimumHP(x + 1, y, dungeon, cache) - num, 1);
            }
            else
            {
                res = Math.Max(1,
                    Math.Min(CalculateMinimumHP(x, y + 1, dungeon, cache),
                        CalculateMinimumHP(x + 1, y, dungeon, cache)) - num);
            }

            cache[x, y] = res;
            return res;
        }

        public int CalculateMinimumHP(int[][] dungeon)
        {
            var cache = new int[dungeon.Length, dungeon[0].Length];
            var live = CalculateMinimumHP(0, 0, dungeon, cache);
            return live;
        }

        #endregion


        #region 面试题 04.09. 二叉搜索树序列

        //https://leetcode-cn.com/problems/bst-sequences-lcci/
        void BSTSequences(ISet<TreeNode> level, IList<IList<int>> result, IList<int> path)
        {
            if (level.Count <= 0)
            {
                result.Add(path.ToArray());
                return;
            }

            //搜索二叉树（需要进行层级遍历）
            var currentLevel = new HashSet<TreeNode>(level);
            foreach (var node in level)
            {
                path.Add(node.val);
                if (node.left != null)
                {
                    currentLevel.Add(node.left);
                }

                if (node.right != null)
                {
                    currentLevel.Add(node.right);
                }

                //遍历当前节点后，下一级与同级节点依旧可以访问，（移除当前节点，遍历下一级与同级其他节点）
                currentLevel.Remove(node);
                BSTSequences(currentLevel, result, path);
                if (node.left != null)
                {
                    currentLevel.Remove(node.left);
                }

                if (node.right != null)
                {
                    currentLevel.Remove(node.right);
                }

                currentLevel.Add(node);
                path.RemoveAt(path.Count - 1);
            }
        }

        public IList<IList<int>> BSTSequences(TreeNode root)
        {
            if (root == null)
            {
                return new IList<int>[] {new int[0]};
            }

            if (root.left == null && root.right == null)
            {
                return new IList<int>[] {new[] {root.val}};
            }

            var paths = new List<IList<int>>();
            BSTSequences(new HashSet<TreeNode>() {root}, paths, new List<int>());
            return paths;
        }

        #endregion

        #region 214. 最短回文串

        //https://leetcode-cn.com/problems/shortest-palindrome/
        //todo 性能优化
        public string ShortestPalindrome(string s)
        {
            var reverseStr = new string(s.Reverse().ToArray());
            for (int i = 0; i < reverseStr.Length; i++)
            {
                if (s.IndexOf(reverseStr.Substring(i, reverseStr.Length - i)) == 0)
                {
                    s = reverseStr.Substring(0, i) + s;
                    break;
                }
            }

            return s;
        }

        #endregion

        #region 1201. 丑数 III

        //https://leetcode-cn.com/problems/ugly-number-iii/
        //二分查找（最大公约数/最大公倍数）
        //解题思路 https://leetcode-cn.com/problems/ugly-number-iii/solution/er-fen-fa-si-lu-pou-xi-by-alfeim/
        public int NthUglyNumber(int n, int a, int b, int c)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region 454. 四数相加 II

        //https://leetcode-cn.com/problems/4sum-ii/
        public int FourSumCount(int[] A, int[] B, int[] C, int[] D)
        {
            var res = 0;
            var exists = new Dictionary<int, int>();
            foreach (var a in A)
            {
                foreach (var b in B)
                {
                    var num = a + b;
                    exists[num] = exists.TryGetValue(num, out var size) ? size + 1 : 1;
                }
            }

            foreach (var c in C)
            {
                foreach (var d in D)
                {
                    var find = 0 - (c + d);
                    if (exists.TryGetValue(find, out var n))
                    {
                        res += n;
                    }
                }
            }

            return res;
        }

        #endregion

        #region 120. 三角形最小路径和

        //https://leetcode-cn.com/problems/triangle/
        public int MinimumTotal(IList<IList<int>> triangle)
        {
            if (triangle.Count <= 0)
            {
                return 0;
            }

            var prev = new List<int>();
            var path = new List<int>();
            for (var i = 0; i < triangle.Count; i++)
            {
                var cur = triangle[i];
                for (var j = 0; j < cur.Count; j++)
                {
                    if (i == 0)
                    {
                        //第一行，只有1个元素
                        path.Add(cur[j]);
                    }
                    else if (j == 0)
                    {
                        //第一列，只有上一行同下标元素
                        path.Add(prev[j] + cur[j]);
                    }
                    else if (j == cur.Count - 1)
                    {
                        //最后一列，只有(i-1,j-1)
                        path.Add(prev[j - 1] + cur[j]);
                    }
                    else
                    {
                        //min((i-1,j) (i-1,j-1))
                        path.Add(Math.Min(prev[j], prev[j - 1]) + cur[j]);
                    }
                }

                var tmp = prev;
                prev = path;
                path = tmp;
                path.Clear();
            }

            path = prev;
            if (path.Count <= 0)
            {
                return 0;
            }

            var res = path[0];
            for (var i = 1; i < path.Count; i++)
            {
                res = Math.Min(res, path[i]);
            }

            return res;
        }

        #endregion
    }
}