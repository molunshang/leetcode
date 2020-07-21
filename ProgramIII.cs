using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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

        void MergeSortCount(int[] nums, int[] tmp, int[] indexs, int[] count, int start, int end)
        {
            if (start >= end)
            {
                return;
            }

            var mid = (start + end) / 2;
            MergeSortCount(nums, tmp, indexs, count, start, mid);
            MergeSortCount(nums, tmp, indexs, count, mid + 1, end);
            int i = start, j = mid + 1, index = 0;
            if (nums[indexs[mid]] <= nums[indexs[j]])
            {
                return;
            }

            var size = 0; //indexs[i]大于后半段数组中数的数量
            while (i <= mid && j <= end)
            {
                if (nums[indexs[i]] <= nums[indexs[j]])
                {
                    count[indexs[i]] += size;
                    tmp[index++] = indexs[i++];
                }
                else
                {
                    tmp[index++] = indexs[j++];
                    size++;
                }
            }

            while (i <= mid)
            {
                count[indexs[i]] += end - mid;
                tmp[index++] = indexs[i++];
            }

            while (j <= end)
            {
                tmp[index++] = indexs[j++];
            }

            Array.Copy(tmp, 0, indexs, start, index);
        }

        //归并排序统计
        public IList<int> CountSmallerByMergeSort(int[] nums)
        {
            var result = new int[nums.Length];
            var tmp = new int[nums.Length];
            var indexs = Enumerable.Range(0, nums.Length).ToArray();
            MergeSortCount(nums, tmp, indexs, result, 0, nums.Length - 1);
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
                return new IList<int>[] { new int[0] };
            }

            if (root.left == null && root.right == null)
            {
                return new IList<int>[] { new[] { root.val } };
            }

            var paths = new List<IList<int>>();
            BSTSequences(new HashSet<TreeNode>() { root }, paths, new List<int>());
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

        #region 785. 判断二分图

        //https://leetcode-cn.com/problems/is-graph-bipartite/

        #region 回溯（超时）

        bool IsBipartite(int point, int[][] graph, IList<ISet<int>> sets)
        {
            if (point >= graph.Length)
            {
                return true;
            }

            if (point == 0)
            {
                sets[point].Add(point);
                return IsBipartite(point + 1, graph, sets);
            }

            var curSet = graph[point];
            for (var i = 0; i < sets.Count; i++)
            {
                var set = sets[i];
                if (curSet.Intersect(set).Any())
                {
                    continue;
                }

                set.Add(point);
                if (IsBipartite(point + 1, graph, sets))
                {
                    return true;
                }

                set.Remove(point);
            }

            return false;
        }

        public bool IsBipartite(int[][] graph)
        {
            return IsBipartite(0, graph, new ISet<int>[] { new HashSet<int>(), new HashSet<int>() });
        }

        #endregion

        #region BFS染色（点加入setA时，与其相连的点则加入setB，如果应该加入setB/setA的点已经存在setA/setB中时则不可能分割成两个集合）

        public bool IsBipartiteBFS(int[][] graph)
        {
            ISet<int> setA = new HashSet<int>(), setB = new HashSet<int>();
            var queue = new Queue<int>();
            for (int j = 0; j < graph.Length; j++)
            {
                if (setA.Contains(j) || setB.Contains(j))
                {
                    continue;
                }

                queue.Enqueue(j);
                while (queue.Count > 0)
                {
                    var point = queue.Dequeue();
                    if (setB.Contains(point))
                    {
                        return false;
                    }

                    if (!setA.Add(point))
                    {
                        continue;
                    }

                    var next = graph[point];
                    foreach (var p in next)
                    {
                        if (p == point)
                        {
                            continue;
                        }

                        if (setA.Contains(p))
                        {
                            return false;
                        }

                        if (!setB.Add(p))
                        {
                            continue;
                        }

                        foreach (var i in graph[p])
                        {
                            if (i == p)
                            {
                                continue;
                            }

                            queue.Enqueue(i);
                        }
                    }
                }
            }

            return true;
        }

        #endregion

        #endregion

        #region 329. 矩阵中的最长递增路径

        //https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/
        int Max(params int[] args)
        {
            var max = args[0];
            for (int i = 1; i < args.Length; i++)
            {
                max = Math.Max(max, args[i]);
            }

            return max;
        }

        int LongestIncreasingPath(int x, int y, int prev, int[][] matrix, bool flag, int[,,] cache)
        {
            if (x < 0 || x >= matrix.Length || y < 0 || y >= matrix[0].Length)
            {
                return 0;
            }

            if (flag) //升序
            {
                if (matrix[x][y] <= prev)
                {
                    return 0;
                }
            }
            else if (matrix[x][y] >= prev) //降序
            {
                return 0;
            }

            var i = flag ? 0 : 1;
            if (cache[x, y, i] != 0)
            {
                return cache[x, y, i];
            }

            var l1 = LongestIncreasingPath(x - 1, y, matrix[x][y], matrix, flag, cache);
            var l2 = LongestIncreasingPath(x + 1, y, matrix[x][y], matrix, flag, cache);
            var l3 = LongestIncreasingPath(x, y - 1, matrix[x][y], matrix, flag, cache);
            var l4 = LongestIncreasingPath(x, y + 1, matrix[x][y], matrix, flag, cache);
            ////递增/递减 最大
            var count = Max(l1, l2, l3, l4) + 1;
            cache[x, y, i] = count;
            return count;
        }

        public int LongestIncreasingPath(int[][] matrix)
        {
            if (matrix.Length <= 0 || matrix[0].Length <= 0)
            {
                return 0;
            }

            //0 大于路径 
            //1 小于路径
            var res = 1;
            var cache = new int[matrix.Length, matrix[0].Length, 2];
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    int prev = LongestIncreasingPath(i, j, int.MaxValue, matrix, false, cache),
                        next = LongestIncreasingPath(i, j, int.MinValue, matrix, true, cache);
                    res = Math.Max(res, prev + next - 1);
                }
            }

            return res;
        }

        #region 未完成

        //    var dp = new int[matrix.Length, matrix[0].Length, 2];
        //    var res = 1;
        //        for (int i = 0; i<matrix.Length; i++)
        //        {
        //            for (int j = 0; j<matrix[0].Length; j++)
        //            {
        //                if (i == 0)
        //                {
        //                    dp[i, j, 0] = j == 0 ? 1 : (matrix[i][j] > matrix[i][j - 1]? dp[i, j - 1, 0] + 1 : 1);
        //    dp[i, j, 1] = j == 0 ? 1 : (matrix[i][j] < matrix[i][j - 1]? dp[i, j - 1, 1] + 1 : 1);
        //}
        //                else if (j == 0)
        //                {
        //                    dp[i, j, 0] = i == 0 ? 1 : (matrix[i][j] > matrix[i - 1][j]? dp[i - 1, j, 0] + 1 : 1);
        //                    dp[i, j, 1] = i == 0 ? 1 : (matrix[i][j] < matrix[i - 1][j]? dp[i - 1, j, 1] + 1 : 1);
        //                }
        //                else
        //                {
        //                    dp[i, j, 0] = 1;
        //                    dp[i, j, 1] = 1;
        //                    if (matrix[i][j] > matrix[i][j - 1] && matrix[i][j] > matrix[i - 1][j])
        //                    {
        //                        dp[i, j, 0] = Math.Max(dp[i - 1, j, 0], dp[i, j - 1, 0]) + 1;
        //                    }
        //                    else if (matrix[i][j] > matrix[i][j - 1])
        //                    {
        //                        dp[i, j, 0] = dp[i, j - 1, 0] + 1;
        //                    }
        //                    else if (matrix[i][j] > matrix[i - 1][j])
        //                    {
        //                        dp[i, j, 0] = dp[i - 1, j, 0] + 1;
        //                    }

        //                    if (matrix[i][j] < matrix[i][j - 1] && matrix[i][j] < matrix[i - 1][j])
        //                    {
        //                        dp[i, j, 1] = Math.Max(dp[i - 1, j, 1], dp[i, j - 1, 1]) + 1;
        //                    }
        //                    else if (matrix[i][j] < matrix[i][j - 1])
        //                    {
        //                        dp[i, j, 1] = dp[i, j - 1, 1] + 1;
        //                    }
        //                    else if (matrix[i][j] < matrix[i - 1][j])
        //                    {
        //                        dp[i, j, 1] = dp[i - 1, j, 1] + 1;
        //                    }
        //                }
        //                res = Math.Max(res, dp[i, j, 0] + dp[i, j, 1] - 1);
        //            }
        //        }
        //        return res;

        #endregion

        #endregion

        #region 324. 摆动排序 II

        //https://leetcode-cn.com/problems/wiggle-sort-ii/

        #region 回溯暴力解

        bool WiggleSort(int index, int[] nums, Dictionary<int, int> dict, int keyIndex, List<int> keys)
        {
            if (index >= nums.Length)
            {
                return true;
            }

            int s, e;
            if ((index & 1) == 0)
            {
                s = 0;
                e = keyIndex - 1;
            }
            else
            {
                s = keyIndex + 1;
                e = keys.Count - 1;
            }

            while (s <= e)
            {
                var key = keys[s];
                if (dict[key] <= 0)
                {
                    continue;
                }

                nums[index] = key;
                dict[key]--;
                if (WiggleSort(index + 1, nums, dict, s, keys))
                {
                    return true;
                }

                dict[key]++;
                s++;
            }

            return false;
        }

        public void WiggleSortByBacktracking(int[] nums)
        {
            var dict = nums.GroupBy(n => n).ToDictionary(g => g.Key, g => g.Count());
            var keys = dict.Keys.OrderBy(n => n).ToList();
            WiggleSort(0, nums, dict, keys.Count, keys);
        }

        #endregion

        public void WiggleSort(int[] nums)
        {
            var copy = nums.OrderBy(n => n).ToArray();
            int e = copy.Length - 1, mid = e / 2;
            for (int j = 0; j < nums.Length; j += 2)
            {
                nums[j] = copy[mid--];
            }

            for (int j = 1; j < nums.Length; j += 2)
            {
                nums[j] = copy[e--];
            }
        }

        #endregion

        #region 97. 交错字符串

        //https://leetcode-cn.com/problems/interleaving-string/
        bool IsInterleave(int i1, int i2, int i3, string s1, string s2, string s3, bool?[,] cache)
        {
            if (i3 >= s3.Length)
            {
                return i1 >= s1.Length && i2 >= s2.Length;
            }

            if (cache[i1, i2].HasValue)
            {
                return cache[i1, i2].Value;
            }

            var flag = false;
            if (i1 < s1.Length && s1[i1] == s3[i3])
            {
                flag = IsInterleave(i1 + 1, i2, i3 + 1, s1, s2, s3, cache);
            }

            if (!flag && i2 < s2.Length && s2[i2] == s3[i3])
            {
                flag = IsInterleave(i1, i2 + 1, i3 + 1, s1, s2, s3, cache);
            }

            cache[i1, i2] = flag;
            return flag;
        }

        public bool IsInterleave(string s1, string s2, string s3)
        {
            if (s1.Length + s2.Length != s3.Length)
            {
                return false;
            }

            var cache = new bool?[s1.Length + 1, s2.Length + 1];
            return IsInterleave(0, 0, 0, s1, s2, s3, cache);
        }

        public bool IsInterleaveByDp(string s1, string s2, string s3)
        {
            if (s1.Length + s2.Length != s3.Length)
            {
                return false;
            }

            var dp = new bool[s1.Length + 1, s2.Length + 1];
            dp[0, 0] = true;
            for (int i = 0; i <= s1.Length; i++)
            {
                for (int j = 0; j <= s2.Length; j++)
                {
                    var k = i + j - 1; //如果匹配，s1匹配字符串+s2匹配字符数=s3已遍历字符数，所以 i+j-1 为s3的索引
                    if (i > 0)
                    {
                        dp[i, j] = dp[i, j] || dp[i - 1, j] && s1[i - 1] == s3[k];
                    }

                    if (j > 0)
                    {
                        dp[i, j] = dp[i, j] || dp[i, j - 1] && s2[j - 1] == s3[k];
                    }
                }
            }

            return dp[s1.Length, s2.Length];
        }

        #endregion

        #region 268. 缺失数字

        //https://leetcode-cn.com/problems/missing-number/
        public int MissingNumberBySum(int[] nums)
        {
            var total = nums.Length * (nums.Length + 1) / 2;
            return nums.Aggregate(total, (current, n) => current - n);
        }

        #endregion

        #region 1130. 叶值的最小代价生成树

        //https://leetcode-cn.com/problems/minimum-cost-tree-from-leaf-values/
        public int MctFromLeafValues(int[] arr)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region 780. 到达终点

        //https://leetcode-cn.com/problems/reaching-points/

        bool ReachingPoints(int sx, int sy, int tx, int ty, Dictionary<string, bool> cache)
        {
            var key = sx + "," + sy;
            if (cache.TryGetValue(key, out var res))
            {
                return res;
            }

            res = ReachingPoints(sx, sx + sy, tx, ty) || ReachingPoints(sx + sy, sy, tx, ty);
            cache[key] = res;
            return res;
        }

        public bool ReachingPoints(int sx, int sy, int tx, int ty)
        {
            while (tx >= sx && ty >= sy)
            {
                if (tx == ty)
                {
                    break;
                }

                if (tx > ty)
                {
                    if (ty > sy)
                    {
                        tx %= ty;
                    }
                    else
                    {
                        return (tx - sx) % ty == 0;
                    }
                }
                else
                {
                    if (tx > sx)
                    {
                        ty %= tx;
                    }
                    else
                    {
                        return (ty - sy) % tx == 0;
                    }
                }
            }

            return tx == sx && ty == sy;
        }

        #endregion

        #region 821. 字符的最短距离

        //https://leetcode-cn.com/problems/shortest-distance-to-a-character/
        public int[] ShortestToChar(string s, char c)
        {
            var stack = new Stack<int>();
            var res = new int[s.Length];
            var prev = -s.Length;
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == c)
                {
                    while (stack.Count > 0)
                    {
                        res[stack.Peek()] = Math.Min(i - stack.Pop(), i - prev);
                    }

                    prev = i;
                }
                else
                {
                    stack.Push(i);
                }
            }

            while (stack.Count > 0)
            {
                res[stack.Peek()] = stack.Pop() - prev;
            }

            return res;
        }

        #endregion

        #region 779. 第K个语法符号

        //https://leetcode-cn.com/problems/k-th-symbol-in-grammar/
        int KthGrammar(int n, int k, int flag)
        {
            if (n == 1)
            {
                return flag;
            }

            var half = 1 << (n - 2); //上一行长度
            if (k <= half)
            {
                return KthGrammar(n - 1, k, flag);
            }
            return KthGrammar(n - 1, k - half, 1 - flag);
        }

        public int KthGrammar(int n, int k)
        {
            return KthGrammar(n, k, 0);
        }

        #endregion

        #region 724. 寻找数组的中心索引

        //https://leetcode-cn.com/problems/find-pivot-index/
        public int PivotIndex(int[] nums)
        {
            var sum = nums.Sum();
            var prev = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                if (prev == sum - nums[i])
                {
                    return i;
                }

                prev += nums[i];
                sum -= nums[i];
            }
            return -1;
        }

        #endregion

        #region 498. 对角线遍历
        //https://leetcode-cn.com/problems/diagonal-traverse/
        public int[] FindDiagonalOrder(int[][] matrix)
        {
            if (matrix.Length <= 0 || matrix[0].Length <= 0)
            {
                return new int[0];
            }
            int size = matrix.Length * matrix[0].Length;
            int[] res = new int[size];
            int x = 0, y = 0, i = 0;
            var up = true;
            while (i < res.Length)
            {
                res[i] = matrix[x][y];
                if (up)
                {
                    //x-1,y+1 向上
                    bool cx = x == 0, cy = y == matrix[0].Length - 1;
                    if (cx || cy)
                    {
                        up = false;
                        if (x == 0)
                        {
                            if (cy)
                            {
                                x++;
                            }
                            else
                            {
                                y++;
                            }
                        }
                        else
                        {
                            x++;
                        }
                    }
                    else
                    {
                        x--;
                        y++;
                    }
                }
                else
                {
                    //x+1,y-1 向下
                    bool cx = x == matrix.Length - 1, cy = y == 0;
                    if (cx || cy)
                    {
                        up = true;
                        if (y == 0)
                        {
                            if (cx)
                            {
                                y++;
                            }
                            else
                            {
                                x++;
                            }
                        }
                        else
                        {
                            y++;
                        }
                    }
                    else
                    {
                        x++;
                        y--;
                    }
                }
                i++;
            }
            return res;
        }
        #endregion
    }
}