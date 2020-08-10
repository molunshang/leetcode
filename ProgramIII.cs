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
            return IsBipartite(0, graph, new ISet<int>[] {new HashSet<int>(), new HashSet<int>()});
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

        #region 找出(x,y)的递增递减路径相加计算

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


        public int LongestIncreasingPathByPrevNext(int[][] matrix)
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

        #endregion

        int LongestIncreasingPath(int x, int y, int[][] matrix, int prev, int[,] cache)
        {
            if (x < 0 || x >= matrix.Length || y < 0 || y >= matrix[0].Length || matrix[x][y] <= prev)
            {
                return 0;
            }

            if (cache[x, y] != 0)
            {
                return cache[x, y];
            }

            var l1 = LongestIncreasingPath(x, y + 1, matrix, matrix[x][y], cache);
            var l2 = LongestIncreasingPath(x, y - 1, matrix, matrix[x][y], cache);
            var l3 = LongestIncreasingPath(x + 1, y, matrix, matrix[x][y], cache);
            var l4 = LongestIncreasingPath(x - 1, y, matrix, matrix[x][y], cache);
            var res = Max(l1, l2, l3, l4) + 1;
            cache[x, y] = res;
            return res;
        }

        public int LongestIncreasingPath(int[][] matrix)
        {
            if (matrix.Length <= 0 || matrix[0].Length <= 0)
            {
                return 0;
            }

            var res = 1;
            var cache = new int[matrix.Length, matrix[0].Length];
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    res = Math.Max(res, LongestIncreasingPath(i, j, matrix, int.MinValue, cache));
                }
            }

            return res;
        }

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


        #region 5. 最长回文子串

        //https://leetcode-cn.com/problems/longest-palindromic-substring/
        //暴力解
        public string LongestPalindrome(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return string.Empty;
            }

            bool Check(int start, int end)
            {
                while (start < end)
                {
                    if (s[start] != s[end])
                    {
                        return false;
                    }

                    start++;
                    end--;
                }

                return true;
            }

            int l = 0, len = 0;
            for (var i = 0; i < s.Length; i++)
            {
                for (var j = i; j < s.Length; j++)
                {
                    if (len >= (j - i) + 1)
                    {
                        continue;
                    }

                    if (Check(i, j))
                    {
                        l = i;
                        len = j - i + 1;
                    }
                }
            }

            return s.Substring(l, len);
        }

        //动态规划
        public string LongestPalindromeByDp(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return string.Empty;
            }

            var dp = new bool[s.Length, s.Length];
            int start = 0, len = 0;
            for (int l = 1; l <= s.Length; l++)
            {
                for (int i = 0, j = i + l - 1; j < s.Length; i++, j++)
                {
                    if (l == 1)
                    {
                        dp[i, j] = true;
                    }
                    else if (l == 2)
                    {
                        dp[i, j] = s[i] == s[j];
                    }
                    else
                    {
                        dp[i, j] = dp[i + 1, j - 1] && s[i] == s[j];
                    }

                    if (dp[i, j] && len < l)
                    {
                        len = l;
                        start = i;
                    }
                }
            }

            return s.Substring(start, len);
        }

        //中心搜索
        public string LongestPalindromeByCenterSearch(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return string.Empty;
            }

            int CenterSearch(int l, int r)
            {
                while (l >= 0 && r < s.Length && s[l] == s[r])
                {
                    l--;
                    r++;
                }

                return r - l - 1;
            }

            int start = 0, len = 0;
            for (int i = 0; i < s.Length; i++)
            {
                var l = Math.Max(CenterSearch(i, i), CenterSearch(i, i + 1));
                if (l > len)
                {
                    start = i - (l - 1) / 2;
                    len = l;
                }
            }

            return s.Substring(start, len);
        }

        #endregion

        public int StrStr(string haystack, string needle)
        {
            if (string.IsNullOrEmpty(needle))
            {
                return 0;
            }

            if (haystack.Length < needle.Length)
            {
                return -1;
            }

            return haystack.IndexOf(needle);
        }

        #region 27. 移除元素

        //https://leetcode-cn.com/problems/remove-element/
        public int RemoveElement(int[] nums, int val)
        {
            int fast = 0, slow = 0;
            while (fast < nums.Length)
            {
                if (nums[fast] != val)
                {
                    nums[slow] = nums[fast];
                    slow++;
                }

                fast++;
            }

            return slow;
        }

        #endregion

        #region 485. 最大连续1的个数

        //https://leetcode-cn.com/problems/max-consecutive-ones/
        public int FindMaxConsecutiveOnes(int[] nums)
        {
            var len = 0;
            int fast = 0, slow = 0;
            while (fast < nums.Length)
            {
                if (nums[fast] == 0)
                {
                    len = Math.Max(fast - slow, len);
                    slow = fast + 1;
                }

                fast++;
            }

            if (slow < fast)
            {
                len = Math.Max(fast - slow, len);
            }

            return len;
        }

        #endregion

        #region 26. 删除排序数组中的重复项

        //https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/
        public int RemoveDuplicates(int[] nums)
        {
            if (nums.Length <= 1)
            {
                return nums.Length;
            }

            int fast = 1, slow = 1;
            while (fast < nums.Length)
            {
                if (nums[fast - 1] != nums[fast])
                {
                    nums[slow] = nums[fast];
                    slow++;
                }

                fast++;
            }

            return slow;
        }

        #endregion

        #region 410. 分割数组的最大值

        //https://leetcode-cn.com/problems/split-array-largest-sum/
        int SplitArray(int[] nums, int index, int m, int[,] cache)
        {
            if (cache[index, m] != 0)
            {
                return cache[index, m];
            }

            if (m == 1)
            {
                var res = nums.Skip(index).Sum();
                cache[index, m] = res;
                return res;
            }

            //n 个子数组中最大值的最小值
            int sum = 0, min = int.MaxValue;
            for (int i = index, l = nums.Length - index; i < nums.Length && l >= m; i++, l--)
            {
                sum += nums[i];
                var cur = Math.Max(sum, SplitArray(nums, i + 1, m - 1, cache));
                min = Math.Min(min, cur);
            }

            cache[index, m] = min;
            return min;
        }

        public int SplitArray(int[] nums, int m)
        {
            var subMax = SplitArray(nums, 0, m, new int[nums.Length, m + 1]);
            return subMax;
        }

        #endregion

        #region 1. 两数之和

        //https://leetcode-cn.com/problems/two-sum/
        public int[] TwoSumI(int[] nums, int target)
        {
            var dict = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                var find = target - nums[i];
                if (dict.TryGetValue(find, out var index))
                {
                    return new[] {index, i};
                }

                dict[nums[i]] = i;
            }

            return new int[0];
        }

        #endregion

        #region LCP 13. 寻宝

        //https://leetcode-cn.com/problems/xun-bao/
        public int MinimalSteps(string[] maze)
        {
            int m = maze.Length, n = maze[0].Length;
            var steps = new[] {(1, 0), (-1, 0), (0, 1), (0, -1)};

            void FillArray(int[,] arr, int val)
            {
                for (int i = 0; i < arr.GetLength(0); i++)
                {
                    for (int j = 0; j < arr.GetLength(1); j++)
                    {
                        arr[i, j] = val;
                    }
                }
            }

            int[,] BfsFill(int x, int y)
            {
                var res = new int[m, n];
                FillArray(res, -1);
                res[x, y] = 0;
                var queue = new Queue<int[]>();
                queue.Enqueue(new[] {x, y});
                while (queue.Count > 0)
                {
                    var cur = queue.Dequeue();
                    int cx = cur[0], cy = cur[1];
                    foreach (var step in steps)
                    {
                        int nx = cx + step.Item1, ny = cy + step.Item2;
                        if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] != '#' && res[nx, ny] == -1)
                        {
                            res[nx, ny] = res[cx, cy] + 1;
                            queue.Enqueue(new[] {nx, ny});
                        }
                    }
                }

                return res;
            }

            int sx = -1, sy = -1, tx = -1, ty = -1;
            List<int[]> mPoints = new List<int[]>(), oPoints = new List<int[]>();
            for (var i = 0; i < maze.Length; i++)
            {
                var str = maze[i];
                for (var j = 0; j < str.Length; j++)
                {
                    if (str[j] == 'M')
                    {
                        mPoints.Add(new[] {i, j});
                    }
                    else if (str[j] == 'O')
                    {
                        oPoints.Add(new[] {i, j});
                    }
                    else if (str[j] == 'S')
                    {
                        sx = i;
                        sy = j;
                    }
                    else if (str[j] == 'T')
                    {
                        tx = i;
                        ty = j;
                    }
                }
            }

            //从起点到其他节点的最短距离
            var startDist = BfsFill(sx, sy);
            if (mPoints.Count <= 0)
            {
                return startDist[tx, ty];
            }

            int nb = mPoints.Count, ns = oPoints.Count;

            var mDist = new int[mPoints.Count][,]; //记录每个机关到其他节点的最短距离
            var mDistinct = new int[mPoints.Count, mPoints.Count + 2]; ////从机关到机关和起点终点的最短距离
            FillArray(mDistinct, -1);
            for (int i = 0; i < mPoints.Count; i++)
            {
                var point = mPoints[i];
                var dist = BfsFill(point[0], point[1]);
                mDist[i] = dist;
                mDistinct[i, mPoints.Count + 1] = dist[tx, ty];
            }

            for (int i = 0; i < mPoints.Count; i++)
            {
                int tmp = -1;
                for (int k = 0; k < oPoints.Count; k++)
                {
                    var point = oPoints[k];
                    int midX = point[0], midY = point[1];
                    if (mDist[i][midX, midY] != -1 && startDist[midX, midY] != -1)
                    {
                        if (tmp == -1 || tmp > mDist[i][midX, midY] + startDist[midX, midY])
                        {
                            tmp = mDist[i][midX, midY] + startDist[midX, midY];
                        }
                    }
                }

                mDistinct[i, mPoints.Count] = tmp;
                for (int j = i + 1; j < mPoints.Count; j++)
                {
                    int mn = -1;
                    for (int k = 0; k < oPoints.Count; k++)
                    {
                        var oPoint = oPoints[k];
                        int midX = oPoint[0], midY = oPoint[1];
                        if (mDist[i][midX, midY] != -1 && mDist[j][midX, midY] != -1)
                        {
                            if (mn == -1 || mn > mDist[i][midX, midY] + mDist[j][midX, midY])
                            {
                                mn = mDist[i][midX, midY] + mDist[j][midX, midY];
                            }
                        }
                    }

                    mDistinct[i, j] = mn;
                    mDistinct[j, i] = mn;
                }
            }

            // 无法达成的情形
            for (int i = 0; i < nb; i++)
            {
                if (mDistinct[i, nb] == -1 || mDistinct[i, nb + 1] == -1)
                {
                    return -1;
                }
            }

            // dp 数组， -1 代表没有遍历到
            var dp = new int[1 << nb, nb];
            FillArray(dp, -1);

            for (int i = 0; i < nb; i++)
            {
                dp[1 << i, i] = mDistinct[i, nb];
            }

            // 由于更新的状态都比未更新的大，所以直接从小到大遍历即可
            for (int mask = 1; mask < (1 << nb); mask++)
            {
                for (int i = 0; i < nb; i++)
                {
                    // 当前 dp 是合法的
                    if ((mask & (1 << i)) != 0)
                    {
                        for (int j = 0; j < nb; j++)
                        {
                            // j 不在 mask 里
                            if ((mask & (1 << j)) == 0)
                            {
                                int next = mask | (1 << j);
                                if (dp[next, j] == -1 || dp[next, j] > dp[mask, i] + mDistinct[i, j])
                                {
                                    dp[next, j] = dp[mask, i] + mDistinct[i, j];
                                }
                            }
                        }
                    }
                }
            }

            int ret = -1;
            int finalMask = (1 << nb) - 1;
            for (int i = 0; i < nb; i++)
            {
                if (ret == -1 || ret > dp[finalMask, i] + mDistinct[i, nb + 1])
                {
                    ret = dp[finalMask, i] + mDistinct[i, nb + 1];
                }
            }

            return ret;
        }

        #endregion

        #region 343. 整数拆分

        //https://leetcode-cn.com/problems/integer-break/
        private Dictionary<int, int> intBreakCache = new Dictionary<int, int>();

        public int IntegerBreak(int n)
        {
            if (n <= 2)
            {
                return 1;
            }

            if (intBreakCache.TryGetValue(n, out var res))
            {
                return res;
            }

            for (int i = 2; i < n; i++)
            {
                res = Math.Max(res, Math.Max(i * (n - i), i * IntegerBreak(n - i)));
            }

            intBreakCache[n] = res;
            return res;
        }

        #endregion

        #region 7. 整数反转

        //https://leetcode-cn.com/problems/reverse-integer/
        public int Reverse(int x)
        {
            long res = 0;
            while (x != 0)
            {
                res = res * 10 + (x % 10);
                if (res > int.MaxValue || res < int.MinValue)
                {
                    return 0;
                }

                x /= 10;
            }

            return (int) res;
        }

        #endregion

        #region 8. 字符串转换整数 (atoi)

        //https://leetcode-cn.com/problems/string-to-integer-atoi/
        public int MyAtoi(string str)
        {
            int l = 0, r = str.Length - 1;
            while (l <= r && str[l] == ' ')
            {
                l++;
            }

            while (l <= r && str[r] == ' ')
            {
                r--;
            }

            if (l > r)
            {
                return 0;
            }

            long res = 0;
            var flag = true;
            if (str[l] == '-' || str[l] == '+')
            {
                flag = str[l] == '+';
                l++;
            }

            while (l <= r)
            {
                var ch = str[l];
                if (char.IsDigit(ch))
                {
                    res = res * 10 + (ch - '0');
                    if (res > int.MaxValue)
                    {
                        return flag ? int.MaxValue : int.MinValue;
                    }
                }
                else
                {
                    break;
                }

                l++;
            }

            return flag ? (int) res : -(int) res;
        }

        #endregion

        #region 面试题 08.03. 魔术索引

        //https://leetcode-cn.com/problems/magic-index-lcci/
        int FindMagicIndex(int[] nums, int l, int r)
        {
            while (l < r)
            {
                var mid = (l + r) / 2;
                //降序/相同数组
                if (nums[l] >= nums[r])
                {
                    if (nums[mid] == mid)
                    {
                        return mid;
                    }

                    if (nums[mid] < mid)
                    {
                        r = mid - 1;
                    }
                    else
                    {
                        l = mid + 1;
                    }
                }
                else
                {
                    if (nums[mid] < 0)
                    {
                        return FindMagicIndex(nums, mid + 1, r);
                    }

                    var t = FindMagicIndex(nums, l, mid);
                    if (t == -1)
                    {
                        t = FindMagicIndex(nums, mid + 1, r);
                    }

                    return t;
                }
            }

            return nums[l] == l ? l : -1;
        }

        public int FindMagicIndex(int[] nums)
        {
            return FindMagicIndex(nums, 0, nums.Length - 1);
        }

        #endregion

        #region 第一个错误的版本

        //https://leetcode-cn.com/problems/first-bad-version/
        bool IsBadVersion(int version)
        {
            return false;
        }

        public int FirstBadVersion(int n)
        {
            int l = 1, r = n;
            while (l < r)
            {
                var mid = l + (r - l) / 2;
                if (IsBadVersion(mid))
                {
                    r = mid;
                }
                else
                {
                    l = mid + 1;
                }
            }

            return IsBadVersion(l) ? l : l + 1;
        }

        #endregion

        #region 632. 最小区间

        //https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/
        public int[] SmallestRange(IList<IList<int>> nums)
        {
            var list = new List<int>();
            var numSet = new HashSet<int>();
            foreach (var arr in nums)
            {
                foreach (var num in arr)
                {
                    if (numSet.Add(num))
                    {
                        list.Add(num);
                    }
                }
            }

            list.Sort();
            numSet.Clear();
            int l = 0, r = 0, min = int.MaxValue;
            var res = new[] {list[0], list[0]};
            for (; r < list.Count; r++)
            {
                numSet.Add(list[r]);
                while (l <= r && nums.All(items => items.Any(n => numSet.Contains(n))))
                {
                    var diff = list[r] - list[l];
                    if (diff <= min)
                    {
                        if (diff < min || res[1] > list[r])
                        {
                            res[0] = list[l];
                            res[1] = list[r];
                        }

                        min = diff;
                    }

                    numSet.Remove(list[l++]);
                }
            }

            return res;
        }

        #endregion

        #region 336. 回文对

        //https://leetcode-cn.com/problems/palindrome-pairs/submissions/

        public IList<IList<int>> PalindromePairs(string[] words)
        {
            var wordsRev = new List<string>();
            var indices = new Dictionary<string, int>();

            int FindWord(string s, int left, int right)
            {
                var key = s.Substring(left, right - left + 1);
                if (indices.TryGetValue(key, out var index))
                {
                    return index;
                }

                return -1;
            }

            bool IsPalindrome(string s, int left, int right)
            {
                int len = right - left + 1;
                for (int i = 0; i < len / 2; i++)
                {
                    if (s[left + i] != s[right - i])
                    {
                        return false;
                    }
                }

                return true;
            }

            int n = words.Length;
            for (int i = 0; i < words.Length; i++)
            {
                var word = new string(words[i].Reverse().ToArray());
                wordsRev.Add(word);
                indices.Add(word, i);
            }

            IList<IList<int>> ret = new List<IList<int>>();
            for (int i = 0; i < n; i++)
            {
                var word = words[i];
                int m = word.Length;
                if (m == 0)
                {
                    continue;
                }

                for (int j = 0; j <= m; j++)
                {
                    if (IsPalindrome(word, j, m - 1))
                    {
                        int leftId = FindWord(word, 0, j - 1);
                        if (leftId != -1 && leftId != i)
                        {
                            ret.Add(new[] {i, leftId});
                        }
                    }

                    if (j != 0 && IsPalindrome(word, 0, j - 1))
                    {
                        int rightId = FindWord(word, j, m - 1);
                        if (rightId != -1 && rightId != i)
                        {
                            ret.Add(new[] {rightId, i});
                        }
                    }
                }
            }

            return ret;
        }

        #endregion

        #region 374. 猜数字大小

        //https://leetcode-cn.com/problems/guess-number-higher-or-lower/
        public int GuessNumber(int n)
        {
            int guess(int x)
            {
                return x;
            }

            int l = 1, r = n;
            while (l < r)
            {
                var t = l + (r - l) / 2;
                var cmp = guess(t);
                if (cmp == 0)
                {
                    return t;
                }

                if (cmp > 0)
                {
                    l = t + 1;
                }
                else
                {
                    r = t - 1;
                }
            }

            return l;
        }

        #endregion

        #region 99. 恢复二叉搜索树

        //https://leetcode-cn.com/problems/recover-binary-search-tree/
        public void RecoverTree(TreeNode root)
        {
            if (root == null)
            {
                return;
            }

            var stack = new Stack<TreeNode>();
            var vals = new List<int>();
            var head = root;
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                vals.Add(root.val);
                root = root.right;
            }

            vals.Sort();
            root = head;
            var i = 0;
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                root.val = vals[i++];
                root = root.right;
            }
        }

        //空间复杂度O(H)
        public void RecoverTreeByO(TreeNode root)
        {
            if (root == null)
            {
                return;
            }

            var stack = new Stack<TreeNode>();
            TreeNode x = null, y = null, prev = null;
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                if (prev != null && prev.val > root.val)
                {
                    y = root;
                    if (x == null)
                    {
                        x = prev;
                    }
                    else
                    {
                        break;
                    }
                }

                prev = root;
                root = root.right;
            }

            if (x == null)
            {
                return;
            }

            var tmp = x.val;
            x.val = y.val;
            y.val = tmp;
        }

        #endregion

        #region 696. 计数二进制子串

        //https://leetcode-cn.com/problems/count-binary-substrings/
        public int CountBinarySubstrings(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return 0;
            }

            int Count(int l, int r)
            {
                var count = 1;
                while (l > 0 && r < s.Length - 1 && s[l] == s[l - 1] && s[r] == s[r + 1])
                {
                    count++;
                    l--;
                    r++;
                }

                return count;
            }

            var res = 0;
            for (var i = 0; i < s.Length - 1; i++)
            {
                if (s[i] != s[i + 1])
                {
                    var len = Count(i, i + 1);
                    res += len;
                    i += len - 1;
                }
            }

            return res;
        }

        public int CountBinarySubstringsByGroup(string s)
        {
            int res = 0, i = 0, count = 0;
            while (i < s.Length)
            {
                var ch = s[i++];
                var len = 1;
                while (i < s.Length && ch == s[i])
                {
                    i++;
                    len++;
                }

                res += Math.Min(count, len);
                count = len;
            }

            return res;
        }

        #endregion

        #region 33. 搜索旋转排序数组

        //https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
        public int SearchRotate(int[] nums, int target)
        {
            if (nums.Length <= 0)
            {
                return -1;
            }

            int left = 0, right = nums.Length - 1;
            while (left < right)
            {
                if (target == nums[left])
                {
                    return left;
                }

                if (target == nums[right])
                {
                    return right;
                }

                var mid = (left + right) / 2;
                if (nums[mid] == target)
                {
                    return mid;
                }

                if (nums[mid] > nums[left]) //左边数组有序 在左边 
                {
                    if (target < nums[left] || target > nums[mid])
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }
                else if (nums[mid] < nums[right]) //右边数组有序 在右边
                {
                    if (target >= nums[right] || target < nums[mid])
                    {
                        right = mid - 1;
                    }
                    else
                    {
                        left = mid + 1;
                    }
                }
                else if (nums[mid] < nums[left]) //左边数组非有序，旋转数在左边
                {
                    if (target >= nums[left] || target < nums[mid]) //m<left,left<t => m <t     // && nums[mid] < target
                    {
                        right = mid - 1;
                    }
                    else //m<left,t<left
                    {
                        left = mid + 1;
                    }
                }
                else if (nums[mid] > nums[right]) //右边数组非有序，旋转数在右边
                {
                    if (target <= nums[right] || target > nums[mid]
                    ) //m<left,left<t => m <t     // && nums[mid] < target
                    {
                        left = mid + 1;
                    }
                    else //m<left,t<left
                    {
                        right = mid - 1;
                    }
                }
                else if (nums[mid] == nums[right] && nums[mid] == nums[left])
                {
                    right--;
                }
                else if (nums[mid] == nums[left])
                {
                    left++;
                }
                else if (nums[mid] == nums[right])
                {
                    right--;
                }
            }

            return nums[left] == target ? left : -1;
        }

        //代码优化
        public int SearchRotateCodeClean(int[] nums, int target)
        {
            if (nums.Length <= 0)
            {
                return -1;
            }

            int l = 0, r = nums.Length - 1;
            while (l < r)
            {
                if (nums[l] == target)
                {
                    return l;
                }

                if (nums[r] == target)
                {
                    return r;
                }

                var mid = l + (r - l) / 2;
                if (nums[mid] == target)
                {
                    return mid;
                }

                //左半边有序
                if (nums[mid] > nums[l])
                {
                    //存在左半边
                    //2 3
                    if (target > nums[l] && target < nums[mid])
                    {
                        r = mid - 1;
                    }
                    else
                    {
                        l = mid + 1;
                    }
                }
                else //右半边可能有序 (m,r)有序
                {
                    if (target > nums[mid] && target < nums[r])
                    {
                        l = mid + 1;
                    }
                    else
                    {
                        r = mid - 1;
                    }
                }
            }

            return nums[l] == target ? l : -1;
        }

        #endregion

        #region 153. 寻找旋转排序数组中的最小值

        //https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/
        public int FindMin(int[] nums)
        {
            if (nums == null || nums.Length <= 0)
            {
                return -1;
            }

            int l = 0, r = nums.Length - 1;
            while (l < r)
            {
                if (nums[l] < nums[r])
                {
                    break;
                }

                var m = l + (r - l) / 2;
                if (nums[m] >= nums[l])
                {
                    //左边有序
                    l = m + 1;
                }
                else
                {
                    r = m;
                }
            }

            return nums[l];
        }

        #endregion

        #region 658. 找到 K 个最接近的元素

        //https://leetcode-cn.com/problems/find-k-closest-elements/
        public IList<int> FindClosestElements(int[] arr, int k, int x)
        {
            int l = 0, r = arr.Length - 1;
            while (l < r)
            {
                var m = (l + r) / 2;
                if (arr[m] == x)
                {
                    l = m;
                    break;
                }

                if (arr[m] < x)
                {
                    l = m + 1;
                }
                else
                {
                    r = m - 1;
                }
            }

            if (arr[l] != x)
            {
                if (l > 0)
                {
                    int ln = x - arr[l - 1], rn = arr[l] - x;
                    if (ln > rn)
                    {
                        l = r;
                    }
                }
            }

            int i = l, j = l;
            var size = 1;
            var res = new int[k];
            while (size < k)
            {
                //[1,3] 1 2
                //0 1 m=0 arr[m]=1 l=1
                if (i > 0 && j < arr.Length - 1)
                {
                    int ln = x - arr[i - 1], rn = arr[j + 1] - x;
                    if (ln <= rn)
                    {
                        i--;
                    }
                    else
                    {
                        j++;
                    }
                }
                else if (i > 0)
                {
                    i--;
                }
                else
                {
                    j++;
                }

                size++;
            }

            Array.Copy(arr, i, res, 0, res.Length);
            return res;
        }

        #endregion
    }
}