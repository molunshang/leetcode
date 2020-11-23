using System;
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
                else if (nums[mid] < nums[l]) //右半边可能有序 (m,r)有序
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
                else
                {
                    l++;
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

        #region 133. 克隆图

        //https://leetcode-cn.com/problems/clone-graph/
        Node CloneGraph(Node node, Dictionary<Node, Node> dict)
        {
            if (node == null)
            {
                return null;
            }

            if (dict.TryGetValue(node, out var newNode))
            {
                return newNode;
            }

            newNode = new Node(node.val);
            dict.Add(node, newNode);
            if (node.neighbors != null)
            {
                newNode.neighbors = new List<Node>();
                foreach (var neighbor in node.neighbors)
                {
                    newNode.neighbors.Add(CloneGraph(neighbor, dict));
                }
            }

            return newNode;
        }

        public Node CloneGraph(Node node)
        {
            return CloneGraph(node, new Dictionary<Node, Node>());
        }

        #endregion

        #region 367. 有效的完全平方数

        //https://leetcode-cn.com/problems/valid-perfect-square/
        public bool IsPerfectSquare(int num)
        {
            if (num <= 0)
            {
                return num == 0;
            }

            long l = 1, r = num / 2;
            while (l < r)
            {
                var t = l + (r - l) / 2;
                var pow = t * t;
                if (pow == num)
                {
                    return true;
                }

                if (pow > num || pow < 0)
                {
                    r = t - 1;
                }
                else
                {
                    l = t + 1;
                }
            }

            return l * l == num;
        }

        #endregion

        #region 20. 有效的括号

        //https://leetcode-cn.com/problems/valid-parentheses/
        public bool IsValid(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return true;
            }

            if ((s.Length & 1) == 1)
            {
                return false;
            }

            var dict = new Dictionary<char, char> {{'(', ')'}, {'[', ']'}, {'{', '}'}};
            var stack = new Stack<char>();
            foreach (var ch in s)
            {
                if (dict.ContainsKey(ch))
                {
                    stack.Push(ch);
                }
                else if (stack.Count <= 0 || ch != dict[stack.Pop()])
                {
                    return false;
                }
            }

            return stack.Count <= 0;
        }

        #endregion

        #region 744. 寻找比目标字母大的最小字母

        //https://leetcode-cn.com/problems/find-smallest-letter-greater-than-target/
        public char NextGreatestLetter(char[] letters, char target)
        {
            int l = 0, r = letters.Length - 1;
            while (l <= r)
            {
                var m = (l + r) / 2;
                if (letters[m] <= target)
                {
                    l = m + 1;
                }
                else
                {
                    r = m - 1;
                }
            }

            return letters[l % letters.Length];
        }

        #endregion

        #region 546. 移除盒子

        //https://leetcode-cn.com/problems/remove-boxes/

        //暴力解
        public int RemoveBoxes(List<int> boxes)
        {
            if (boxes.Count <= 1)
            {
                return boxes.Count;
            }

            var max = 0;
            int slow = 0, fast = 0;
            while (slow < boxes.Count && fast <= boxes.Count)
            {
                if (fast != boxes.Count && boxes[slow] == boxes[fast])
                {
                    fast++;
                }
                else
                {
                    var count = fast - slow;
                    var rm = boxes[slow];
                    boxes.RemoveRange(slow, count);
                    max = Math.Max(max, count * count + RemoveBoxes(boxes));
                    boxes.InsertRange(slow, Enumerable.Repeat(rm, count));
                    slow = fast;
                }
            }

            return max;
        }

        public int RemoveBoxes(int[] boxes)
        {
            return RemoveBoxes(new List<int>(boxes));
        }

        //动态规划（记忆化）
        public int RemoveBoxesByDp(int[] boxes)
        {
            var cache = new int[boxes.Length, boxes.Length, 100];

            int Dfs(int l, int r, int k)
            {
                if (l > r)
                {
                    return 0;
                }

                if (cache[l, r, k] != 0)
                {
                    return cache[l, r, k];
                }

                while (l < r && boxes[r] == boxes[r - 1])
                {
                    k++;
                    r--;
                }

                cache[l, r, k] = Dfs(l, r - 1, 0) + (k + 1) * (k + 1);
                for (int i = l; i < r; i++)
                {
                    if (boxes[i] == boxes[r])
                    {
                        cache[l, r, k] = Math.Max(cache[l, r, k], Dfs(l, i, k + 1) + Dfs(i + 1, r - 1, 0));
                    }
                }

                return cache[l, r, k];
            }

            ;

            return Dfs(0, boxes.Length - 1, 0);
        }

        #endregion

        #region 733. 图像渲染/面试题 08.10. 颜色填充

        //https://leetcode-cn.com/problems/flood-fill/
        //https://leetcode-cn.com/problems/color-fill-lcci/
        public int[][] FloodFill(int[][] image, int sr, int sc, int newColor)
        {
            var color = image[sr][sc];
            if (color == newColor)
            {
                return image;
            }

            var queue = new Queue<int[]>();
            queue.Enqueue(new[] {sr, sc});
            while (queue.TryDequeue(out var point))
            {
                int x = point[0], y = point[1];
                if (x < 0 || x >= image.Length || y < 0 || y >= image[x].Length || image[x][y] != color)
                {
                    continue;
                }

                image[x][y] = newColor;
                queue.Enqueue(new[] {x + 1, y});
                queue.Enqueue(new[] {x - 1, y});
                queue.Enqueue(new[] {x, y + 1});
                queue.Enqueue(new[] {x, y - 1});
            }

            return image;
        }

        #endregion

        #region 寻找旋转排序数组中的最小值 II

        //https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/
        public int FindMinII(int[] nums)
        {
            int l = 0, r = nums.Length - 1;
            while (l < r)
            {
                if (nums[l] < nums[r])
                {
                    return nums[l];
                }

                var m = (l + r) / 2;
                if (nums[m] > nums[l])
                {
                    l = m + 1;
                }
                else if (nums[m] < nums[l])
                {
                    r = m;
                }
                else
                {
                    l++;
                }
            }

            return nums[l];
        }

        #endregion

        #region 109. 有序链表转换二叉搜索树

        //https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/

        ListNode SplitHalf(ListNode head)
        {
            ListNode slow = head, fast = head, prev = null;
            while (fast != null && fast.next != null)
            {
                prev = slow;
                slow = slow.next;
                fast = fast.next.next;
            }

            if (prev != null)
            {
                prev.next = null;
            }

            return slow;
        }

        public TreeNode SortedListToBST(ListNode head)
        {
            if (head == null)
            {
                return null;
            }

            if (head.next == null)
            {
                return new TreeNode(head.val);
            }

            var half = SplitHalf(head);
            var root = new TreeNode(half.val) {left = SortedListToBST(head), right = SortedListToBST(half.next)};
            return root;
        }

        #endregion

        #region 719. 找出第 k 小的距离对

        //https://leetcode-cn.com/problems/find-k-th-smallest-pair-distance/

        public int SmallestDistancePair(int[] nums, int k)
        {
            Array.Sort(nums);
            int l = 0, h = nums[nums.Length - 1] - nums[0];
            while (l < h)
            {
                var m = (l + h) / 2;
                int count = 0, left = 0;
                for (int i = 0; i < nums.Length; i++)
                {
                    while (left < i && nums[i] - nums[left] > m)
                    {
                        left++;
                    }

                    count += i - left;
                }

                if (count >= k)
                {
                    h = m;
                }
                else
                {
                    l = m + 1;
                }
            }

            return l;
        }

        #endregion

        #region 81. 搜索旋转排序数组 II

        //https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/submissions/
        public bool SearchII(int[] nums, int target)
        {
            if (nums.Length <= 0)
            {
                return false;
            }

            int l = 0, r = nums.Length - 1;
            while (l < r)
            {
                if (nums[l] == target || nums[r] == target)
                {
                    return true;
                }

                var mid = l + (r - l) / 2;
                if (nums[mid] == target)
                {
                    return true;
                }

                if (nums[mid] > nums[l])
                {
                    if (target > nums[l] && target < nums[mid])
                    {
                        r = mid - 1;
                    }
                    else
                    {
                        l = mid + 1;
                    }
                }
                else if (nums[mid] < nums[l])
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
                else
                {
                    l++;
                }
            }

            return nums[l] == target;
        }

        #endregion

        #region 647. 回文子串

        //https://leetcode-cn.com/problems/palindromic-substrings/
        public int CountSubstrings(string s)
        {
            int CenterCount(string str, int l, int r)
            {
                var count = 0;
                while (l >= 0 && r < str.Length && str[l] == str[r])
                {
                    l--;
                    r++;
                    count++;
                }

                return count;
            }

            var res = 0;
            for (int i = 0; i < s.Length; i++)
            {
                res += CenterCount(s, i, i);
                res += CenterCount(s, i, i + 1);
            }

            return res;
        }

        public int CountSubstringsByDp(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return 0;
            }

            var res = 0;
            var dp = new bool[s.Length, s.Length];
            for (int len = 1; len <= s.Length; len++)
            {
                for (int i = 0, j = i + len - 1; j < s.Length; i++, j++)
                {
                    if (len == 1)
                    {
                        dp[i, j] = true;
                    }
                    else if (len == 2)
                    {
                        dp[i, j] = s[i] == s[j];
                    }
                    else
                    {
                        dp[i, j] = s[i] == s[j] && dp[i + 1, j - 1];
                    }

                    if (dp[i, j])
                    {
                        res++;
                    }
                }
            }

            return res;
        }

        #endregion

        #region 1510. 石子游戏 IV

        //https://leetcode-cn.com/problems/stone-game-iv/

        public bool WinnerSquareGame(int n)
        {
            var cahce = new Dictionary<int, bool>();

            bool Dfs(int num)
            {
                if (num <= 1)
                {
                    return num == 1;
                }

                if (cahce.TryGetValue(num, out var flag))
                {
                    return flag;
                }

                for (int i = 1; i * i <= num; i++)
                {
                    if (!Dfs(num - i * i))
                    {
                        flag = true;
                        break;
                    }
                }

                cahce[num] = flag;
                return flag;
            }

            return Dfs(n);
        }

        #endregion

        #region 面试题 08.14. 布尔运算

        //https://leetcode-cn.com/problems/boolean-evaluation-lcci/
        int[] CountEval(string s, int l, int r, int[,][] cache)
        {
            int[] res;
            if (l >= r)
            {
                res = new int[2];
                res[s[l] - '0'] = 1;
                return res;
            }

            if (cache[l, r] != null)
            {
                return cache[l, r];
            }

            res = new int[2];
            for (int i = l + 1; i < r; i += 2)
            {
                int[] left = CountEval(s, l, i - 1, cache), right = CountEval(s, i + 1, r, cache);
                switch (s[i])
                {
                    case '|':
                        //0 0|0
                        //1 1|1,1|0,0|1
                        res[0] += left[0] * right[0];
                        res[1] += left[0] * right[1] + left[1] * right[0] + left[1] * right[1];
                        break;
                    case '^':
                        //0 0^0,1^1
                        //1 0^1,1^0
                        res[0] += left[0] * right[0] + left[1] * right[1];
                        res[1] += left[0] * right[1] + left[1] * right[0];
                        break;
                    case '&':
                        //0 0&0,1&0,0&1
                        //1 1&1
                        res[0] += left[0] * right[0] + left[1] * right[0] + left[0] * right[1];
                        res[1] += left[1] * right[1];
                        break;
                }
            }

            cache[l, r] = res;
            return res;
        }

        public int CountEval(string s, int result)
        {
            var cache = new int[s.Length, s.Length][];
            var res = CountEval(s, 0, s.Length - 1, cache);
            return res[result];
        }

        #endregion

        #region 529. 扫雷游戏

        //https://leetcode-cn.com/problems/minesweeper/
        public char[][] UpdateBoard(char[][] board, int[] click)
        {
            var queue = new Queue<int[]>();
            queue.Enqueue(click);
            var steps = new[] {-1, 0, 1};
            while (queue.TryDequeue(out click))
            {
                int x = click[0], y = click[1];
                if (board[x][y] == 'M')
                {
                    board[x][y] = 'X';
                    break;
                }

                if (board[x][y] != 'E')
                {
                    continue;
                }

                var m = 0;
                foreach (var xstep in steps)
                {
                    foreach (var ystep in steps)
                    {
                        int nx = x + xstep, ny = y + ystep;
                        if (nx < 0 || nx >= board.Length || ny < 0 || ny >= board[x].Length || nx == x && ny == y)
                        {
                            continue;
                        }

                        if (board[nx][ny] == 'M')
                        {
                            m++;
                        }
                    }
                }

                if (m == 0)
                {
                    board[x][y] = 'B';
                    foreach (var xstep in steps)
                    {
                        foreach (var ystep in steps)
                        {
                            int nx = x + xstep, ny = y + ystep;
                            if (nx < 0 || nx >= board.Length || ny < 0 || ny >= board[x].Length || nx == x && ny == y)
                            {
                                continue;
                            }

                            queue.Enqueue(new[] {nx, ny});
                        }
                    }
                }
                else
                {
                    board[x][y] = (char) ('0' + m);
                }
            }

            return board;
        }

        #endregion

        #region 921. 使括号有效的最少添加

        //https://leetcode-cn.com/problems/minimum-add-to-make-parentheses-valid/
        public int MinAddToMakeValid(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return 0;
            }

            int res = 0, l = 0;
            foreach (var ch in s)
            {
                l += ch == '(' ? 1 : -1;
                if (l < 0)
                {
                    l = 0;
                    res++;
                }
            }

            return res + l;
        }

        #endregion

        #region 1129. 颜色交替的最短路径

        //https://leetcode-cn.com/problems/shortest-path-with-alternating-colors/
        public int[] ShortestAlternatingPaths(int n, int[][] red_edges, int[][] blue_edges)
        {
            IList<int>[] redDict = new IList<int>[n], blueDict = new IList<int>[n];
            foreach (var edge in red_edges)
            {
                if (redDict[edge[0]] == null)
                {
                    redDict[edge[0]] = new List<int>();
                }

                redDict[edge[0]].Add(edge[1]);
            }

            foreach (var edge in blue_edges)
            {
                if (blueDict[edge[0]] == null)
                {
                    blueDict[edge[0]] = new List<int>();
                }

                blueDict[edge[0]].Add(edge[1]);
            }

            int Bfs(bool isRed, int target)
            {
                var queue = new Queue<int>();
                HashSet<int> red = new HashSet<int>(), blue = new HashSet<int>();
                var path = 0;
                queue.Enqueue(0);
                while (queue.Count > 0)
                {
                    var size = queue.Count;
                    var set = isRed ? red : blue;
                    var dict = isRed ? redDict : blueDict;
                    while (size > 0)
                    {
                        size--;
                        var point = queue.Dequeue();
                        if (target == point)
                        {
                            return path;
                        }

                        if (!set.Add(point))
                        {
                            continue;
                        }

                        var next = dict[point];
                        if (next == null)
                        {
                            continue;
                        }

                        foreach (var p in next)
                        {
                            queue.Enqueue(p);
                        }
                    }

                    isRed = !isRed;
                    path++;
                }

                return -1;
            }


            var res = new int[n];
            for (int i = 1; i < n; i++)
            {
                int r = Bfs(true, i), b = Bfs(false, i);
                if (r != -1 && b != -1)
                {
                    res[i] = Math.Min(r, b);
                }
                else if (r == -1)
                {
                    res[i] = b;
                }
                else if (b == -1)
                {
                    res[i] = r;
                }
                else
                {
                    res[i] = -1;
                }
            }

            return res;
        }

        #endregion

        #region 679. 24 点游戏

        //https://leetcode-cn.com/problems/24-game/
        public bool JudgePoint24(int[] nums)
        {
            var operatorDict = new Func<double, double, double>[4];
            operatorDict[0] = (a, b) => a + b;
            operatorDict[1] = (a, b) => a - b;
            operatorDict[2] = (a, b) => a * b;
            operatorDict[3] = (a, b) => a / b;

            bool Dfs(IList<double> list)
            {
                if (list.Count <= 0)
                {
                    return false;
                }

                if (list.Count == 1)
                {
                    return Math.Abs(list[0] - 24) <= 0.000001;
                }

                for (int i = 0; i < list.Count; i++)
                {
                    var one = list[i];
                    list.RemoveAt(i);
                    for (int j = 0; j < list.Count; j++)
                    {
                        if (list[j] < 0.000001)
                        {
                            continue;
                        }

                        var two = list[j];
                        list.RemoveAt(j);
                        for (int k = 0; k < 4; k++)
                        {
                            var num = operatorDict[k](one, two);
                            list.Add(num);
                            if (Dfs(list))
                            {
                                return true;
                            }

                            list.RemoveAt(list.Count - 1);
                        }

                        list.Insert(j, two);
                    }

                    list.Insert(i, one);
                }

                return false;
            }

            var numbers = new List<double>(nums.Select(n => (double) n));
            return Dfs(numbers);
        }

        //官解
        public bool JudgePoint24Answer(int[] nums)
        {
            bool Dfs(IList<double> list)
            {
                if (list.Count <= 0)
                {
                    return false;
                }

                if (list.Count == 1)
                {
                    return Math.Abs(list[0] - 24) <= 0.000001;
                }

                var next = new List<double>();
                for (int i = 0; i < list.Count; i++)
                {
                    var one = list[i];
                    for (int j = 0; j < list.Count; j++)
                    {
                        if (i == j)
                        {
                            continue;
                        }

                        var two = list[j];
                        next.AddRange(list.Where((n, m) => m != i && m != j));
                        for (int k = 0; k < 4; k++)
                        {
                            if (k < 2 && i > j)
                            {
                                continue;
                            }

                            if (k == 0)
                            {
                                next.Add(one + two);
                            }
                            else if (k == 1)
                            {
                                next.Add(one * two);
                            }
                            else if (k == 2)
                            {
                                next.Add(one - two);
                            }
                            else if (k == 3)
                            {
                                if (Math.Abs(two) < 0.000001)
                                {
                                    continue;
                                }

                                next.Add(one / two);
                            }

                            if (Dfs(next))
                            {
                                return true;
                            }

                            next.RemoveAt(next.Count - 1);
                        }

                        next.Clear();
                    }
                }

                return false;
            }

            var numbers = new List<double>(nums.Select(n => (double) n));
            return Dfs(numbers);
        }

        #endregion

        #region 201. 数字范围按位与

        //https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/
        public int RangeBitwiseAnd(int m, int n)
        {
            if (m == 0)
            {
                return 0;
            }

            int BitCount(int num)
            {
                var count = 0;
                while (num != 0)
                {
                    count++;
                    num >>= 1;
                }

                return count;
            }

            var most = BitCount(m);
            var res = 0;
            for (int i = most - 1; i >= 0; i--)
            {
                var mask = int.MaxValue ^ (1 << i);
                int mnum = n & mask, nnum = m & mask;
                if (mnum >= m && mnum <= n || (nnum >= m && nnum <= n))
                {
                    res = (res << 1) | 0;
                }
                else
                {
                    res = (res << 1) | 1;
                }
            }

            return res;
        }

        //官方解答
        public int RangeBitwiseAndAnswer(int m, int n)
        {
            if (m == 0)
            {
                return 0;
            }

            int BitMove(int i, int j)
            {
                var move = 0;
                while (i < j)
                {
                    i >>= 1;
                    j >>= 1;
                    move++;
                }

                return i << move;
            }

            //Brian Kernighan 算法
            int BrianKernighan(int i, int j)
            {
                while (i < j)
                {
                    j = j & (j - 1);
                }

                return j;
            }

            return BitMove(m, n);
        }

        #endregion

        #region 459. 重复的子字符串

        //https://leetcode-cn.com/problems/repeated-substring-pattern/
        public bool RepeatedSubstringPattern(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return false;
            }

            for (int len = 1; len <= s.Length / 2; len++)
            {
                if (s.Length % len != 0)
                {
                    continue;
                }

                var flag = true;
                var subStr = s.Substring(0, len);
                for (int i = len; i < s.Length; i += len)
                {
                    if (subStr.Where((t, j) => subStr[j] != s[i + j]).Any())
                    {
                        flag = false;
                        break;
                    }
                }

                if (flag)
                {
                    return true;
                }
            }

            return false;
        }

        #endregion

        #region 1391. 检查网格中是否存在有效路径

        //https://leetcode-cn.com/problems/check-if-there-is-a-valid-path-in-a-grid/
        public bool HasValidPath(int[][] grid)
        {
            var pathDict = new IList<int>[13];
            pathDict[0] = new[] {1, 2, 3, 4, 5, 6};
            pathDict[1] = new[] {1, 3, 5};
            pathDict[2] = new[] {1, 4, 6};
            pathDict[3] = new[] {2, 5, 6};
            pathDict[4] = new[] {2, 3, 4};
            pathDict[5] = new[] {1, 4, 6};
            pathDict[6] = new[] {2, 5, 6};
            pathDict[7] = new[] {1, 3, 5};
            pathDict[8] = new[] {2, 5, 6};
            pathDict[9] = new[] {1, 3, 6};
            pathDict[10] = new[] {2, 3, 4};
            pathDict[11] = new[] {1, 3, 5};
            pathDict[12] = new[] {2, 3, 4};
            int targetX = grid.Length - 1, targetY = grid[0].Length - 1;
            var visited = new bool[grid.Length, grid[0].Length];

            bool Bfs()
            {
                var queue = new Queue<int[]>();
                queue.Enqueue(new[] {0, 0, 0});
                while (queue.TryDequeue(out var point))
                {
                    int x = point[0], y = point[1], step = point[2];
                    if (x < 0 || x > targetX || y < 0 || y > targetY || visited[x, y])
                    {
                        continue;
                    }

                    var type = grid[x][y];
                    if (!pathDict[step].Contains(type))
                    {
                        continue;
                    }

                    if (x == targetX && y == targetY)
                    {
                        return true;
                    }

                    visited[x, y] = true;
                    switch (type)
                    {
                        case 1: //左右
                            queue.Enqueue(new[] {x, y + 1, 1});
                            queue.Enqueue(new[] {x, y - 1, 2});
                            break;
                        case 2: //上下
                            queue.Enqueue(new[] {x + 1, y, 3});
                            queue.Enqueue(new[] {x - 1, y, 4});
                            break;
                        case 3: //左下
                            queue.Enqueue(new[] {x, y - 1, 5});
                            queue.Enqueue(new[] {x + 1, y, 6});
                            break;
                        case 4: //右下
                            queue.Enqueue(new[] {x, y + 1, 7});
                            queue.Enqueue(new[] {x + 1, y, 8});
                            break;
                        case 5: //左上
                            queue.Enqueue(new[] {x, y - 1, 9});
                            queue.Enqueue(new[] {x - 1, y, 10});
                            break;
                        case 6: //右上
                            queue.Enqueue(new[] {x, y + 1, 11});
                            queue.Enqueue(new[] {x - 1, y, 12});
                            break;
                    }
                }

                return false;
            }

            bool Dfs(int x, int y, int direct)
            {
                if (x < 0 || x > targetX || y < 0 || y > targetY || visited[x, y])
                {
                    return false;
                }

                var type = grid[x][y];
                if (!pathDict[direct].Contains(type))
                {
                    return false;
                }

                if (x == targetX && y == targetY)
                {
                    return true;
                }

                var flag = false;
                visited[x, y] = true;
                switch (type)
                {
                    case 1: //左右
                        flag = Dfs(x, y + 1, 1) || Dfs(x, y - 1, 2);
                        break;
                    case 2: //上下
                        flag = Dfs(x + 1, y, 3) || Dfs(x - 1, y, 4);
                        break;
                    case 3: //左下
                        flag = Dfs(x, y - 1, 5) || Dfs(x + 1, y, 6);
                        break;
                    case 4: //右下
                        flag = Dfs(x, y + 1, 7) || Dfs(x + 1, y, 8);
                        break;
                    case 5: //左上
                        flag = Dfs(x, y - 1, 9) || Dfs(x - 1, y, 10);
                        break;
                    case 6: //右上
                        flag = Dfs(x, y + 1, 11) || Dfs(x - 1, y, 12);
                        break;
                }

                return flag;
            }

            return Bfs();
        }

        #endregion

        #region 491. 递增子序列

        //https://leetcode-cn.com/problems/increasing-subsequences/
        public IList<IList<int>> FindSubsequences(int[] nums)
        {
            var result = new List<IList<int>>();
            var path = new List<int>();

            void Dfs(int index)
            {
                if (index >= nums.Length)
                {
                    return;
                }

                var visited = new HashSet<int>();
                for (int i = index; i < nums.Length; i++)
                {
                    if (!visited.Add(nums[i]))
                    {
                        continue;
                    }

                    var flag = path.Count <= 0 || path[path.Count - 1] <= nums[i];
                    if (!flag)
                    {
                        continue;
                    }

                    path.Add(nums[i]);
                    if (path.Count > 1)
                    {
                        result.Add(path.ToArray());
                    }

                    Dfs(i + 1);
                    path.RemoveAt(path.Count - 1);
                }
            }

            void DFS(int index, int last)
            {
                while (true)
                {
                    if (index >= nums.Length)
                    {
                        if (path.Count > 1)
                        {
                            result.Add(path.ToArray());
                        }

                        return;
                    }

                    if (nums[index] >= last)
                    {
                        path.Add(nums[index]);
                        DFS(index + 1, nums[index]);
                        path.RemoveAt(path.Count - 1);
                    }

                    if (nums[index] != last)
                    {
                        index = index + 1;
                        continue;
                    }

                    break;
                }
            }

            Dfs(0);
            return result;
        }

        #endregion

        #region 1486. 数组异或操作

        //https://leetcode-cn.com/problems/xor-operation-in-an-array/
        public int XorOperation(int n, int start)
        {
            var res = start;
            for (int i = 1; i < n; i++)
            {
                res = res ^ (start + i * 2);
            }

            return res;
        }

        #endregion

        #region 1309. 解码字母到整数映射

        //https://leetcode-cn.com/problems/decrypt-string-from-alphabet-to-integer-mapping/
        public string FreqAlphabetsByDfs(string s)
        {
            var dict = new Dictionary<string, char>();
            for (int i = 1; i < 10; i++)
            {
                dict[i.ToString()] = (char) ('a' + (i - 1));
            }

            for (int i = 10; i < 27; i++)
            {
                dict[i.ToString() + "#"] = (char) ('a' + (i - 1));
            }

            var res = new StringBuilder();

            bool Dfs(int index)
            {
                if (index >= s.Length)
                {
                    return true;
                }

                if (s[index] == '#')
                {
                    return false;
                }

                for (int i = 1, j = Math.Min(s.Length - index, 3); i <= j; i++)
                {
                    var k = s.Substring(index, i);
                    if (dict.TryGetValue(k, out var sub))
                    {
                        res.Append(sub);
                        if (Dfs(index + i))
                        {
                            return true;
                        }

                        res.Remove(res.Length - i, i);
                    }
                }

                return false;
            }

            Dfs(0);
            return res.ToString();
        }

        public string FreqAlphabets(string s)
        {
            var res = string.Empty;
            for (int i = s.Length - 1; i >= 0; i--)
            {
                var cur = 0;
                if (s[i] == '#')
                {
                    cur = (s[i - 2] - '0') * 10 + s[i - 1] - '0';
                    i -= 2;
                }
                else
                {
                    cur = s[i] - '0';
                }

                res = (char) (cur - 1 + 'a') + res;
            }

            return res;
        }

        #endregion
        
        
    }
}