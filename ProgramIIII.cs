using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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

        #region 17. 电话号码的字母组合

        //https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/
        public IList<string> LetterCombinations(string digits)
        {
            if (string.IsNullOrEmpty(digits))
            {
                return new string[0];
            }

            var result = new List<string>();
            var dict = new IList<char>[8];
            var chars = new List<char>();
            for (int i = 0, j = 0; i < dict.Length; i++)
            {
                var step = i == 5 || i == 7 ? 4 : 3;
                while (step != 0)
                {
                    chars.Add((char)('a' + j));
                    j++;
                    step--;
                }

                dict[i] = chars.ToArray();
                chars.Clear();
            }

            void Dfs(int index, List<char> sub)
            {
                if (index >= digits.Length)
                {
                    result.Add(new string(sub.ToArray()));
                    return;
                }

                var ch = digits[index];
                var next = dict[ch - 2];
                foreach (var c in next)
                {
                    sub.Add(c);
                    Dfs(index + 1, sub);
                    sub.RemoveAt(sub.Count - 1);
                }
            }

            Dfs(0, new List<char>());
            return result;
        }

        #endregion

        #region 947. 移除最多的同行或同列石头

        //https://leetcode-cn.com/problems/most-stones-removed-with-same-row-or-column/
        public int RemoveStones(int[][] stones)
        {
            Dictionary<int, int> rows = new Dictionary<int, int>(), cols = new Dictionary<int, int>();
            foreach (var stone in stones)
            {
                int r = stone[0], c = stone[1];
                int size;
                if (rows.TryGetValue(r, out size))
                {
                    rows[r] = size + 1;
                }
                else
                {
                    rows[r] = 1;
                }

                if (cols.TryGetValue(c, out size))
                {
                    cols[c] = size + 1;
                }
                else
                {
                    cols[c] = 1;
                }
            }

            var rm = new bool[stones.Length];

            int Dfs()
            {
                var res = 0;
                for (var i = 0; i < stones.Length; i++)
                {
                    if (rm[i])
                    {
                        continue;
                    }

                    int x = stones[i][0], y = stones[i][1];
                    if (rows[x] <= 1 && cols[y] <= 1)
                    {
                        continue;
                    }

                    rm[i] = true;
                    rows[x]--;
                    cols[y]--;
                    res = Math.Max(res, Dfs() + 1);
                    rows[x]++;
                    cols[y]++;
                    rm[i] = false;
                }

                return res;
            }

            return Dfs();
        }

        //将点连接为图进行DFS
        public int RemoveStonesDfsGraph(int[][] stones)
        {
            var graph = new Dictionary<int, IList<int>>();
            for (var i = 0; i < stones.Length; i++)
            {
                for (int j = i + 1; j < stones.Length; j++)
                {
                    if (stones[i][0] != stones[j][0] && stones[i][1] != stones[j][1])
                    {
                        continue;
                    }

                    if (!graph.TryGetValue(i, out var points))
                    {
                        points = new List<int>();
                        graph[i] = points;
                    }

                    points.Add(j);
                    if (!graph.TryGetValue(j, out points))
                    {
                        points = new List<int>();
                        graph[j] = points;
                    }

                    points.Add(i);
                }
            }

            var stack = new Stack<int>();
            var rm = new bool[stones.Length];
            var res = 0;
            for (int i = 0; i < stones.Length; i++)
            {
                if (rm[i])
                {
                    continue;
                }

                res--;
                stack.Push(i);
                while (stack.TryPop(out var point))
                {
                    if (rm[point])
                    {
                        continue;
                    }

                    rm[point] = true;
                    res++;
                    if (graph.TryGetValue(point, out var next))
                    {
                        foreach (var n in next)
                        {
                            stack.Push(n);
                        }
                    }
                }
            }

            return res;
        }

        #endregion

        #region 822. 翻转卡片游戏

        //https://leetcode-cn.com/problems/card-flipping-game/
        public int Flipgame(int[] fronts, int[] backs)
        {
            var set = new HashSet<int>();
            for (int i = 0; i < fronts.Length; i++)
            {
                if (fronts[i] == backs[i])
                {
                    set.Add(fronts[i]);
                }
            }

            var res = fronts.Concat(backs).Where(n => !set.Contains(n));
            return res.Any() ? res.Min() : 0;
        }

        #endregion

        #region 691. 贴纸拼词

        //https://leetcode-cn.com/problems/stickers-to-spell-word/
        public int MinStickers(string[] stickers, string target)
        {
            var targetDict = target.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
            var stickerDict = stickers
                .Select(sticker => sticker.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count()))
                .Where(dict => targetDict.Any(kv => dict.ContainsKey(kv.Key))).ToList();

            var cache = new Dictionary<string, int>();

            int Dfs(Dictionary<char, int> dict, string key)
            {
                if (dict.Count <= 0)
                {
                    return 0;
                }

                if (cache.TryGetValue(key, out var res))
                {
                    return res;
                }

                res = int.MaxValue;
                var nextDict = new Dictionary<char, int>();
                var keyBuiler = new StringBuilder();
                for (var i = 0; i < stickerDict.Count; i++)
                {
                    nextDict.Clear();
                    keyBuiler.Clear();
                    var sticker = stickerDict[i];
                    var skip = true;
                    foreach (var kv in dict)
                    {
                        if (sticker.TryGetValue(kv.Key, out var size))
                        {
                            if (size < kv.Value)
                            {
                                nextDict.Add(kv.Key, kv.Value - size);
                                keyBuiler.Append(kv.Key).Append(',').Append(kv.Value - size);
                            }

                            skip = false;
                        }
                        else
                        {
                            nextDict.Add(kv.Key, kv.Value);
                            keyBuiler.Append(kv.Key).Append(',').Append(kv.Value);
                        }
                    }

                    if (skip)
                    {
                        continue;
                    }

                    var step = Dfs(nextDict, keyBuiler.ToString());
                    if (step != -1)
                    {
                        res = Math.Min(res, step + 1);
                    }
                }

                res = res == int.MaxValue ? -1 : res;
                cache[key] = res;
                return res;
            }

            return Dfs(targetDict, string.Empty);
        }

        #endregion

        #region 332. 重新安排行程

        //https://leetcode-cn.com/problems/reconstruct-itinerary/
        public IList<string> FindItinerary(IList<IList<string>> tickets)
        {
            if (tickets.Count <= 0)
            {
                return new string[0];
            }

            var ticketDict = new Dictionary<string, List<string>>();
            foreach (var ticket in tickets)
            {
                string from = ticket[0], to = ticket[1];
                if (!ticketDict.TryGetValue(from, out var tos))
                {
                    tos = new List<string>();
                    ticketDict[from] = tos;
                }

                tos.Add(to);
            }

            foreach (var list in ticketDict.Values)
            {
                list.Sort();
            }

            var paths = new List<string>();

            bool Dfs(string from)
            {
                if (ticketDict.Count <= 0)
                {
                    paths.Add(from);
                    return true;
                }

                if (!ticketDict.TryGetValue(from, out var tos) || tos.Count <= 0)
                {
                    return false;
                }

                paths.Add(from);
                for (var i = 0; i < tos.Count; i++)
                {
                    var to = tos[i];
                    tos.RemoveAt(i);
                    if (tos.Count <= 0)
                    {
                        ticketDict.Remove(from);
                    }

                    if (Dfs(to))
                    {
                        return true;
                    }

                    if (tos.Count <= 0)
                    {
                        ticketDict[from] = tos;
                    }

                    tos.Insert(i, to);
                }

                paths.RemoveAt(paths.Count - 1);
                return false;
            }

            Dfs("JFK");
            return paths;
        }

        #endregion

        #region 435. 无重叠区间

        //https://leetcode-cn.com/problems/non-overlapping-intervals/
        public int EraseOverlapIntervals(int[][] intervals)
        {
            if (intervals.Length <= 1)
            {
                return 0;
            }

            Array.Sort(intervals, Comparer<int[]>.Create((a, b) => a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]));

            //暴力解，超时
            int Dfs(int prev, int cur)
            {
                if (cur >= intervals.Length)
                {
                    return 0;
                }

                var no = int.MaxValue;
                if (prev < 0 || intervals[prev][1] <= intervals[cur][0])
                {
                    no = Dfs(cur, cur + 1);
                }

                var remove = Dfs(prev, cur + 1) + 1;
                return Math.Min(no, remove);
            }

            //动态规划
            //求出数组中最多不相交的区间数
            //res= 总区间数-不相交区间数
            int Dp()
            {
                var dp = new int[intervals.Length];
                dp[0] = 1;
                var ans = 0;
                for (int i = 1; i < intervals.Length; i++)
                {
                    var max = 0;
                    for (int j = i - 1; j >= 0; j--)
                    {
                        if (intervals[j][1] <= intervals[i][0])
                        {
                            max = Math.Max(dp[j], max);
                        }
                    }

                    dp[i] = max + 1;
                    ans = Math.Max(ans, dp[i]);
                }

                return intervals.Length - ans;
            }

            //贪心算法
            int Greedy()
            {
                int prev = 0, count = 0; //prev 应保留区间
                for (int i = 1; i < intervals.Length; i++)
                {
                    if (intervals[prev][1] > intervals[i][0]) //相交，需要移除数+1
                    {
                        if (intervals[prev][1] > intervals[i][1]) //判断应该移除覆盖范围较大区间
                        {
                            prev = i;
                        }

                        count++;
                    }
                    else //两个区间不相交，不需要移除，检查后面的是否有相交
                    {
                        prev = i;
                    }
                }

                return count;
            }

            return Max(Dfs(-1, 0), Dp(), Greedy());
        }

        #endregion

        #region 657. 机器人能否返回原点

        //https://leetcode-cn.com/problems/robot-return-to-origin/
        public bool JudgeCircle(string moves)
        {
            if (string.IsNullOrEmpty(moves))
            {
                return true;
            }

            int x = 0, y = 0;
            foreach (var ch in moves)
            {
                switch (ch)
                {
                    case 'U':
                        x--;
                        break;
                    case 'D':
                        x++;
                        break;
                    case 'L':
                        y--;
                        break;
                    case 'R':
                        y++;
                        break;
                }
            }

            return x == 0 && y == 0;
        }

        #endregion

        #region 673. 最长递增子序列的个数

        //https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/
        public int FindNumberOfLIS(int[] nums)
        {
            if (nums.Length <= 0)
            {
                return 0;
            }

            var dp = new int[nums.Length];
            var counters = new int[nums.Length];
            Array.Fill(counters, 1);
            var len = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    if (nums[i] <= nums[j])
                    {
                        continue;
                    }

                    if (dp[j] >= dp[i])
                    {
                        dp[i] = dp[j] + 1;
                        counters[i] = counters[j];
                    }
                    else if (dp[j] + 1 == dp[i])
                    {
                        counters[i] += counters[j];
                    }
                }

                len = Math.Max(len, dp[i]);
            }

            return counters.Where((c, i) => dp[i] == len).Sum();
        }

        #endregion

        #region 214. 最短回文串

        //https://leetcode-cn.com/problems/shortest-palindrome/
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
        //递归

        public string ShortestPalindromeII(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return s;
            }

            int l = 0;
            for (int i = s.Length - 1; i >= 0; i--)
            {
                if (s[l] == s[i])
                {
                    l++;
                }
            }

            if (l == s.Length)
            {
                return s;
            }

            if (l == 0)
            {
                return new string(s.Reverse().ToArray()) + s;
            }

            var keep = s.Substring(l);
            var reverseStr = new string(keep.Reverse().ToArray());
            return reverseStr + ShortestPalindromeII(s.Substring(0, l)) + keep;
        }

        #endregion

        #region 841. 钥匙和房间

        //https://leetcode-cn.com/problems/keys-and-rooms/
        public bool CanVisitAllRooms(IList<IList<int>> rooms)
        {
            if (rooms.Count <= 0)
            {
                return true;
            }

            var stack = new Stack<int>();
            var visited = new HashSet<int>();
            stack.Push(0);
            while (stack.TryPop(out var key))
            {
                if (!visited.Add(key))
                {
                    continue;
                }

                var keys = rooms[key];
                foreach (var k in keys)
                {
                    stack.Push(k);
                }
            }

            return visited.Count == rooms.Count;
        }

        #endregion

        #region 486. 预测赢家

        //https://leetcode-cn.com/problems/predict-the-winner/
        public bool PredictTheWinner(int[] nums)
        {
            if (nums.Length <= 1)
            {
                return true;
            }

            //当前者选择值减去下一人的选择值，最终结果>=0说明可以
            var cache = new int?[nums.Length, nums.Length];

            int Dfs(int l, int r)
            {
                if (l == r)
                {
                    return nums[l];
                }

                if (cache[l, r].HasValue)
                {
                    return cache[l, r].Value;
                }

                var res = Math.Max(nums[l] - Dfs(l + 1, r), nums[r] - Dfs(l, r - 1));
                cache[l, r] = res;
                return res;
            }

            return Dfs(0, nums.Length - 1) >= 0;
        }

        #endregion

        #region 145. 二叉树的后序遍历

        //https://leetcode-cn.com/problems/binary-tree-postorder-traversal/
        public IList<int> PostorderTraversal(TreeNode root)
        {
            var result = new List<int>();

            void Dfs(TreeNode node)
            {
                if (node == null)
                {
                    return;
                }

                Dfs(node.left);
                Dfs(node.right);
                result.Add(node.val);
            }

            var stack = new Stack<TreeNode>();
            var visited = new HashSet<TreeNode>();
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Peek();
                if (root.right == null || visited.Contains(root))
                {
                    stack.Pop();
                    result.Add(root.val);
                    root = null;
                }
                else
                {
                    visited.Add(root);
                    root = root.right;
                }
            }

            return result;
        }

        #endregion

        #region 165. 比较版本号

        //https://leetcode-cn.com/problems/compare-version-numbers/
        public int CompareVersion(string version1, string version2)
        {
            int v1 = 0, v2 = 0;
            while (v1 < version1.Length || v2 < version2.Length)
            {
                int n1 = 0, n2 = 0;
                while (v1 < version1.Length && version1[v1] != '.')
                {
                    n1 = n1 * 10 + version1[v1++] - '0';
                }

                while (v2 < version2.Length && version2[v2] != '.')
                {
                    n2 = n2 * 10 + version2[v2++] - '0';
                }

                if (n1 == n2)
                {
                    v1++;
                    v2++;
                }
                else if (n1 > n2)
                {
                    return 1;
                }
                else
                {
                    return -1;
                }
            }

            return 0;
        }

        #endregion

        #region 1567. 乘积为正数的最长子数组长度

        //https://leetcode-cn.com/problems/maximum-length-of-subarray-with-positive-product/
        public int GetMaxLen(int[] nums)
        {
            //动态规划
            int Dp()
            {
                if (nums.Length <= 0)
                {
                    return 0;
                }

                var dp = new int[nums.Length, 2];
                if (nums[0] > 0)
                {
                    dp[0, 0] = 1;
                }
                else if (nums[0] < 0)
                {
                    dp[0, 1] = 1;
                }

                var ans = dp[0, 0];
                for (int i = 1; i < nums.Length; i++)
                {
                    if (nums[i] == 0)
                    {
                        dp[i, 0] = dp[i, 1] = 0;
                    }
                    else if (nums[i] > 0)
                    {
                        dp[i, 0] = dp[i - 1, 0] + 1;
                        dp[i, 1] = dp[i - 1, 1] > 0 ? dp[i - 1, 1] + 1 : 0;
                    }
                    else
                    {
                        dp[i, 0] = dp[i - 1, 1] > 0 ? dp[i - 1, 1] + 1 : 0;
                        dp[i, 1] = dp[i - 1, 0] + 1;
                    }

                    ans = Math.Max(ans, dp[i, 0]);
                }

                return ans;
            }

            int res = 0, negative = 0, left = 0, right = 0;
            for (int i = 0, j = 0, e = nums.Length - 1; i < nums.Length; i++)
            {
                if (nums[i] == 0 || i == e)
                {
                    if (nums[i] < 0)
                    {
                        negative++;
                        right = i;
                    }

                    if (negative % 2 == 0)
                    {
                        res = Math.Max(res, i - j + (nums[i] == 0 ? 0 : 1));
                    }
                    else
                    {
                        res = Math.Max(res, Math.Max(right - j, i - left - (nums[i] == 0 ? 1 : 0)));
                    }

                    j = i + 1;
                    negative = 0;
                    left = j;
                    right = j;
                }
                else if (nums[i] < 0)
                {
                    negative++;
                    right = i;
                    if (nums[left] >= 0)
                    {
                        left = i;
                    }
                }
            }

            return res;
        }

        #endregion

        #region 45. 跳跃游戏 II

        //https://leetcode-cn.com/problems/jump-game-ii/
        //动态规划（超时）
        public int JumpIIByDp(int[] nums)
        {
            var cache = new int[nums.Length];

            int JumpDfs(int i)
            {
                if (i >= nums.Length - 1)
                {
                    return 0;
                }

                if (cache[i] != 0)
                {
                    return cache[i];
                }

                var step = int.MaxValue - 1;
                for (int s = 1; s <= nums[i]; s++)
                {
                    step = Math.Min(step, JumpDfs(i + s) + 1);
                }

                cache[i] = step;
                return step;
            }

            return JumpDfs(0);
        }

        //贪心算法
        public int JumpII(int[] nums)
        {
            var step = 0;
            int maxPos = 0, end = 0;
            for (int i = 0; i < nums.Length - 1; i++)
            {
                maxPos = Math.Max(maxPos, nums[i] + i);
                if (i == end)
                {
                    end = maxPos;
                    step++;
                }
            }

            return step;
        }

        #endregion

        #region 216. 组合总和 III

        //https://leetcode-cn.com/problems/combination-sum-iii/
        public IList<IList<int>> CombinationSum3(int k, int n)
        {
            var result = new List<IList<int>>();
            var seqs = new List<int>();

            void Dfs(int num, int target)
            {
                if (seqs.Count >= k || target <= 0)
                {
                    if (seqs.Count == k && target == 0)
                    {
                        result.Add(seqs.ToArray());
                    }

                    return;
                }

                for (int i = num; i < 10 && i <= target; i++)
                {
                    seqs.Add(i);
                    Dfs(i + 1, target - i);
                    seqs.RemoveAt(seqs.Count - 1);
                }
            }

            Dfs(1, n);
            return result;
        }

        #endregion

        #region 1302. 层数最深叶子节点的和

        //https://leetcode-cn.com/problems/deepest-leaves-sum/
        public int DeepestLeavesSum(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            var res = 0;
            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                var size = queue.Count;
                var sum = 0;
                while (size > 0)
                {
                    root = queue.Dequeue();
                    sum += root.val;
                    if (root.left != null)
                    {
                        queue.Enqueue(root.left);
                    }

                    if (root.right != null)
                    {
                        queue.Enqueue(root.right);
                    }

                    size--;
                }

                res = sum;
            }

            return res;
        }

        #endregion

        #region 637. 二叉树的层平均值

        //https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/
        public IList<double> AverageOfLevels(TreeNode root)
        {
            if (root == null)
            {
                return new double[0];
            }

            var result = new List<double>();
            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                var size = queue.Count;
                double sum = 0.0D;
                for (int i = size; i > 0; i--)
                {
                    root = queue.Dequeue();
                    if (root.left != null)
                    {
                        queue.Enqueue(root.left);
                    }

                    if (root.right != null)
                    {
                        queue.Enqueue(root.right);
                    }

                    sum += root.val;
                }

                result.Add(sum / size);
            }

            return result;
        }

        #endregion

        #region 面试题 03.01. 三合一

        //https://leetcode-cn.com/problems/three-in-one-lcci/
        public class TripleInOne
        {
            private int[] element;
            private int[] index = new int[3];
            private int[] count = new int[3];
            private int stackSize;

            public TripleInOne(int stackSize)
            {
                this.stackSize = stackSize;
                element = new int[stackSize * 3];
                for (var i = 0; i < index.Length; i++)
                {
                    index[i] = i * stackSize;
                }
            }

            public void Push(int stackNum, int value)
            {
                if (count[stackNum] == stackSize)
                {
                    return;
                }

                element[index[stackNum]] = value;
                index[stackNum]++;
                count[stackNum]++;
            }

            public int Pop(int stackNum)
            {
                if (IsEmpty(stackNum))
                {
                    return -1;
                }

                var ele = element[index[stackNum] - 1];
                index[stackNum]--;
                count[stackNum]--;
                return ele;
            }

            public int Peek(int stackNum)
            {
                if (IsEmpty(stackNum))
                {
                    return -1;
                }

                return element[index[stackNum] - 1];
            }

            public bool IsEmpty(int stackNum)
            {
                return count[stackNum] == 0;
            }
        }

        #endregion

        #region 面试题 05.03. 翻转数位

        //https://leetcode-cn.com/problems/reverse-bits-lcci/
        public int ReverseBits(int num)
        {
            if (num == 0)
            {
                return 1;
            }

            int size = 0, preSize = 0, res = 0;
            for (int i = 0; i < 32; i++)
            {
                var mask = 1 << i;
                if ((mask & num) == 0)
                {
                    res = Math.Max(res, size + preSize + 1);
                    preSize = size;
                    size = 0;
                }
                else
                {
                    size++;
                }
            }

            if (size > 0)
            {
                res = Math.Max(res, size + preSize + 1);
            }

            return Math.Min(res, 32);
        }

        #endregion

        #region 226. 翻转二叉树
        //https://leetcode-cn.com/problems/invert-binary-tree/
        public TreeNode InvertTree(TreeNode root)
        {
            if (root == null)
            {
                return root;
            }
            var tmp = root.left;
            root.left = InvertTree(root.right);
            root.right = InvertTree(tmp);
            return root;

        }
        #endregion

        #region 面试题 05.06. 整数转换

        //https://leetcode-cn.com/problems/convert-integer-lcci/
        public int ConvertInteger(int A, int B)
        {
            var xor = A ^ B;
            var res = 0;
            while (xor != 0)
            {
                xor &= (xor - 1);
                res++;
            }

            return res;
        }

        #endregion

        #region 面试题 05.04. 下一个数

        //https://leetcode-cn.com/problems/closed-number-lcci/
        public int[] FindClosedNumbers(int num)
        {
            //大  找到1和高于1位的0交换 后面的1排到最后
            //小 找到0和高于0位的1交换 后面的1排到最前
            var res = new int[] { -1, -1 };
            int val = num, flagBit = -1, bit = 0, oneSize = 0;
            while (val != 0)
            {
                if ((val & 1) == 1)
                {
                    flagBit = bit;
                    if (val == 1 && flagBit < 30)
                    {
                        int high = 1 << (flagBit + 1), low = int.MaxValue ^ (1 << flagBit);
                        res[0] = high | (num & low);
                        break;
                    }

                    oneSize++;
                }
                else if (flagBit != -1)
                {
                    var baseNum = num;
                    var high = 1 << bit;
                    baseNum = (baseNum >> bit) << bit;
                    for (int i = 0; i < oneSize - 1; i++)
                    {
                        baseNum |= (1 << i);
                    }

                    res[0] = high | baseNum;
                    break;
                }

                val >>= 1;
                bit++;
            }

            val = num;
            flagBit = -1;
            bit = 0;
            oneSize = 0;
            while (val != 0)
            {
                if ((val & 1) == 0)
                {
                    flagBit = bit;
                }
                else
                {
                    oneSize++;
                    if (flagBit != -1)
                    {
                        num >>= bit + 1;
                        num <<= 1;
                        for (int i = 0; i < oneSize; i++)
                        {
                            num = (num << 1) | 1;
                        }

                        num <<= bit - oneSize;
                        res[1] = num;
                        break;
                    }
                }

                val >>= 1;
                bit++;
            }

            return res;
        }

        #endregion

        #region 面试题 08.05. 递归乘法

        //https://leetcode-cn.com/problems/recursive-mulitply-lcci/
        public int Multiply(int A, int B)
        {
            if (A == 0 || B == 0)
            {
                return 0;
            }

            var flag = A > 0 && B > 0 || A < 0 && B < 0;

            int Dfs(int a, int b)
            {
                if (b == 1)
                {
                    return a;
                }

                var num = Dfs(a, b >> 1);
                return (b & 1) == 0 ? num + num : num + num + a;
            }

            var result = Dfs(Math.Abs(A), Math.Abs(B));
            return flag ? result : -result;
        }

        #endregion

        #region 面试题 08.02. 迷路的机器人

        //https://leetcode-cn.com/problems/robot-in-a-grid-lcci/
        public IList<IList<int>> PathWithObstacles(int[][] obstacleGrid)
        {
            var path = new List<IList<int>>();
            int targetX = obstacleGrid.Length - 1, targetY = obstacleGrid[0].Length - 1;
            var visited = new bool[obstacleGrid.Length, obstacleGrid[0].Length];
            //回溯+剪枝
            bool Dfs(int x, int y)
            {
                if (x < 0 || x >= obstacleGrid.Length || y < 0 || y >= obstacleGrid[0].Length ||
                    obstacleGrid[x][y] == 1 || visited[x, y])
                {
                    return false;
                }

                visited[x, y] = true;
                path.Add(new[] { x, y });
                if (x == targetX && y == targetY)
                {
                    return true;
                }

                if (Dfs(x + 1, y) || Dfs(x, y + 1))
                {
                    return true;
                }

                path.RemoveAt(path.Count - 1);
                return false;
            }
            //动态规划
            IList<IList<int>> Dp()
            {
                var paths = new IList<IList<int>>[obstacleGrid.Length, obstacleGrid[0].Length];
                paths[0, 0] = new IList<int>[] { new[] { 0, 0 } };
                for (int i = 0; i <= targetX; i++)
                {
                    for (int j = 0; j <= targetY; j++)
                    {
                        if (obstacleGrid[i][j] == 1)
                        {
                            paths[i, j] = new IList<int>[0];
                        }
                        else
                        {
                            if (i == 0 && j == 0)
                            {
                                continue;
                            }

                            if (i == 0)
                            {
                                if (paths[i, j - 1].Count > 0)
                                {
                                    var newPath = new List<IList<int>>(paths[i, j - 1]) { new[] { i, j } };
                                    paths[i, j] = newPath;
                                }
                                else
                                {
                                    paths[i, j] = paths[i, j - 1];
                                }
                            }
                            else if (j == 0)
                            {
                                paths[i, j] = paths[i - 1, j].Count > 0
                                    ? new List<IList<int>>(paths[i - 1, j]) { new[] { i, j } }
                                    : paths[i - 1, j];
                            }
                            else
                            {
                                if (paths[i - 1, j].Count <= 0 && paths[i, j - 1].Count <= 0)
                                {
                                    paths[i, j] = paths[i - 1, j];
                                }
                                else if (paths[i - 1, j].Count <= 0)
                                {
                                    paths[i, j] = new List<IList<int>>(paths[i, j - 1]) { new[] { i, j } };
                                }
                                else if (paths[i, j - 1].Count <= 0)
                                {
                                    paths[i, j] = new List<IList<int>>(paths[i - 1, j]) { new[] { i, j } };
                                }
                                else
                                {
                                    paths[i, j] = new List<IList<int>>(paths[i - 1, j].Count > paths[i, j - 1].Count
                                        ? paths[i, j - 1]
                                        : paths[i - 1, j]) { new[] { i, j } };
                                }
                            }
                        }
                    }
                }

                return paths[targetX, targetY];
            }
            Dfs(0, 0);
            return path;
        }

        #endregion


    }
}