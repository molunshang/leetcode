using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
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
                return new[] {-1, -1};
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

            return new[] {left, right};
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
                    chars.Add((char) ('a' + j));
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
            var res = new int[] {-1, -1};
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
                path.Add(new[] {x, y});
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
                paths[0, 0] = new IList<int>[] {new[] {0, 0}};
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
                                    var newPath = new List<IList<int>>(paths[i, j - 1]) {new[] {i, j}};
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
                                    ? new List<IList<int>>(paths[i - 1, j]) {new[] {i, j}}
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
                                    paths[i, j] = new List<IList<int>>(paths[i, j - 1]) {new[] {i, j}};
                                }
                                else if (paths[i, j - 1].Count <= 0)
                                {
                                    paths[i, j] = new List<IList<int>>(paths[i - 1, j]) {new[] {i, j}};
                                }
                                else
                                {
                                    paths[i, j] = new List<IList<int>>(paths[i - 1, j].Count > paths[i, j - 1].Count
                                        ? paths[i, j - 1]
                                        : paths[i - 1, j]) {new[] {i, j}};
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

        #region 685. 冗余连接 II

        //https://leetcode-cn.com/problems/redundant-connection-ii/
        class UnionSet
        {
            private int[] edges;

            public UnionSet(int n)
            {
                edges = new int[n];
                for (int i = 0; i < n; i++)
                {
                    edges[i] = i;
                }
            }

            public int GetParent(int n)
            {
                if (edges[n] == n)
                    return n;
                edges[n] = GetParent(edges[n]);
                return edges[n];
            }

            public void Union(int parent, int child)
            {
                edges[GetParent(child)] = GetParent(parent);
            }

            public bool IsConcat(int a, int b)
            {
                return GetParent(a) == GetParent(b);
            }
        }

        public int[] FindRedundantDirectedConnection(int[][] edges)
        {
            //暴力解
            int[] ForceDfs()
            {
                var items = new HashSet<int>[edges.Length + 1];
                var starts = new HashSet<int>();
                for (var i = 0; i < edges.Length; i++)
                {
                    int parent = edges[i][0], child = edges[i][1];
                    if (items[parent] == null)
                    {
                        items[parent] = new HashSet<int>();
                    }

                    starts.Add(parent);
                    items[parent].Add(child);
                }

                var visited = new HashSet<int>();
                var queue = new Queue<int>();

                bool Bfs()
                {
                    foreach (var start in starts)
                    {
                        queue.Enqueue(start);
                        while (queue.TryDequeue(out var node) && visited.Count < edges.Length)
                        {
                            if (!visited.Add(node))
                            {
                                break;
                            }

                            if (items[node] == null || items[node].Count <= 0)
                            {
                                continue;
                            }

                            foreach (var n in items[node])
                            {
                                queue.Enqueue(n);
                            }
                        }

                        if (visited.Count >= edges.Length)
                        {
                            return true;
                        }

                        visited.Clear();
                        queue.Clear();
                    }

                    return false;
                }

                for (var i = edges.Length - 1; i >= 0; i--)
                {
                    int parent = edges[i][0], child = edges[i][1];
                    items[parent].Remove(child);
                    if (Bfs())
                    {
                        return edges[i];
                    }

                    items[parent].Add(child);
                }

                return null;
            }

            var nodeCount = edges.Length;
            var unionSet = new UnionSet(nodeCount + 1);
            var parents = new int[nodeCount + 1];
            for (var i = 0; i < parents.Length; i++)
            {
                parents[i] = i;
            }

            int confilct = -1, cycle = -1;
            for (int i = 0; i < nodeCount; i++)
            {
                var edge = edges[i];
                int node1 = edge[0], node2 = edge[1];
                if (parents[node2] != node2)
                {
                    confilct = i;
                }
                else
                {
                    parents[node2] = node1;
                    if (unionSet.GetParent(node1) == unionSet.GetParent(node2))
                    {
                        cycle = i;
                    }
                    else
                    {
                        unionSet.Union(node1, node2);
                    }
                }
            }

            if (confilct < 0)
            {
                return new[] {edges[cycle][0], edges[cycle][1]};
            }

            return cycle < 0
                ? new[] {edges[confilct][0], edges[confilct][1]}
                : new[] {parents[edges[confilct][1]], edges[confilct][1]};
        }

        #endregion

        #region 684. 冗余连接

        //https://leetcode-cn.com/problems/redundant-connection/
        public int[] FindRedundantConnection(int[][] edges)
        {
            var nodeCount = edges.Length;
            var unionSet = new UnionSet(nodeCount + 1);
            var rm = -1;
            for (var i = 0; i < edges.Length; i++)
            {
                int n1 = edges[i][0], n2 = edges[i][1];
                int f1 = unionSet.GetParent(n1), f2 = unionSet.GetParent(n2);
                if (f1 == f2)
                {
                    rm = i;
                }
                else
                {
                    unionSet.Union(f1, f2);
                }
            }

            return edges[rm];
        }

        #endregion

        #region 721. 账户合并

        //https://leetcode-cn.com/problems/accounts-merge/
        public IList<IList<string>> AccountsMerge(IList<IList<string>> accounts)
        {
            var result = new List<IList<string>>();
            var dict = new Dictionary<string, IList<int>>();
            for (int i = 0; i < accounts.Count; i++)
            {
                var account = accounts[i];
                for (int j = 1; j < account.Count; j++)
                {
                    var email = account[j];
                    if (!dict.TryGetValue(email, out var items))
                    {
                        items = new List<int>();
                        dict[email] = items;
                    }

                    items.Add(i);
                }
            }

            var visited = new HashSet<int>();
            for (int i = 0; i < accounts.Count; i++)
            {
                if (visited.Contains(i))
                {
                    continue;
                }

                var union = new List<string>();
                union.Add(accounts[i][0]);
                Dfs(i, union);
                union.Sort(1, union.Count - 1, StringComparer.Ordinal);
                result.Add(union.Distinct().ToArray());
            }

            void Dfs(int index, List<string> items)
            {
                if (!visited.Add(index))
                {
                    return;
                }

                var account = accounts[index];
                for (int i = 1; i < account.Count; i++)
                {
                    var email = account[i];
                    items.Add(email);
                    if (dict.TryGetValue(email, out var next))
                    {
                        foreach (var n in next)
                        {
                            Dfs(n, items);
                        }
                    }
                }
            }

            return result;
        }

        #endregion

        #region 784. 字母大小写全排列

        //https://leetcode-cn.com/problems/letter-case-permutation/
        public IList<string> LetterCasePermutation(string s)
        {
            var result = new List<string>();
            var chars = s.ToCharArray();

            void Dfs(int index)
            {
                result.Add(new string(chars));
                if (index >= chars.Length)
                {
                    return;
                }

                for (int i = index; i < chars.Length; i++)
                {
                    var old = chars[i];
                    if (char.IsLower(old))
                    {
                        chars[i] = char.ToUpper(old);
                    }
                    else if (char.IsUpper(old))
                    {
                        chars[i] = char.ToLower(old);
                    }
                    else
                    {
                        continue;
                    }

                    Dfs(i + 1);
                    chars[i] = old;
                }
            }

            Dfs(0);
            return result;
        }

        #endregion

        #region 496. 下一个更大元素 I

        //https://leetcode-cn.com/problems/next-greater-element-i/
        public int[] NextGreaterElement(int[] nums1, int[] nums2)
        {
            var stack = new Stack<int>();
            var dict = new Dictionary<int, int>();
            foreach (var n in nums2)
            {
                while (stack.Count > 0 && n > stack.Peek())
                {
                    dict[stack.Pop()] = n;
                }

                stack.Push(n);
            }

            var res = new int[nums1.Length];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = dict.TryGetValue(nums1[i], out var n) ? n : -1;
            }

            return res;
        }

        #endregion

        #region 224. 基本计算器

        //https://leetcode-cn.com/problems/basic-calculator/
        public int BasicCalculate(string s)
        {
            var nums = new Stack<int>();
            var operators = new Stack<char>();
            for (var i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                if (ch == ' ')
                {
                    continue;
                }

                if (char.IsDigit(ch))
                {
                    var num = ch - '0';
                    while (i < s.Length - 1 && char.IsDigit(s[i + 1]))
                    {
                        i++;
                        num = num * 10 + (s[i] - '0');
                    }

                    nums.Push(operators.Count > 0 && operators.Peek() == '-' ? -num : num);
                }
                else if (ch == ')')
                {
                    var num = nums.Pop();
                    while (operators.TryPop(out ch))
                    {
                        if (ch == '(')
                        {
                            nums.Push(operators.Count > 0 && operators.Peek() == '-' ? -num : num);
                            break;
                        }

                        num += nums.Pop();
                    }
                }
                else
                {
                    operators.Push(ch);
                }
            }

            var res = nums.Pop();
            while (nums.TryPop(out var num))
            {
                res += num;
            }

            return res;
        }

        #endregion

        #region 173. 二叉搜索树迭代器

        //https://leetcode-cn.com/problems/binary-search-tree-iterator/
        public class BSTIterator
        {
            private Stack<TreeNode> stack = new Stack<TreeNode>();
            private TreeNode current;

            public BSTIterator(TreeNode root)
            {
                current = root;
            }

            /** @return the next smallest number */
            public int Next()
            {
                var num = -1;
                while (stack.Count > 0 || current != null)
                {
                    while (current != null)
                    {
                        stack.Push(current);
                        current = current.left;
                    }

                    current = stack.Pop();
                    num = current.val;
                    current = current.right;
                    break;
                }

                return num;
            }

            /** @return whether we have a next smallest number */
            public bool HasNext()
            {
                return stack.Count > 0 || current != null;
            }
        }

        #endregion

        #region 316. 去除重复字母

        //https://leetcode-cn.com/problems/remove-duplicate-letters/
        public string RemoveDuplicateLetters(string s)
        {
            //记录相同字符最后一次的出现位置
            var indexDict = new Dictionary<char, int>();
            for (var i = 0; i < s.Length; i++)
            {
                indexDict[s[i]] = i;
            }

            var stack = new Stack<char>();
            var charSet = new HashSet<char>();
            for (var i = 0; i < s.Length; i++)
            {
                //过滤掉已经选择的字符
                if (charSet.Contains(s[i]))
                {
                    continue;
                }

                //单调栈，移除队列中大于当前字符同时后面依旧会出现的字符
                while (stack.TryPeek(out var ch) && s[i] < ch && i < indexDict[ch])
                {
                    charSet.Remove(stack.Pop());
                }

                stack.Push(s[i]);
                charSet.Add(s[i]);
            }

            var chars = new char[stack.Count];
            for (int i = chars.Length - 1; i >= 0; i--)
            {
                chars[i] = stack.Pop();
            }

            return new string(chars);
        }

        #endregion

        #region 404. 左叶子之和

        //https://leetcode-cn.com/problems/sum-of-left-leaves/
        public int SumOfLeftLeaves(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            var left = root.left;
            int sum;
            if (left != null && left.left == null && left.right == null)
            {
                sum = left.val;
            }
            else
            {
                sum = SumOfLeftLeaves(left);
            }

            sum += SumOfLeftLeaves(root.right);
            return sum;
        }

        #endregion

        #region 904. 水果成篮

        //https://leetcode-cn.com/problems/fruit-into-baskets/
        public int TotalFruit(int[] tree)
        {
            var fruits = new Dictionary<int, int>();
            var ans = 0;
            for (int i = 0, j = 0; i < tree.Length; i++)
            {
                var fruit = tree[i];
                if (!fruits.TryGetValue(fruit, out var count))
                {
                    count = 0;
                }

                fruits[fruit] = count + 1;
                while (fruits.Count > 2 && j < i)
                {
                    count = fruits[tree[j]];
                    if (count <= 1)
                    {
                        fruits.Remove(tree[j]);
                    }
                    else
                    {
                        fruits[tree[j]] = count - 1;
                    }

                    j++;
                }

                ans = Math.Max(ans, i - j + 1);
            }

            return ans;
        }

        #endregion

        #region 1038. 从二叉搜索树到更大和树

        //https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree/
        public TreeNode BstToGst(TreeNode root)
        {
            var node = root;
            var stack = new Stack<TreeNode>();
            TreeNode prev = null;
            while (node != null || stack.Count > 0)
            {
                while (node != null)
                {
                    stack.Push(node);
                    node = node.right;
                }

                node = stack.Pop();
                if (prev != null)
                {
                    node.val += prev.val;
                }

                prev = node;
                node = node.left;
            }

            return root;
        }

        #endregion

        #region 106. 从中序与后序遍历序列构造二叉树

        //https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
        public TreeNode BuildTreeFromInAndPost(int[] inorder, int[] postorder)
        {
            TreeNode Dfs(int il, int ir, int pl, int pr)
            {
                if (pl >= pr)
                {
                    return pl > pr ? null : new TreeNode(postorder[pl]);
                }

                var root = new TreeNode(postorder[pr]);
                var imid = -1;
                for (int i = il; i <= ir; i++)
                {
                    if (inorder[i] != postorder[pr])
                        continue;
                    imid = i;
                    break;
                }

                if (imid > -1)
                {
                    var pmid = pl + imid - il;
                    root.left = Dfs(il, imid - 1, pl, pmid - 1);
                    root.right = Dfs(imid + 1, ir, pmid, pr - 1);
                }

                return root;
            }

            var indexDict = new Dictionary<int, int>();
            for (var i = 0; i < inorder.Length; i++)
            {
                indexDict[inorder[i]] = i;
            }

            var rootIndex = postorder.Length - 1;

            TreeNode BuildTree(int l, int r)
            {
                if (l > r)
                {
                    return null;
                }

                var root = new TreeNode(postorder[rootIndex--]);
                root.right = BuildTree(indexDict[root.val] + 1, r);
                root.left = BuildTree(l, indexDict[root.val] - 1);
                return root;
            }

            return BuildTree(0, rootIndex) ?? Dfs(0, inorder.Length - 1, 0, postorder.Length - 1);
        }

        #endregion

        #region 117. 填充每个节点的下一个右侧节点指针 II

        //https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/
        public Node ConnectII(Node root)
        {
            void Connect(Node node)
            {
                if (node == null)
                {
                    return;
                }

                Node firstNode = null, lastNode = null;
                while (node != null)
                {
                    Node left = node.left, right = node.right;
                    if (left != null && right != null)
                    {
                        if (firstNode == null)
                        {
                            firstNode = left;
                        }

                        if (lastNode != null)
                        {
                            lastNode.next = left;
                        }

                        left.next = right;
                        lastNode = right;
                    }
                    else if (left != null)
                    {
                        if (firstNode == null)
                        {
                            firstNode = left;
                        }

                        if (lastNode != null)
                        {
                            lastNode.next = left;
                        }

                        lastNode = left;
                    }
                    else if (right != null)
                    {
                        if (firstNode == null)
                        {
                            firstNode = right;
                        }

                        if (lastNode != null)
                        {
                            lastNode.next = right;
                        }

                        lastNode = right;
                    }

                    node = node.next;
                }

                Connect(firstNode);
            }

            Connect(root);

            //作为链表进行链接
            var head = root;
            while (head != null)
            {
                var level = new Node(-1);
                var tail = level;
                while (head != null)
                {
                    if (head.left != null)
                    {
                        tail.next = head.left;
                        tail = tail.next;
                    }

                    if (head.right != null)
                    {
                        tail.next = head.right;
                        tail = tail.next;
                    }

                    head = head.next;
                }

                head = level.next;
            }

            return root;
        }

        #endregion

        #region 129. 求根到叶子节点数字之和

        //https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/
        public int SumNumbers(TreeNode root)
        {
            int Dfs(TreeNode node, int num)
            {
                while (true)
                {
                    if (node == null)
                    {
                        return num;
                    }

                    num = num * 10 + node.val;
                    if (node.right == null)
                    {
                        node = node.left;
                        continue;
                    }

                    if (node.left != null)
                    {
                        return Dfs(node.left, num) + Dfs(node.right, num);
                    }

                    node = node.right;
                }
            }

            return Dfs(root, 0);
        }

        #endregion

        #region 429. N叉树的层序遍历

        //https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/
        public IList<IList<int>> LevelOrder(Node root)
        {
            if (root == null)
            {
                return new IList<int>[0];
            }

            var result = new List<IList<int>>();
            var queue = new Queue<Node>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                var size = queue.Count;
                var items = new List<int>();
                while (size > 0)
                {
                    size--;
                    root = queue.Dequeue();
                    items.Add(root.val);
                    if (root.children != null && root.children.Count > 0)
                    {
                        foreach (var child in root.children)
                        {
                            queue.Enqueue(child);
                        }
                    }
                }

                result.Add(items);
            }

            return result;
        }

        #endregion

        #region 199. 二叉树的右视图

        //https://leetcode-cn.com/problems/binary-tree-right-side-view/
        public IList<int> RightSideView(TreeNode root)
        {
            var result = new List<int>();

            void Dfs(TreeNode node, int depth)
            {
                if (node == null)
                {
                    return;
                }

                if (result.Count <= depth)
                {
                    result.Add(node.val);
                }
                else
                {
                    result[depth] = node.val;
                }

                Dfs(node.left, depth + 1);
                Dfs(node.right, depth + 1);
            }

            Dfs(root, 0);
            return result;
        }

        #endregion

        #region 222. 完全二叉树的节点个数

        //https://leetcode-cn.com/problems/count-complete-tree-nodes/
        public int CountNodes(TreeNode root)
        {
            //todo 二分计算二叉树节点
            if (root == null)
            {
                return 0;
            }

            return CountNodes(root.left) + CountNodes(root.right) + 1;
        }

        #endregion

        #region 700. 二叉搜索树中的搜索

        //https://leetcode-cn.com/problems/search-in-a-binary-search-tree/
        public TreeNode SearchBST(TreeNode root, int val)
        {
            while (true)
            {
                if (root == null || root.val == val)
                {
                    return root;
                }


                root = root.val > val ? root.left : root.right;
            }
        }

        #endregion

        #region 701. 二叉搜索树中的插入操作

        //https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/
        public TreeNode InsertIntoBST(TreeNode root, int val)
        {
            if (root == null)
            {
                return new TreeNode(val);
            }

            var node = root;
            while (node != null)
            {
                if (node.val > val)
                {
                    if (node.left == null)
                    {
                        node.left = new TreeNode(val);
                        break;
                    }

                    node = node.left;
                }
                else
                {
                    if (node.right == null)
                    {
                        node.right = new TreeNode(val);
                        break;
                    }

                    node = node.right;
                }
            }

            return root;
        }

        #endregion

        #region 1130. 叶值的最小代价生成树

        //https://leetcode-cn.com/problems/minimum-cost-tree-from-leaf-values/
        public int MctFromLeafValues(int[] arr)
        {
            var len = arr.Length;
            //1.获取区间范围内的最大值
            var maxVals = new int[len, len];
            for (int i = 0; i < len; i++)
            {
                var max = int.MinValue;
                for (int j = i; j < len; j++)
                {
                    max = Math.Max(max, arr[j]);
                    maxVals[i, j] = max;
                }
            }

            //2.二叉树划分区间分别进行计算
            //[0,1][2,3]…………[n-1,n]
            //[0,2][3,5]…………[n-5,n]
            var dp = new int[len, len];
            for (int i = 1; i < len; i++)
            {
                for (int j = 0; j < len - i; j++)
                {
                    dp[j, j + i] = int.MaxValue;
                    for (int k = j; k < j + i; k++)
                    {
                        dp[j, j + i] = Math.Min(dp[j, j + i],
                            dp[j, k] + dp[k + 1, j + i] + maxVals[j, k] * maxVals[k + 1, j + i]);
                    }
                }
            }

            return dp[0, len - 1];
        }

        #endregion

        #region 968. 监控二叉树

        //https://leetcode-cn.com/problems/binary-tree-cameras/
        public int MinCameraCover(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            //三种情况
            //0.root安装监控，覆盖整个树
            //1.覆盖整个树，不管root是否安装
            //2.覆盖左右子树，不管root是否安装
            //递推过程：https://leetcode-cn.com/problems/binary-tree-cameras/solution/shou-hua-tu-jie-cong-di-gui-you-hua-dao-dong-tai-g/
            int[] Dfs(TreeNode node)
            {
                if (node == null)
                {
                    return new[] {int.MaxValue / 2, 0, 0};
                }

                var left = Dfs(node.left);
                var right = Dfs(node.right);
                var ans = new int[3];
                ans[0] = left[2] + right[2] + 1;
                ans[1] = Math.Min(ans[0], Math.Min(left[0] + right[1], left[1] + right[0]));
                ans[2] = Math.Min(ans[0], left[1] + right[1]);
                return ans;
            }

            var res = Dfs(root);
            return res[1];
        }

        #endregion

        #region 979. 在二叉树中分配硬币

        //https://leetcode-cn.com/problems/distribute-coins-in-binary-tree/
        public int DistributeCoins(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            //1.遇到0，找>1
            //2.遇到>1,找0
            var ans = 0;

            int Dfs(TreeNode node)
            {
                if (node == null)
                {
                    return 0;
                }

                var l = Dfs(node.left);
                var r = Dfs(node.right);
                ans += Math.Abs(l) + Math.Abs(r); //累加node需要经过的节点数
                return node.val + l + r - 1; //获取到当前位置需要的节点数
            }

            Dfs(root);
            return ans;
        }

        #endregion

        #region 面试题 05.07. 配对交换

        //https://leetcode-cn.com/problems/exchange-lcci/
        public int ExchangeBits(int num)
        {
            //计算mask数值
            int mask0 = 1, mask1 = 2;
            for (int i = 0; i < 15; i++)
            {
                mask0 |= mask0 << 2; //偶数位为1
                mask1 |= mask1 << 2; //奇数位为1
            }

            //0xaaaaaaaa 奇数位全为1数
            //0x55555555 偶数位全为1数
            //&操作保留奇数位和偶数位，同时偶数位左移和奇数位右移得到最后结果
            return (int) (((num & 0xaaaaaaaa) >> 1) | ((num & 0x55555555) << 1));
        }

        #endregion

        #region 面试题 08.04. 幂集/78. 子集

        //https://leetcode-cn.com/problems/power-set-lcci/
        //https://leetcode-cn.com/problems/subsets/

        public IList<IList<int>> Subsets(int[] nums)
        {
            var result = new List<IList<int>>();
            var subs = new List<int>();

            void Dfs(int i)
            {
                result.Add(subs.ToArray());
                if (i >= nums.Length)
                {
                    return;
                }

                for (int j = i; j < nums.Length; j++)
                {
                    subs.Add(nums[j]);
                    Dfs(j + 1);
                    subs.RemoveAt(subs.Count - 1);
                }
            }

            Dfs(0);
            return result;
        }

        #endregion


        #region 面试题38. 字符串的排列/面试题 08.07. 无重复字符串的排列组合

        //https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/
        //https://leetcode-cn.com/problems/permutation-i-lcci/

        public string[] Permutation(string s)
        {
            var result = new List<string>();
            var chars = s.ToCharArray();

            void Swap(int i, int j)
            {
                var tmp = chars[j];
                chars[j] = chars[i];
                chars[i] = tmp;
            }

            void Dfs(int i)
            {
                if (i >= chars.Length)
                {
                    result.Add(new string(chars));
                    return;
                }

                for (int j = i; j < chars.Length; j++)
                {
                    Swap(i, j);
                    Dfs(i + 1);
                    Swap(i, j);
                }
            }

            Dfs(0);
            return result.ToArray();
        }

        #endregion

        #region 面试题 08.08. 有重复字符串的排列组合

        //https://leetcode-cn.com/problems/permutation-ii-lcci/
        public string[] PermutationII(string s)
        {
            var result = new List<string>();
            var chars = s.OrderBy(c => c).ToArray();

            void Swap(int i, int j)
            {
                var tmp = chars[j];
                chars[j] = chars[i];
                chars[i] = tmp;
            }

            void Dfs(int i)
            {
                if (i >= chars.Length)
                {
                    result.Add(new string(chars));
                    return;
                }

                for (int j = i; j < chars.Length; j++)
                {
                    if (j > i && (chars[j] == chars[j - 1] || chars[j] == chars[i]))
                    {
                        continue;
                    }

                    Swap(i, j);
                    Dfs(i + 1);
                    Swap(i, j);
                }
            }

            Dfs(0);
            return result.ToArray();
        }

        #endregion

        #region 面试题 08.13. 堆箱子

        //https://leetcode-cn.com/problems/pile-box-lcci/
        public int PileBox(int[][] box)
        {
            if (box.Length <= 0)
            {
                return 0;
            }

            Array.Sort(box, Comparer<int[]>.Create((a, b) =>
            {
                var cmp = b[0] - a[0];
                if (cmp != 0)
                {
                    return cmp;
                }

                cmp = b[1] - a[1];
                if (cmp == 0)
                {
                    cmp = b[2] - a[2];
                }

                return cmp;
            }));
            var mem = new int[box.Length];

            int Dfs(int i)
            {
                if (i >= box.Length)
                {
                    return 0;
                }

                if (mem[i] != 0)
                {
                    return mem[i];
                }

                var cur = box[i];
                var res = cur[2];
                for (int j = i + 1; j < box.Length; j++)
                {
                    var next = box[j];
                    if (cur[0] <= next[0] || cur[1] <= next[1] || cur[2] <= next[2])
                    {
                        continue;
                    }

                    res = Math.Max(res, Dfs(j) + cur[2]);
                }

                mem[i] = res;
                return res;
            }

            var ans = 0;
            for (int i = 0; i < box.Length; i++)
            {
                ans = Math.Max(ans, Dfs(i));
            }

            return ans;
        }

        #endregion

        #region 42. 接雨水/面试题 17.21. 直方图的水量

        //https://leetcode-cn.com/problems/trapping-rain-water/
        //https://leetcode-cn.com/problems/volume-of-histogram-lcci/submissions/
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
            if (height.Length <= 2)
            {
                return 0;
            }

            var res = 0;
            int[] leftMax = new int[height.Length], rightMax = new int[height.Length];
            leftMax[0] = height[0];
            rightMax[rightMax.Length - 1] = height[height.Length - 1];
            for (int i = 1, j = height.Length - 2; i < height.Length; i++, j--)
            {
                leftMax[i] = Math.Max(leftMax[i - 1], height[i]);
                rightMax[j] = Math.Max(rightMax[j + 1], height[j]);
            }

            for (int i = 1; i < height.Length - 1; i++)
            {
                res += Math.Min(leftMax[i], rightMax[i]) - height[i];
            }

            return res;
        }

        #endregion

        #region 面试题 10.05. 稀疏数组搜索

        //https://leetcode-cn.com/problems/sparse-array-search-lcci/
        public int FindString(string[] words, string s)
        {
            int Find(int l, int r)
            {
                if (l >= r)
                {
                    return l > r ? -1 : words[l] == s ? l : -1;
                }

                var mid = (l + r) / 2;
                if (string.IsNullOrEmpty(words[mid]))
                {
                    var index = Find(l, mid - 1);
                    return index == -1 ? Find(mid + 1, r) : index;
                }

                var cmp = s.CompareTo(words[mid]);
                if (cmp == 0)
                {
                    return mid;
                }

                return cmp < 0 ? Find(l, mid - 1) : Find(mid + 1, r);
            }

            int FindRemoveEmpty(int l, int r)
            {
                while (l <= r)
                {
                    var m = l + (r - l) / 2;
                    var tmp = m;
                    while (m < r && string.IsNullOrEmpty(words[m]))
                    {
                        m++;
                    }

                    if (string.IsNullOrEmpty(words[m]))
                    {
                        r = tmp - 1;
                        continue;
                    }

                    var cmp = string.Compare(s, words[m], StringComparison.Ordinal);
                    if (cmp == 0)
                    {
                        return m;
                    }

                    if (cmp < 0)
                    {
                        r = tmp - 1;
                    }
                    else
                    {
                        l = m + 1;
                    }
                }

                return -1;
            }

            return Find(0, words.Length - 1);
        }

        #endregion

        #region 324. 摆动排序 II/面试题 10.11. 峰与谷

        //https://leetcode-cn.com/problems/wiggle-sort-ii/
        //https://leetcode-cn.com/problems/peaks-and-valleys-lcci/

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

        public void WiggleSortByOn(int[] nums)
        {
            for (var i = 1; i < nums.Length; i++)
            {
                if (i % 2 == 0)
                {
                    //i为波峰 i-1为波谷 此时i-1已经小于i-2,所有i小于i-2
                    if (nums[i] < nums[i - 1])
                    {
                        Swap(nums, i, i - 1);
                        //交换
                    }
                }
                else
                {
                    //i为波谷 i-1为波峰 此时i-1大于i-2,所有i大于i-2
                    if (nums[i] > nums[i - 1])
                    {
                        Swap(nums, i, i - 1);
                    }
                }
            }
        }

        #endregion

        #region 面试题 17.04. 消失的数字（无序数组）

        //https://leetcode-cn.com/problems/missing-number-lcci/
        public int MissingNumber_NoSort(int[] nums)
        {
            //位运算
            int ByBit()
            {
                var ans = nums.Length;
                for (int i = 0; i < nums.Length; i++)
                {
                    ans = ans ^ i ^ nums[i];
                }

                return ans;
            }

            var sum = (nums.Length + 1) * nums.Length / 2;
            return nums.Aggregate(sum, (current, num) => current - num);
        }

        #endregion

        #region 501. 二叉搜索树中的众数

        //https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/
        public int[] FindMode(TreeNode root)
        {
            if (root == null)
            {
                return new int[0];
            }

            var res = new List<int>();
            int max = 0, count = 0;
            var stack = new Stack<TreeNode>();
            TreeNode prev = null;
            while (stack.Count > 0 || root != null)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();

                if (prev == null || prev.val == root.val)
                {
                    count++;
                }
                else
                {
                    count = 1;
                }

                if (count == max)
                {
                    res.Add(root.val);
                }
                else if (count > max)
                {
                    max = count;
                    res.Clear();
                    res.Add(root.val);
                }

                prev = root;
                root = root.right;
            }

            return res.ToArray();
        }

        #endregion


        #region 面试题 17.05.  字母与数字

        //https://leetcode-cn.com/problems/find-longest-subarray-lcci/
        public string[] FindLongestSubarray(string[] array)
        {
            //暴力解
            string[] Force()
            {
                int[] nums = new int[array.Length + 1], letters = new int[array.Length + 1];
                for (var i = 1; i <= array.Length; i++)
                {
                    if (char.IsDigit(array[i - 1][0]))
                    {
                        nums[i] = nums[i - 1] + 1;
                        letters[i] = letters[i - 1];
                    }
                    else
                    {
                        letters[i] = letters[i - 1] + 1;
                        nums[i] = nums[i - 1];
                    }
                }

                int len = int.MinValue, index = -1;
                for (int l = array.Length % 2 == 0 ? array.Length : array.Length - 1; l > 1 && l > len; l -= 2)
                {
                    for (int i = 0, j = i + l; j <= array.Length && l > len; i++, j++)
                    {
                        int ncount = nums[j] - nums[i], lcount = letters[j] - letters[i];
                        if (ncount == lcount && l > len)
                        {
                            len = l;
                            index = i;
                        }
                    }
                }

                if (len <= 0)
                {
                    return new string[0];
                }

                var result = new string[len];
                Array.Copy(array, index, result, 0, len);
                return result;
            }

            //前缀和
            string[] PrefixSum()
            {
                var dict = new Dictionary<int, int>();
                int len = int.MinValue, index = -1, count = 0;
                for (var i = 0; i < array.Length; i++)
                {
                    var isNum = char.IsDigit(array[i][0]);
                    count += isNum ? 1 : -1;
                    if (dict.TryGetValue(count, out var s) || count == 0)
                    {
                        if (count == 0)
                        {
                            s = -1;
                        }

                        var cur = i - s;
                        if (cur > len)
                        {
                            len = cur;
                            index = s > -1 && isNum == char.IsDigit(array[s][0]) ? s : s + 1;
                        }
                    }
                    else
                    {
                        dict[count] = i;
                    }
                }

                if (len <= 0)
                {
                    return count == 0 ? array : new string[0];
                }

                var result = new string[len];
                Array.Copy(array, index, result, 0, len);
                return result;
            }

            return PrefixSum();
        }

        #endregion

        #region 567. 字符串的排列

        //https://leetcode-cn.com/problems/permutation-in-string/
        public bool CheckInclusion(string s1, string s2)
        {
            if (s1.Length > s2.Length)
            {
                return false;
            }

            var dict = s1.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
            var sub = new Dictionary<char, int>();
            int size = 0, n;
            for (int i = 0, j = 0; i < s2.Length; i++)
            {
                var c = s2[i];
                if (dict.ContainsKey(c))
                {
                    if (sub.TryGetValue(c, out n))
                    {
                        n++;
                    }
                    else
                    {
                        n = 1;
                    }

                    sub[c] = n;
                    if (dict[c] == n)
                    {
                        size++;
                    }
                }

                var len = i - j + 1;
                if (len < s1.Length)
                {
                    continue;
                }

                if (size == dict.Count)
                {
                    return true;
                }

                c = s2[j++];
                if (sub.TryGetValue(c, out n))
                {
                    if (n == dict[c])
                    {
                        size--;
                    }

                    sub[c] = n - 1;
                }
            }

            return false;
        }

        #endregion

        #region 1288. 删除被覆盖区间

        //https://leetcode-cn.com/problems/remove-covered-intervals/
        public int RemoveCoveredIntervals(int[][] intervals)
        {
            if (intervals.Length <= 1)
            {
                return 0;
            }

            Array.Sort(intervals, Comparer<int[]>.Create((a, b) =>
            {
                var cmp = a[0] - b[0];
                return cmp == 0 ? b[1] - a[1] : cmp;
            }));
            int i = 1, j = 0, count = 0;
            while (i < intervals.Length)
            {
                int[] prev = intervals[j], cur = intervals[i];
                if (cur[0] >= prev[0] && cur[1] <= prev[1])
                {
                    count++;
                }
                else
                {
                    j = i;
                }

                i++;
            }

            return intervals.Length - count;
        }

        #endregion

        #region 986. 区间列表的交集

        //https://leetcode-cn.com/problems/interval-list-intersections/
        public int[][] IntervalIntersection(int[][] a, int[][] b)
        {
            var result = new List<int[]>();
            int i = 0, j = 0;
            while (i < a.Length && j < b.Length)
            {
                int[] arrA = a[i], arrB = b[j];
                if (arrA[0] <= arrB[1] && arrA[1] >= arrB[0])
                {
                    //交集<=集合数，只可能在最大起点和最小终点
                    result.Add(new[] {Math.Max(arrA[0], arrB[0]), Math.Min(arrA[1], arrB[1])});
                }

                //保留大的区间
                if (arrA[1] < arrB[1])
                {
                    i++;
                }
                else
                {
                    j++;
                }
            }

            return result.ToArray();
        }

        #endregion

        #region 752. 打开转盘锁

        //https://leetcode-cn.com/problems/open-the-lock/
        public int OpenLock(string[] deadends, string target)
        {
            var set = new HashSet<string>(deadends);
            if (set.Contains(target) || set.Contains("0000"))
            {
                return -1;
            }

            var queue = new Queue<string>();
            queue.Enqueue("0000");
            var step = 0;
            while (queue.Count > 0)
            {
                for (int i = 0, c = queue.Count; i < c; i++)
                {
                    var start = queue.Dequeue();
                    if (start == target)
                    {
                        return step;
                    }

                    for (var j = 0; j < start.Length; j++)
                    {
                        for (int s = -1; s < 2; s += 2)
                        {
                            var next = start.ToArray();
                            next[j] = (char) ((next[j] - '0' + s + 10) % 10 + '0');
                            var str = new string(next);
                            if (set.Add(str))
                            {
                                queue.Enqueue(str);
                            }
                        }
                    }
                }

                step++;
            }

            return -1;
        }

        #endregion


        #region 416. 分割等和子集

        //https://leetcode-cn.com/problems/partition-equal-subset-sum/
        bool CanPartition(int index, int[] nums, int prevSum, int sum)
        {
            if (index < 0 || prevSum < 0 || sum < 0)
            {
                return false;
            }

            if (prevSum == 0 || sum == 0)
            {
                return true;
            }

            return CanPartition(index - 1, nums, prevSum - nums[index], sum) ||
                   CanPartition(index - 1, nums, prevSum, sum - nums[index]);
        }

        public bool CanPartition(int[] nums)
        {
            if (nums.Length <= 1)
            {
                return false;
            }

            var sum = nums.Sum();
            if (sum % 2 == 1)
            {
                return false;
            }

            Array.Sort(nums);
            return CanPartition(nums.Length - 1, nums, sum / 2, sum / 2);
        }

        #endregion

        #region 357. 计算各个位数不同的数字个数

        //https://leetcode-cn.com/problems/count-numbers-with-unique-digits/
        public int CountNumbersWithUniqueDigits(int n)
        {
            if (n <= 1)
            {
                return n == 1 ? 10 : 1;
            }
            var count = 9;
            for (int i = 1; i < n; i++)
            {
                count *= 10 - i;
            }

            return count + CountNumbersWithUniqueDigits(n - 1);
        }

        #endregion
    }
}