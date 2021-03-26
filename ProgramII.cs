using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 32. 最长有效括号

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
                if ((i & 1) == 1)
                {
                    continue;
                }

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

        public int LongestValidParenthesesByStack(string s)
        {
            if (string.IsNullOrEmpty(s) || s.Length < 2)
            {
                return 0;
            }

            var max = 0;
            var stack = new Stack<int>();
            stack.Push(-1);
            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                if (ch == '(')
                {
                    stack.Push(i);
                }
                else
                {
                    stack.Pop();
                    if (stack.Count <= 0)
                    {
                        stack.Push(i);
                    }
                    else
                    {
                        max = Math.Max(max, i - stack.Peek());
                    }
                }
            }

            return max;
        }

        #endregion

        #region 347. 前 K 个高频元素

        //https://leetcode-cn.com/problems/top-k-frequent-elements/
        public int[] TopKFrequent(int[] nums, int k)
        {
            var len = 0;
            var dict = new Dictionary<int, int>();
            foreach (var num in nums)
            {
                if (!dict.TryGetValue(num, out var count))
                {
                    count = 1;
                }
                else
                {
                    count++;
                }

                dict[num] = count;
                len = Math.Max(len, count);
            }

            var buckets = new IList<int>[len + 1];
            foreach (var kv in dict)
            {
                if (buckets[kv.Value] == null)
                {
                    buckets[kv.Value] = new List<int>();
                }

                buckets[kv.Value].Add(kv.Key);
            }

            var result = new int[k];
            for (int i = buckets.Length - 1; i >= 0 && k > 0; i--)
            {
                if (buckets[i] == null)
                {
                    continue;
                }

                for (int j = 0; j < buckets[i].Count && k > 0; j++, k--)
                {
                    result[k - 1] = buckets[i][j];
                }
            }

            return result;
        }

        #endregion

        #region 378. 有序矩阵中第K小的元素

        //https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/
        //二分法
        public int KthSmallest(int[][] matrix, int k)
        {
            int CountNum(int target)
            {
                int i = matrix.Length - 1, j = 0, cnt = 0;
                while (i >= 0 && j < matrix.Length)
                {
                    if (matrix[i][j] <= target)
                    {
                        cnt = cnt + i + 1;
                        j++;
                    }
                    else
                    {
                        i--;
                    }
                }

                return cnt;
            }

            var n = matrix.Length;
            int left = matrix[0][0], right = matrix[n - 1][n - 1];
            while (left < right)
            {
                var mid = (left + right) / 2;
                var count = CountNum(mid);
                if (count < k)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid;
                }
            }

            return left;
        }

        //优先队列
        // public int kthSmallest(int[][] matrix, int k) {
        //     PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
        //         @Override
        //         public int compare(Integer o1, Integer o2) {
        //         return o2 - o1;
        //     }
        //     });
        //     for (int i = 0; i < matrix.length; i++) {
        //         for (int j = 0; j < matrix[i].length; j++) {
        //             if (queue.size() < k) {
        //                 queue.offer(matrix[i][j]);
        //             } else {
        //                 if (queue.peek() <= matrix[i][j]) {
        //                     break;
        //                 }
        //                 queue.offer(matrix[i][j]);
        //                 queue.poll();
        //             }
        //         }
        //     }
        //     return queue.peek();
        // }

        #endregion

        #region 395. 至少有K个重复字符的最长子串

        //https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/
        public int LongestSubstring(string s, int k)
        {
            var res = 0;
            var bucket = new int[26];
            var len = s.Length;
            for (int l = len; l > 0; l--)
            {
                for (int i = 0; i < s.Length - l + 1; i++)
                {
                    var str = s.Substring(i, l);
                    foreach (var ch in str)
                    {
                        bucket[ch - 'a']++;
                    }

                    if (bucket.Where(n => n > 0).All(n => n >= k))
                    {
                        return l;
                    }

                    Array.Clear(bucket, 0, bucket.Length);
                }
            }

            return res;
        }

        int LongestSubstringPart(string s, int k, int left, int right)
        {
            var len = right - left + 1;
            if (len < k)
            {
                return 0;
            }

            var bucket = new int[26];
            for (int i = left; i <= right; i++)
            {
                bucket[s[i] - 'a']++;
            }

            while (len >= k && bucket[s[left] - 'a'] < k)
            {
                left++;
                len--;
            }

            while (len >= k && bucket[s[right] - 'a'] < k)
            {
                right--;
                len--;
            }

            if (len < k)
            {
                return 0;
            }

            for (int i = left; i <= right; i++)
            {
                if (bucket[s[i] - 'a'] < k)
                {
                    return Math.Max(LongestSubstringPart(s, k, left, i - 1), LongestSubstringPart(s, k, i + 1, right));
                }
            }

            return len;
        }

        public int LongestSubstringByPart(string s, int k)
        {
            return k < 2 ? s.Length : LongestSubstringPart(s, k, 0, s.Length - 1);
        }

        #endregion

        #region 1464. 数组中两元素的最大乘积

        //https://leetcode-cn.com/problems/maximum-product-of-two-elements-in-an-array/
        public int MaxProductI(int[] nums)
        {
            Array.Sort(nums);
            return (nums[nums.Length - 1] - 1) * (nums[nums.Length - 2] - 1);
        }

        #endregion

        #region 1480. 一维数组的动态和

        //https://leetcode-cn.com/problems/running-sum-of-1d-array/
        public int[] RunningSum(int[] nums)
        {
            if (nums.Length <= 1)
            {
                return nums;
            }

            var res = new int[nums.Length];
            res[0] = nums[0];
            for (int i = 1; i < nums.Length; i++)
            {
                res[i] = res[i - 1] + nums[i];
            }

            return res;
        }

        #endregion

        #region 300. 最长上升子序列

        //https://leetcode-cn.com/problems/longest-increasing-subsequence/
        public int LengthOfLIS(int[] nums)
        {
            if (nums.Length <= 0)
            {
                return 0;
            }

            //求出数组【0-i】之间小于的数
            //当有数大于num[i]时，此时dp[j]=dp[i]+1
            var dp = new int[nums.Length];
            dp[0] = 1;
            var res = 0;
            for (int i = 1; i < nums.Length; i++)
            {
                var max = 0;
                for (int j = 0; j < i; j++)
                {
                    if (nums[i] > nums[j])
                    {
                        max = Math.Max(max, dp[j]);
                    }
                }

                dp[i] = max + 1;
                res = Math.Max(dp[i], res);
            }

            return res;
        }

        #endregion

        #region 60. 第k个排列

        //https://leetcode-cn.com/problems/permutation-sequence/

        public string GetPermutation(int n, int k)
        {
            var nums = new int[n];
            for (int i = 0; i < nums.Length; i++)
            {
                nums[i] = i + 1;
            }

            for (int i = 1; i < k; i++)
            {
                NextPermutation(nums);
            }

            return string.Join(string.Empty, nums);
        }

        public string GetPermutationByDfs(int n, int k)
        {
            var nums = new int[n];
            var visited = new bool[n + 1];
            nums[0] = 1;
            for (int i = 1; i < n; i++)
            {
                nums[i] = nums[i - 1] * i;
            }

            var res = new StringBuilder();

            void Dfs(int index)
            {
                if (index >= n)
                {
                    return;
                }

                var size = nums[n - index - 1];
                for (int i = 1; i <= n; i++)
                {
                    if (visited[i])
                    {
                        continue;
                    }

                    if (k > size)
                    {
                        k -= size;
                        continue;
                    }

                    visited[i] = true;
                    res.Append(i);
                    Dfs(index + 1);
                }
            }

            Dfs(0);
            return res.ToString();
        }

        #endregion

        #region 207. 课程表

        //https://leetcode-cn.com/problems/course-schedule/
        public bool CanFinish(int numCourses, int[][] prerequisites)
        {
            if (prerequisites.Length <= 0)
            {
                return true;
            }

            var depend = new Dictionary<int, ISet<int>>();
            foreach (var num in prerequisites)
            {
                if (!depend.TryGetValue(num[0], out var set))
                {
                    set = new HashSet<int>();
                    depend[num[0]] = set;
                }

                set.Add(num[1]);
            }

            var visited = new HashSet<int>();
            var queue = new Queue<Tuple<int, ISet<int>>>();
            foreach (var kv in depend)
            {
                var key = kv.Key;
                if (visited.Contains(key))
                {
                    continue;
                }

                queue.Enqueue(new Tuple<int, ISet<int>>(key, new HashSet<int>()));
                while (queue.Count > 0)
                {
                    var item = queue.Dequeue();
                    key = item.Item1;
                    var path = item.Item2;
                    if (!path.Add(key))
                    {
                        return false;
                    }

                    if (depend.TryGetValue(key, out var next))
                    {
                        foreach (var k in next)
                        {
                            if (visited.Contains(key))
                            {
                                continue;
                            }

                            queue.Enqueue(new Tuple<int, ISet<int>>(k, new HashSet<int>(path)));
                        }
                    }
                    else
                    {
                        foreach (var k in path)
                        {
                            visited.Add(k);
                        }
                    }
                }
            }

            return visited.Count <= numCourses;
        }

        public bool CanFinishDfs(int numCourses, int[][] prerequisites)
        {
            if (prerequisites.Length <= 0)
            {
                return true;
            }

            var dict = new Dictionary<int, IList<int>>();
            ISet<int> paths = new HashSet<int>(), resultSet = new HashSet<int>();

            bool Dfs(int key)
            {
                if (resultSet.Contains(key))
                {
                    return true;
                }

                if (!paths.Add(key))
                {
                    return false;
                }

                if (dict.TryGetValue(key, out var next))
                {
                    if (next.Any(k => !Dfs(k)))
                    {
                        return false;
                    }
                }

                paths.Remove(key);
                resultSet.Add(key);
                return true;
            }

            foreach (var num in prerequisites)
            {
                if (!dict.TryGetValue(num[0], out var set))
                {
                    set = new List<int>();
                    dict[num[0]] = set;
                }

                set.Add(num[1]);
            }

            return dict.All(kv => Dfs(kv.Key));
        }

        #endregion

        #region 494. 目标和

        //https://leetcode-cn.com/problems/target-sum/
        void FindTargetSumWays(int index, int[] nums, int s, int sum, ref int res)
        {
            if (index >= nums.Length)
            {
                if (s == sum)
                {
                    res++;
                }

                return;
            }

            FindTargetSumWays(index + 1, nums, s, sum + nums[index], ref res);
            FindTargetSumWays(index + 1, nums, s, sum - nums[index], ref res);
        }

        public int FindTargetSumWays(int[] nums, int s)
        {
            var res = 0;
            FindTargetSumWays(0, nums, s, 0, ref res);
            return res;
        }

        #endregion

        #region 92. 反转链表 II

        //https://leetcode-cn.com/problems/reverse-linked-list-ii/
        public ListNode ReverseBetween(ListNode head, int m, int n)
        {
            ListNode Reverse(ListNode listNode)
            {
                ListNode prevNode = null;
                while (listNode != null)
                {
                    var next = listNode.next;
                    listNode.next = prevNode;
                    prevNode = listNode;
                    listNode = next;
                }

                return prevNode;
            }

            if (head == null)
            {
                return null;
            }

            ListNode prev = null, node = head;
            while (m > 1)
            {
                prev = node;
                node = node.next;
                m--;
                n--;
            }

            while (n > 1)
            {
                node = node.next;
                n--;
            }

            ListNode newNode, nextHead = node.next;
            node.next = null;
            if (prev == null)
            {
                newNode = Reverse(head);
                head.next = nextHead;
                return newNode;
            }

            var tail = prev.next;
            newNode = Reverse(tail);
            prev.next = newNode;
            tail.next = nextHead;
            return head;
        }

        #endregion

        #region 24. 两两交换链表中的节点

        //https://leetcode-cn.com/problems/swap-nodes-in-pairs/
        //1.递归版
        public ListNode SwapPairs(ListNode head)
        {
            if (head == null || head.next == null)
            {
                return head;
            }

            ListNode first = head, second = head.next;
            first.next = SwapPairs(second.next);
            second.next = first;
            return second;
        }

        //2.迭代版
        public ListNode SwapPairsI(ListNode head)
        {
            if (head == null || head.next == null)
            {
                return head;
            }

            ListNode swapNode = head.next, prev = head;
            head = swapNode;
            while (true)
            {
                var next = swapNode.next;
                swapNode.next = prev;
                if (next == null || next.next == null)
                {
                    prev.next = next;
                    break;
                }

                swapNode = next.next;
                prev.next = swapNode;
                prev = next;
            }

            return head;
        }

        #endregion

        #region 37. 解数独

        //https://leetcode-cn.com/problems/sudoku-solver/

        public void SolveSudoku(char[][] board)
        {
            bool[,] rows = new bool[9, 9], cols = new bool[9, 9];
            var matrix = new bool[3, 3][];
            for (var i = 0; i < board.Length; i++)
            {
                for (var j = 0; j < board[i].Length; j++)
                {
                    if (board[i][j] == '.')
                    {
                        continue;
                    }

                    var num = board[i][j] - '1';
                    rows[i, num] = true;
                    cols[j, num] = true;
                    int x = i / 3, y = j / 3;
                    if (matrix[x, y] == null)
                    {
                        matrix[x, y] = new bool[9];
                    }

                    matrix[x, y][num] = true;
                }
            }

            bool Set(int x, int y)
            {
                if (y >= 9)
                {
                    x++;
                    y = 0;
                }

                if (x >= 9)
                {
                    return true;
                }

                if (board[x][y] != '.')
                {
                    return Set(x, y + 1);
                }

                var flag = matrix[x / 3, y / 3];
                for (int i = 0; i < 9; i++)
                {
                    if (rows[x, i] || cols[y, i] || flag[i])
                    {
                        continue;
                    }

                    board[x][y] = (char) (i + '1');
                    rows[x, i] = true;
                    cols[y, i] = true;
                    flag[i] = true;
                    if (Set(x, y + 1))
                    {
                        return true;
                    }

                    board[x][y] = '.';
                    rows[x, i] = false;
                    cols[y, i] = false;
                    flag[i] = false;
                }

                return false;
            }

            Set(0, 0);
        }

        #endregion

        #region 209. 长度最小的子数组

        //https://leetcode-cn.com/problems/minimum-size-subarray-sum/
        public int MinSubArrayLen(int s, int[] nums)
        {
            var len = int.MaxValue;
            var sum = 0;
            for (int i = 0, j = 0; i < nums.Length; i++)
            {
                sum += nums[i];
                while (sum >= s)
                {
                    len = Math.Min(len, i - j + 1);
                    sum -= nums[j];
                    if (j >= i)
                    {
                        return 1;
                    }

                    j++;
                }
            }

            return len == int.MaxValue ? 0 : len;
        }

        #endregion

        #region 95. 不同的二叉搜索树 II

        //https://leetcode-cn.com/problems/unique-binary-search-trees-ii/
        public IList<TreeNode> GenerateTrees(int n)
        {
            IList<TreeNode> Generate(int start, int end)
            {
                if (start > end)
                {
                    return new TreeNode[] {null};
                }

                var items = new List<TreeNode>();
                for (int i = start; i <= end; i++)
                {
                    var lefts = Generate(start, i - 1);
                    var rights = Generate(i + 1, end);
                    for (int l = 0; l < lefts.Count; l++)
                    {
                        for (int r = 0; r < rights.Count; r++)
                        {
                            var node = new TreeNode(i);
                            items.Add(node);
                            node.left = lefts[l];
                            node.right = rights[r];
                        }
                    }
                }

                return items;
            }

            return Generate(1, n);
        }

        #endregion

        #region 1470. 重新排列数组

        //https://leetcode-cn.com/problems/shuffle-the-array/
        public int[] Shuffle(int[] nums, int n)
        {
            var res = new int[nums.Length];
            int i1 = 0, i2 = n;
            for (int i = 0; i < res.Length; i += 2)
            {
                res[i] = nums[i1++];
            }

            for (int i = 1; i < res.Length; i += 2)
            {
                res[i] = nums[i2++];
            }

            return res;
        }

        #endregion

        #region 200. 岛屿数量

        //https://leetcode-cn.com/problems/number-of-islands/
        public int NumIslands(char[][] grid)
        {
            if (grid.Length <= 0)
            {
                return 0;
            }

            var flags = new bool[grid.Length, grid[0].Length];
            var res = 0;

            void CheckLand(int x, int y)
            {
                if (x < 0 || x >= grid.Length || y < 0 || y >= grid[0].Length || flags[x, y] || grid[x][y] != '1')
                {
                    return;
                }


                flags[x, y] = true;
                CheckLand(x + 1, y);
                CheckLand(x - 1, y);
                CheckLand(x, y - 1);
                CheckLand(x, y + 1);
            }

            for (int i = 0; i < grid.Length; i++)
            {
                for (int j = 0; j < grid[i].Length; j++)
                {
                    if (flags[i, j] || grid[i][j] != '1')
                    {
                        continue;
                    }

                    CheckLand(i, j);
                    res++;
                }
            }

            return res;
        }

        #endregion

        #region 44. 通配符匹配

        //https://leetcode-cn.com/problems/wildcard-matching/
        public bool IsMatchI(string s, string p)
        {
            if (string.IsNullOrEmpty(p))
            {
                return string.IsNullOrEmpty(s);
            }

            bool?[,] flags = new bool?[s.Length, p.Length];

            bool Match(int si, int pi)
            {
                if (si >= s.Length)
                {
                    while (pi < p.Length)
                    {
                        if (p[pi] != '*')
                        {
                            return false;
                        }

                        pi++;
                    }

                    return true;
                }

                if (pi >= p.Length)
                {
                    return false;
                }

                if (flags[si, pi].HasValue)
                {
                    return flags[si, pi].Value;
                }

                if (s[si] == p[pi] || p[pi] == '?')
                {
                    flags[si, pi] = Match(si + 1, pi + 1);
                    return flags[si, pi].Value;
                }

                if (p[pi] == '*')
                {
                    //01234
                    for (int l = si; l <= s.Length; l++)
                    {
                        if (Match(l, pi + 1))
                        {
                            flags[si, pi] = true;
                            return true;
                        }
                    }
                }

                flags[si, pi] = false;
                return false;
            }

            return Match(0, 0);
        }

        #endregion

        #region 63. 不同路径 II

        //https://leetcode-cn.com/problems/unique-paths-ii/
        public int UniquePathsWithObstacles(int[][] obstacleGrid)
        {
            if (obstacleGrid == null || obstacleGrid.Length <= 0)
            {
                return 0;
            }

            int m = obstacleGrid.Length, n = obstacleGrid[0].Length;
            var dp = new int[m, n];
            dp[0, 0] = obstacleGrid[0][0] == 1 ? 0 : 1;
            for (int i = 1; i < m; i++)
            {
                dp[i, 0] = obstacleGrid[i][0] == 1 || dp[i - 1, 0] == 0 ? 0 : 1;
            }

            for (int i = 1; i < n; i++)
            {
                dp[0, i] = obstacleGrid[0][i] == 1 || dp[0, i - 1] == 0 ? 0 : 1;
            }

            for (int i = 1; i < m; i++)
            {
                for (int j = 1; j < n; j++)
                {
                    if (obstacleGrid[i][j] == 1)
                    {
                        continue;
                    }

                    dp[i, j] = dp[i - 1, j] + dp[i, j - 1];
                }
            }

            return dp[m - 1, n - 1];
        }

        #endregion

        #region 90. 子集 II

        //https://leetcode-cn.com/problems/subsets-ii/
        void SubsetsWithDup(int index, int[] nums, IList<IList<int>> result, IList<int> subSet)
        {
            result.Add(subSet.ToArray());
            if (index >= nums.Length)
            {
                return;
            }

            for (int i = index; i < nums.Length; i++)
            {
                if (i > index && nums[i] == nums[i - 1])
                {
                    continue;
                }

                subSet.Add(nums[i]);
                SubsetsWithDup(i + 1, nums, result, subSet);
                subSet.RemoveAt(subSet.Count - 1);
            }
        }

        public IList<IList<int>> SubsetsWithDup(int[] nums)
        {
            if (nums == null || nums.Length <= 0)
            {
                return new IList<int>[0];
            }

            var result = new List<IList<int>>();
            Array.Sort(nums);
            SubsetsWithDup(0, nums, result, new List<int>());
            return result;
        }

        #endregion

        #region 71. 简化路径

        //https://leetcode-cn.com/problems/simplify-path/
        public string SimplifyPath(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                return path;
            }

            var stack = new Stack<string>();
            string part;
            for (int i = 0, j = 0; i < path.Length; i++)
            {
                if (path[i] == '/')
                {
                    if (i == j)
                    {
                        j = i + 1;
                        continue;
                    }

                    part = path.Substring(j, i - j);
                    j = i + 1;
                }
                else if (i == path.Length - 1)
                {
                    part = path.Substring(j, i - j + 1);
                }
                else
                {
                    continue;
                }

                if (part == "..")
                {
                    stack.TryPop(out _);
                }
                else if (part != ".")
                {
                    stack.Push(part);
                }
            }

            var res = new StringBuilder();
            while (stack.TryPop(out var p))
            {
                res.Insert(0, p);
                res.Insert(0, '/');
            }

            return res.Length > 0 ? res.ToString() : "/";
        }

        #endregion

        #region 77. 组合

        //https://leetcode-cn.com/problems/combinations/
        void Combine(int index, int n, int k, IList<IList<int>> result, IList<int> combine)
        {
            if (index >= n || combine.Count == k)
            {
                if (combine.Count == k)
                {
                    result.Add(combine.ToArray());
                }

                return;
            }

            for (int i = index; i <= n; i++)
            {
                combine.Add(i);
                Combine(i + 1, n, k, result, combine);
                combine.RemoveAt(combine.Count - 1);
            }
        }

        public IList<IList<int>> Combine(int n, int k)
        {
            var result = new List<IList<int>>();
            Combine(1, n, k, result, new List<int>());
            return result;
        }

        #endregion

        #region 72. 编辑距离

        //https://leetcode-cn.com/problems/edit-distance/
        void MinDistance(int i1, int i2, string word1, string word2, int step, ref int res)
        {
            if (i1 >= word1.Length && i2 >= word2.Length)
            {
                res = Math.Min(res, step);
                return;
            }

            if (step >= res)
            {
                return;
            }

            if (i1 >= word1.Length)
            {
                MinDistance(i1, i2 + 1, word1, word2, step + 1, ref res);
                return;
            }

            if (i2 >= word2.Length)
            {
                //删除一个字符
                MinDistance(i1 + 1, i2, word1, word2, step + 1, ref res);
                return;
            }

            if (word1[i1] == word2[i2])
            {
                MinDistance(i1 + 1, i2 + 1, word1, word2, step, ref res);
            }
            else
            {
                //插入一个字符
                MinDistance(i1, i2 + 1, word1, word2, step + 1, ref res);
                //删除一个字符
                MinDistance(i1 + 1, i2, word1, word2, step + 1, ref res);
                //替换一个字符
                MinDistance(i1 + 1, i2 + 1, word1, word2, step + 1, ref res);
            }
        }

        int MinDistanceCache(int i1, int i2, string word1, string word2, int[,] cache)
        {
            if (i1 >= word1.Length && i2 >= word2.Length)
            {
                return 0;
            }

            if (i1 >= word1.Length)
            {
                return word2.Length - i2;
            }

            if (i2 >= word2.Length)
            {
                //删除一个字符
                return word1.Length - i1;
            }

            if (cache[i1, i2] != 0)
            {
                return cache[i1, i2];
            }

            if (word1[i1] == word2[i2])
            {
                cache[i1, i2] = MinDistanceCache(i1 + 1, i2 + 1, word1, word2, cache);
            }
            else
            {
                //插入一个字符
                var s1 = MinDistanceCache(i1, i2 + 1, word1, word2, cache);
                //删除一个字符
                var s2 = MinDistanceCache(i1 + 1, i2, word1, word2, cache);
                //替换一个字符
                var s3 = MinDistanceCache(i1 + 1, i2 + 1, word1, word2, cache);
                cache[i1, i2] = Math.Min(Math.Min(s1, s2), s3) + 1;
            }

            return cache[i1, i2];
        }

        public int MinDistance(string word1, string word2)
        {
            if (string.IsNullOrEmpty(word2))
            {
                return word1.Length;
            }

            return MinDistanceCache(0, 0, word1, word2, new int[word1.Length, word2.Length]);
        }

        #endregion

        #region 82. 删除排序链表中的重复元素 II

        //https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/
        public ListNode DeleteDuplicatesII(ListNode head)
        {
            if (head == null || head.next == null)
            {
                return head;
            }

            var dict = new Dictionary<int, int>();
            var node = head;
            while (node != null)
            {
                if (dict.ContainsKey(node.val))
                {
                    dict[node.val]++;
                }
                else
                {
                    dict[node.val] = 1;
                }

                node = node.next;
            }

            var newHead = new ListNode(0);
            node = newHead;
            while (head != null)
            {
                if (dict[head.val] == 1)
                {
                    node.next = head;
                    node = node.next;
                }

                head = head.next;
            }

            node.next = null;
            return newHead.next;
        }

        public ListNode DeleteDuplicatesO1(ListNode head)
        {
            if (head?.next == null)
            {
                return head;
            }

            var newHead = new ListNode(-1);
            var count = 0;
            ListNode prev = head, node = newHead;
            while (head != null)
            {
                if (prev.val == head.val)
                {
                    count++;
                }
                else
                {
                    if (count == 1)
                    {
                        node.next = prev;
                        node = node.next;
                    }

                    count = 1;
                }

                prev = head;
                head = head.next;
            }

            if (count == 1)
            {
                node.next = prev;
                node = node.next;
            }

            node.next = null;
            return newHead.next;
        }

        #endregion

        #region 51. N皇后/面试题 08.12. 八皇后

        //https://leetcode-cn.com/problems/eight-queens-lcci/
        //https://leetcode-cn.com/problems/n-queens/

        //棋子从上往下放，只需要检查上层
        bool CheckQueen(int x, int y, char[][] flags, int n)
        {
            for (int i = 0; i <= x; i++)
            {
                if (flags[i][y] == 'Q')
                {
                    return false;
                }
            }

            int x1 = x, y1 = y;
            while (x >= 0 && y >= 0)
            {
                if (flags[x][y] == 'Q')
                {
                    return false;
                }

                x--;
                y--;
            }

            while (x1 >= 0 && y1 < n)
            {
                if (flags[x1][y1] == 'Q')
                {
                    return false;
                }

                x1--;
                y1++;
            }

            return true;
        }

        bool SolveNQueens(int row, char[][] flags, int n, IList<IList<string>> result)
        {
            if (row >= n)
            {
                var items = flags.Select(chars => new string(chars)).ToArray();
                result.Add(items);
                return true;
            }

            var flag = false;
            for (int i = 0; i < n; i++)
            {
                if (!CheckQueen(row, i, flags, n))
                {
                    continue;
                }

                flags[row][i] = 'Q';
                flag = SolveNQueens(row + 1, flags, n, result) || flag;
                flags[row][i] = '.';
            }

            return flag;
        }

        public IList<IList<string>> SolveNQueens(int n)
        {
            var result = new List<IList<string>>();
            var flags = new char[n][];
            for (int i = 0; i < n; i++)
            {
                flags[i] = new char[n];
                Array.Fill(flags[i], '.');
            }

            SolveNQueens(0, flags, n, result);
            return result;
        }

        //重新实现
        public IList<IList<string>> SolveNQueensRedo(int n)
        {
            var result = new List<IList<string>>();
            var mask = new List<string>();
            var line = new char[n];
            Array.Fill(line, '.');
            var cols = new bool[n];

            bool CanSet(int y)
            {
                if (cols[y])
                {
                    return false;
                }

                for (int i = mask.Count - 1, j = y - 1; i >= 0 && j >= 0; i--, j--)
                {
                    if (mask[i][j] == 'Q')
                    {
                        return false;
                    }
                }

                for (int i = mask.Count - 1, j = y + 1; i >= 0 && j < n; i--, j++)
                {
                    if (mask[i][j] == 'Q')
                    {
                        return false;
                    }
                }

                return true;
            }

            void Dfs(int num)
            {
                if (num <= 0)
                {
                    result.Add(mask.ToArray());
                    return;
                }

                for (int i = 0; i < n; i++)
                {
                    if (!CanSet(i))
                    {
                        continue;
                    }

                    cols[i] = true;
                    line[i] = 'Q';
                    mask.Add(new string(line));
                    line[i] = '.';
                    Dfs(num - 1);
                    mask.RemoveAt(mask.Count - 1);
                    cols[i] = false;
                }
            }

            Dfs(n);
            return result;
        }

        #endregion

        #region 221. 最大正方形

        //https://leetcode-cn.com/problems/maximal-square/
        public int MaximalSquare(char[][] matrix)
        {
            if (matrix == null || matrix.Length <= 0)
            {
                return 0;
            }

            var dp = new int[matrix.Length, matrix[0].Length];
            var res = 0;
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    if (matrix[i][j] == '0')
                    {
                        continue;
                    }

                    if (i == 0 || j == 0)
                    {
                        dp[i, j] = 1;
                    }
                    else
                    {
                        dp[i, j] = Math.Min(Math.Min(dp[i - 1, j], dp[i, j - 1]), dp[i - 1, j - 1]) + 1;
                    }

                    res = Math.Max(res, dp[i, j] * dp[i, j]);
                }
            }

            return res;
        }

        #endregion

        #region 279. 完全平方数

        //https://leetcode-cn.com/problems/perfect-squares/

        private Dictionary<int, int> squaresCache = new Dictionary<int, int>();

        public int NumSquares(int n)
        {
            if (n == 1)
            {
                return 1;
            }

            if (squaresCache.TryGetValue(n, out var res))
            {
                return res;
            }

            var num = (int) Math.Floor(Math.Sqrt(n));
            if (num * num == n)
            {
                res = 1;
            }
            else
            {
                res = int.MaxValue;
                for (int i = num; i > 0; i--)
                {
                    res = Math.Min(NumSquares(n - i * i), res);
                }

                res++;
            }

            squaresCache[n] = res;
            return res;
        }

        #endregion

        #region 718. 最长重复子数组

        //https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/
        public int FindLength(int[] A, int[] B)
        {
            var dict = new Dictionary<int, IList<int>>();
            for (int i = 0; i < A.Length; i++)
            {
                if (!dict.ContainsKey(A[i]))
                {
                    dict[A[i]] = new List<int>();
                }

                dict[A[i]].Add(i);
            }

            var res = 0;
            for (int i = 0; i < B.Length; i++)
            {
                var num = B[i];
                if (!dict.TryGetValue(num, out var indexs))
                {
                    continue;
                }

                if (B.Length - i <= res)
                {
                    break;
                }

                foreach (var index in indexs)
                {
                    if (A.Length - index <= res)
                    {
                        continue;
                    }

                    var len = 1;
                    for (int j = index + 1, k = i + 1; j < A.Length && k < B.Length; j++, k++)
                    {
                        if (A[j] != B[k])
                        {
                            break;
                        }

                        len++;
                    }

                    res = Math.Max(res, len);
                }
            }

            return res;
        }

        public int FindLengthDP(int[] A, int[] B)
        {
            var dp = new int[A.Length + 1, B.Length + 1];
            var res = 0;
            for (int i = A.Length - 1; i >= 0; i--)
            {
                for (int j = B.Length - 1; j >= 0; j--)
                {
                    dp[i, j] = A[i] == B[j] ? dp[i + 1, j + 1] + 1 : 0;
                    res = Math.Max(res, dp[i, j]);
                }
            }

            return res;
        }

        #endregion

        #region 322. 零钱兑换

        //https://leetcode-cn.com/problems/coin-change/
        bool CoinChange(int[] coins, int amount, int step, ref int res)
        {
            if (amount == 0 || step >= res)
            {
                res = Math.Min(step, res);
                return true;
            }

            var flag = false;
            for (int i = coins.Length - 1; i >= 0; i--)
            {
                if (amount < coins[i])
                {
                    continue;
                }

                flag = CoinChange(coins, amount - coins[i], step + 1, ref res) || flag;
            }

            return flag;
        }

        public int CoinChangeDP(int[] coins, int amount, Dictionary<int, int> cache)
        {
            if (amount == 0)
            {
                return 0;
            }

            if (amount < 0)
            {
                return -1;
            }

            if (cache.TryGetValue(amount, out var res))
            {
                return res;
            }

            res = -1;
            for (int i = coins.Length - 1; i >= 0; i--)
            {
                if (amount < coins[i])
                {
                    continue;
                }

                var next = CoinChangeDP(coins, amount - coins[i], cache);
                if (next >= 0)
                {
                    res = res == -1 ? next + 1 : Math.Min(next + 1, res);
                }
            }

            cache[amount] = res;
            return res;
        }

        public int CoinChange(int[] coins, int amount)
        {
            if (coins.Length <= 0 || amount <= 0)
            {
                return 0;
            }

            var res = CoinChangeDP(coins, amount, new Dictionary<int, int>());
            return res;
        }

        #endregion

        #region 338. 比特位计数

        //https://leetcode-cn.com/problems/counting-bits/
        public int[] CountBits(int num)
        {
            var result = new int[num + 1];
            for (int i = 1; i <= num; i++)
            {
                if (((i - 1) & i) == 0)
                {
                    result[i] = 1;
                }
                else
                {
                    var n = i;
                    while (n != 0)
                    {
                        result[i]++;
                        n = (n - 1) & n;
                    }
                }
            }

            return result;
        }

        #endregion

        #region 438. 找到字符串中所有字母异位词

        //https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/
        public IList<int> FindAnagrams(string s, string p)
        {
            var dict = new Dictionary<char, int>();
            foreach (var ch in p)
            {
                if (dict.ContainsKey(ch))
                {
                    dict[ch]++;
                }
                else
                {
                    dict[ch] = 1;
                }
            }

            var result = new List<int>();
            var counter = new Dictionary<char, int>();
            var len = 0;
            for (int i = 0, j = 0; i < s.Length; i++)
            {
                if (counter.ContainsKey(s[i]))
                {
                    counter[s[i]]++;
                }
                else
                {
                    counter[s[i]] = 1;
                }

                len++;
                if (len < p.Length)
                {
                    continue;
                }

                if (counter.Count == dict.Count &&
                    counter.All(kv => dict.ContainsKey(kv.Key) && dict[kv.Key] == kv.Value))
                {
                    result.Add(j);
                }

                counter[s[j]]--;
                if (counter[s[j]] == 0)
                {
                    counter.Remove(s[j]);
                }

                j++;
                len--;
            }

            return result;
        }

        #endregion

        #region 337. 打家劫舍 III

        //https://leetcode-cn.com/problems/house-robber-iii/
        private int Rob(TreeNode root, IDictionary<TreeNode, int> cache)
        {
            if (root == null)
            {
                return 0;
            }

            if (cache.TryGetValue(root, out var res))
            {
                return res;
            }

            res = root.val;
            if (root.left != null)
            {
                res += Rob(root.left.left, cache) + Rob(root.left.right, cache);
            }

            if (root.right != null)
            {
                res += Rob(root.right.right, cache) + Rob(root.right.left, cache);
            }

            res = Math.Max(res, Rob(root.left, cache) + Rob(root.right, cache));
            cache[root] = res;
            return res;
        }

        public int Rob(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            return Rob(root, new Dictionary<TreeNode, int>());
        }

        #endregion

        #region 213. 打家劫舍 II

        //https://leetcode-cn.com/problems/house-robber-ii/
        public int RobII(int[] nums)
        {
            int Dfs(int start, int end)
            {
                int pre1 = 0, pre2 = 0, current = 0;
                for (int i = start; i < end; i++)
                {
                    current = Math.Max(pre1, pre2 + nums[i]);
                    pre2 = pre1;
                    pre1 = current;
                }

                return current;
            }

            if (nums == null || nums.Length == 0)
            {
                return 0;
            }

            return nums.Length == 1 ? nums[0] : Math.Max(Dfs(0, nums.Length - 1), Dfs(1, nums.Length));
        }

        #endregion

        #region 85. 最大矩形

        //https://leetcode-cn.com/problems/maximal-rectangle/
        public int MaximalRectangle(char[][] matrix)
        {
            if (matrix.Length == 0 || matrix[0].Length == 0)
            {
                return 0;
            }

            var w = new int[matrix.Length, matrix[0].Length];
            var res = 0;
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    if (matrix[i][j] == '0')
                    {
                        continue;
                    }

                    if (j == 0)
                    {
                        w[i, j] = 1;
                    }
                    else
                    {
                        w[i, j] = w[i, j - 1] + 1;
                    }

                    var width = w[i, j];
                    for (int h = i; h >= 0; h--)
                    {
                        width = Math.Min(width, w[h, j]);
                        res = Math.Max(res, width * (i - h + 1));
                    }
                }
            }

            return res;
        }

        #endregion

        #region 301. 删除无效的括号

        //https://leetcode-cn.com/problems/remove-invalid-parentheses/

        #region 自实现

        bool IsValid(IList<char> chars)
        {
            var left = 0;
            foreach (var ch in chars)
            {
                if (ch == '(')
                {
                    left++;
                }
                else if (ch == ')')
                {
                    left--;
                }

                if (left < 0)
                {
                    return false;
                }
            }

            return left == 0;
        }


        void RemoveInvalidParentheses(char[] chars, int l, int r, ISet<string> result, ISet<string> visited,
            ref int max)
        {
            if (chars.Length == 0)
            {
                return;
            }

            if (!visited.Add(new string(chars)))
            {
                return;
            }

            if (l == r)
            {
                if (chars.Length >= max && IsValid(chars))
                {
                    if (chars.Length > max)
                    {
                        max = chars.Length;
                        result.Clear();
                    }

                    result.Add(new string(chars));
                    return;
                }

                if (chars.Length <= max)
                {
                    return;
                }

                for (int li = 0; li < chars.Length; li++)
                {
                    for (int ri = 0; ri < chars.Length; ri++)
                    {
                        if (chars[li] == '(' && chars[ri] == ')')
                        {
                            RemoveInvalidParentheses(chars.Where((c, i) => i != ri && i != li).ToArray(), l - 1, r - 1,
                                result, visited, ref max);
                        }
                    }
                }
            }
            else if (l > r)
            {
                //删除（
                for (int i = 0; i < chars.Length; i++)
                {
                    if (chars[i] != '(' || (i > 0 && chars[i - 1] == '('))
                    {
                        continue;
                    }

                    RemoveInvalidParentheses(chars.Where((c, li) => li != i).ToArray(), l - 1, r, result, visited,
                        ref max);
                }
            }
            else
            {
                //删除）
                for (int i = 0; i < chars.Length; i++)
                {
                    if (chars[i] != ')' || (i > 0 && chars[i - 1] == ')'))
                    {
                        continue;
                    }

                    RemoveInvalidParentheses(chars.Where((c, ri) => ri != i).ToArray(), l, r - 1, result, visited,
                        ref max);
                }
            }
        }

        public IList<string> RemoveInvalidParentheses(string s)
        {
            int l = 0, r = 0;
            foreach (var c in s)
            {
                if (c == '(')
                {
                    l++;
                }
                else if (c == ')')
                {
                    r++;
                }
            }

            var max = 0;
            var result = new HashSet<string>();
            RemoveInvalidParentheses(s.ToCharArray(), l, r, result, new HashSet<string>(), ref max);
            return result.Count <= 0 ? new[] {string.Empty} : result.ToArray();
        }

        #endregion

        #region 回溯实现

        void RemoveInvalidParenthesesDP(string s, int index, int left, int right, ISet<string> result,
            StringBuilder str, ref int max)
        {
            if (index >= s.Length)
            {
                if (left == right && str.Length >= max)
                {
                    if (str.Length > max)
                    {
                        result.Clear();
                        max = str.Length;
                    }

                    result.Add(str.ToString());
                }

                return;
            }

            var ch = s[index];
            if (ch != '(' && ch != ')')
            {
                str.Append(ch);
                RemoveInvalidParenthesesDP(s, index + 1, left, right, result, str, ref max);
            }
            else
            {
                RemoveInvalidParenthesesDP(s, index + 1, left, right, result, str, ref max);
                str.Append(ch);
                if (ch == '(')
                {
                    RemoveInvalidParenthesesDP(s, index + 1, left + 1, right, result, str, ref max);
                }
                else if (ch == ')' && left > right)
                {
                    RemoveInvalidParenthesesDP(s, index + 1, left, right + 1, result, str, ref max);
                }
            }

            str.Remove(str.Length - 1, 1);
        }

        public IList<string> RemoveInvalidParenthesesDP(string s)
        {
            var result = new HashSet<string>();
            int left = 0, right = 0, max = 0;
            RemoveInvalidParenthesesDP(s, 0, left, right, result, new StringBuilder(), ref max);
            return result.ToArray();
        }

        #endregion

        #endregion

        #region 309. 最佳买卖股票时机含冷冻期

        //https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
        public int MaxProfitI(int[] prices)
        {
            if (prices.Length < 2)
            {
                return 0;
            }

            var day = prices.Length;
            var dp = new int[day, 2];
            // dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
            // max(   选择 rest  ,             选择 sell      )
            // 解释：今天我没有持有股票，有两种可能：
            // 要么是我昨天就没有持有，然后今天选择 rest，所以我今天还是没有持有；
            // 要么是我昨天持有股票，但是今天我 sell 了，所以我今天没有持有股票了。
            // dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
            // max(   选择 rest  ,           选择 buy         )
            // 解释：今天我持有着股票，有两种可能：
            // 要么我昨天就持有着股票，然后今天选择 rest，所以我今天还持有着股票；
            // 要么我昨天本没有持有，但今天我选择 buy，所以今天我就持有股票了。
            dp[0, 0] = 0;
            dp[0, 1] = -prices[0];
            for (int i = 1; i < prices.Length; i++)
            {
                dp[i, 0] = Math.Max(dp[i - 1, 0] /*昨天没有*/, dp[i - 1, 1] + prices[i] /*昨天有，今天卖*/);
                dp[i, 1] = Math.Max(dp[i - 1, 1] /*昨天有*/, (i > 1 ? dp[i - 2, 0] : 0) - prices[i] /*前天没有,今天买*/);
            }

            return dp[day - 1, 0];
        }

        #endregion

        #region 621. 任务调度器

        //https://leetcode-cn.com/problems/task-scheduler/
        public int LeastInterval(char[] tasks, int n)
        {
            Dictionary<char, int> taskDict = new Dictionary<char, int>(), waitDict = new Dictionary<char, int>();
            foreach (var task in tasks)
            {
                if (taskDict.TryGetValue(task, out var size))
                {
                    taskDict[task] = size + 1;
                }
                else
                {
                    taskDict[task] = 1;
                    waitDict[task] = 0;
                }
            }

            var time = 0;
            while (taskDict.Count > 0)
            {
                time++;
                var task = waitDict.Where(kv => kv.Value == 0 || time - kv.Value > n)
                    .OrderByDescending(kv => taskDict[kv.Key]).FirstOrDefault();
                if (!taskDict.ContainsKey(task.Key))
                {
                    continue;
                }

                var size = taskDict[task.Key];
                if (size == 1)
                {
                    taskDict.Remove(task.Key);
                    waitDict.Remove(task.Key);
                }
                else
                {
                    taskDict[task.Key] = size - 1;
                    waitDict[task.Key] = time;
                }
            }

            return time;
        }

        public int LeastIntervalBySort(char[] tasks, int n)
        {
            var bucket = new int[26];
            foreach (var task in tasks)
            {
                bucket[task - 'A']++;
            }

            Array.Sort(bucket);
            var time = 0;
            while (bucket[25] > 0)
            {
                var i = 0;
                while (i <= n && bucket[25] > 0)
                {
                    if (i < 26 && bucket[25 - i] > 0)
                    {
                        bucket[25 - i]--;
                    }

                    i++;
                    time++;
                }

                Array.Sort(bucket);
            }

            return time;
        }

        #endregion

        #region 312. 戳气球

        //https://leetcode-cn.com/problems/burst-balloons/

        //暴力解
        void MaxCoins(IList<int> nums, int sum, ref int result)
        {
            if (nums.Count == 1)
            {
                result = Math.Max(result, sum + nums[0]);
                return;
            }

            for (int i = 0; i < nums.Count; i++)
            {
                var rm = nums[i];
                var num = (i > 0 ? nums[i - 1] : 1) * rm * (i < nums.Count - 1 ? nums[i + 1] : 1);
                nums.RemoveAt(i);
                MaxCoins(nums, sum + num, ref result);
                nums.Insert(i, rm);
            }
        }

        int MaxCoinsDp(int[] nums, int[,] cache, int l, int r)
        {
            //头尾各添加1个元素,当nums剩余长度<=2时，实际上nums已空
            if (l + 1 == r)
            {
                return 0;
            }

            if (cache[l, r] != 0)
            {
                return cache[l, r];
            }

            var res = 0;
            for (int i = l + 1; i < r; i++)
            {
                res = Math.Max(res,
                    MaxCoinsDp(nums, cache, l, i) + nums[l] * nums[i] * nums[r] + MaxCoinsDp(nums, cache, i, r));
            }

            cache[l, r] = res;
            return res;
        }

        public int MaxCoins(int[] nums)
        {
            if (nums.Length <= 0)
            {
                return 0;
            }

            var items = new int[nums.Length + 2];
            items[0] = items[items.Length - 1] = 1;
            for (int i = 0; i < nums.Length; i++)
            {
                items[i + 1] = nums[i];
            }

            return MaxCoinsDp(items, new int[items.Length, items.Length], 0, items.Length - 1);
        }

        #endregion

        #region 399. 除法求值

        //https://leetcode-cn.com/problems/evaluate-division/
        public double[] CalcEquation(IList<IList<string>> equations, double[] values, IList<IList<string>> queries)
        {
            var dict = new Dictionary<string, IList<Tuple<string, double>>>();
            for (int i = 0; i < equations.Count; i++)
            {
                var eq = equations[i];
                if (!dict.TryGetValue(eq[0], out var items))
                {
                    items = new List<Tuple<string, double>>();
                    dict[eq[0]] = items;
                }

                items.Add(new Tuple<string, double>(eq[1], values[i]));
                if (!dict.TryGetValue(eq[1], out items))
                {
                    items = new List<Tuple<string, double>>();
                    dict[eq[1]] = items;
                }

                items.Add(new Tuple<string, double>(eq[0], 1.0 / values[i]));
            }

            var result = new double[queries.Count];
            var queue = new Queue<Tuple<string, double>>();
            var visited = new HashSet<string>();
            for (int i = 0; i < queries.Count; i++)
            {
                var query = queries[i];
                result[i] = -1.0;
                string key = query[0], target = query[1];
                if (key == target)
                {
                    result[i] = 1.0;
                    continue;
                }

                if (!dict.ContainsKey(key) || !dict.ContainsKey(target))
                {
                    continue;
                }

                queue.Enqueue(new Tuple<string, double>(key, 1.0));
                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (!visited.Add(current.Item1))
                    {
                        continue;
                    }

                    var next = dict[current.Item1];
                    foreach (var tuple in next)
                    {
                        if (tuple.Item1 == target)
                        {
                            result[i] = tuple.Item2 * current.Item2;
                            queue.Clear();
                            break;
                        }

                        queue.Enqueue(new Tuple<string, double>(tuple.Item1, tuple.Item2 * current.Item2));
                    }
                }

                visited.Clear();
            }

            return result;
        }

        #endregion

        #region 130. 被围绕的区域

        //https://leetcode-cn.com/problems/surrounded-regions/
        public void Solve(char[][] board)
        {
            if (board.Length <= 0 || board[0].Length <= 0)
            {
                return;
            }

            void Set(int x, int y)
            {
                if (x < 0 || x >= board.Length || y < 0 || y >= board[0].Length || board[x][y] == '#' ||
                    board[x][y] == 'X')
                {
                    return;
                }

                board[x][y] = '#';
                Set(x + 1, y);
                Set(x - 1, y);
                Set(x, y + 1);
                Set(x, y - 1);
            }

            for (int i = 0; i < board.Length; i++)
            {
                for (int j = 0; j < board[0].Length; j++)
                {
                    if (board[i][j] == 'X')
                    {
                        continue;
                    }

                    if (i == 0 || j == 0 || i == board.Length - 1 || j == board[0].Length - 1)
                    {
                        Set(i, j);
                    }
                }
            }

            for (int i = 0; i < board.Length; i++)
            {
                for (int j = 0; j < board[0].Length; j++)
                {
                    if (board[i][j] == '#')
                    {
                        board[i][j] = 'O';
                    }
                    else if (board[i][j] == '#')
                    {
                        board[i][j] = 'X';
                    }
                }
            }
        }

        #endregion

        #region 409. 最长回文串

        //https://leetcode-cn.com/problems/longest-palindrome/
        public int LongestPalindromeI(string s)
        {
            var dict = new Dictionary<char, int>();
            foreach (var c in s)
            {
                if (dict.TryGetValue(c, out var n))
                {
                    dict[c] = n + 1;
                }
                else
                {
                    dict[c] = 1;
                }
            }

            var res = 0;
            var single = false;
            foreach (var kv in dict)
            {
                if (kv.Value % 2 == 0)
                {
                    res += kv.Value;
                }
                else
                {
                    res += kv.Value - 1;
                    single = true;
                }
            }

            return res + (single ? 1 : 0);
        }

        #endregion

        #region 1502. 判断能否形成等差数列

        //https://leetcode-cn.com/problems/can-make-arithmetic-progression-from-sequence/
        public bool CanMakeArithmeticProgression(int[] arr)
        {
            if (arr.Length < 3)
            {
                return true;
            }

            ISet<int> set = new HashSet<int>();
            int one = arr[0], two = arr[1];
            if (one < two)
            {
                one = two;
                two = arr[0];
            }

            set.Add(one);
            set.Add(two);
            for (int i = 2; i < arr.Length; i++)
            {
                if (arr[i] > one)
                {
                    two = one;
                    one = arr[i];
                }
                else if (arr[i] > two)
                {
                    two = arr[i];
                }

                set.Add(arr[i]);
            }

            int step = one - two;
            if (step == 0)
            {
                return set.Count == 1;
            }

            set.Add(one + step);
            return arr.All(anArr => set.Contains(anArr + step));
        }

        #endregion

        #region 210. 课程表 II

        //https://leetcode-cn.com/problems/course-schedule-ii/
        public int[] FindOrder(int numCourses, int[][] prerequisites)
        {
            if (prerequisites.Length <= 0 || prerequisites[0].Length <= 0)
            {
                return Enumerable.Range(0, numCourses).ToArray();
            }

            var preDict = new Dictionary<int, ISet<int>>();
            var indexs = new int[numCourses];
            foreach (var prerequisite in prerequisites)
            {
                int k = prerequisite[1], v = prerequisite[0];
                if (!preDict.TryGetValue(k, out var set))
                {
                    set = new HashSet<int>();
                    preDict[k] = set;
                }

                set.Add(v);
                indexs[v]++;
            }

            var starts = new Queue<int>();
            for (int i = 0; i < numCourses; i++)
            {
                if (indexs[i] == 0)
                {
                    starts.Enqueue(i);
                }
            }

            if (starts.Count <= 0)
            {
                return new int[0];
            }

            var result = new int[numCourses];
            var index = 0;
            while (starts.Count > 0)
            {
                var start = starts.Dequeue();
                result[index++] = start;
                if (!preDict.TryGetValue(start, out var pres))
                {
                    continue;
                }

                foreach (var pre in pres)
                {
                    indexs[pre]--;
                    if (indexs[pre] == 0)
                    {
                        starts.Enqueue(pre);
                    }
                }
            }

            return index != numCourses ? new int[0] : result;
        }

        #endregion

        #region 150. 逆波兰表达式求值

        //https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/
        public int EvalRPN(string[] tokens)
        {
            var dictFunc = new Dictionary<string, Func<int, int, int>>();
            dictFunc.Add("+", (a, b) => a + b);
            dictFunc.Add("-", (a, b) => a - b);
            dictFunc.Add("*", (a, b) => a * b);
            dictFunc.Add("/", (a, b) => a / b);
            var stack = new Stack<int>();
            foreach (var token in tokens)
            {
                if (dictFunc.TryGetValue(token, out var func))
                {
                    int a = stack.Pop(), b = stack.Pop();
                    stack.Push(func(b, a));
                }
                else
                {
                    stack.Push(int.Parse(token));
                }
            }

            return stack.Pop();
        }

        #endregion

        #region 227. 基本计算器 II

        //https://leetcode-cn.com/problems/basic-calculator-ii/
        public int Calculate(string s)
        {
            var level = new Dictionary<char, int>();
            level.Add('+', 0);
            level.Add('-', 0);
            level.Add('*', 1);
            level.Add('/', 1);
            var stack = new Stack<int>();
            var operators = new Stack<char>();
            var num = 0;
            s = s.Trim();
            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                if (ch == ' ')
                {
                    continue;
                }

                if (char.IsDigit(ch))
                {
                    num = num * 10 + (ch - '0');
                    if (i != s.Length - 1)
                    {
                        continue;
                    }
                }

                stack.Push(num);
                num = 0;
                while (operators.Count > 0 && (i == s.Length - 1 || level[ch] <= level[operators.Peek()]))
                {
                    int n1 = stack.Pop(), n2 = stack.Pop();
                    var pre = operators.Pop();
                    switch (pre)
                    {
                        case '*':
                            stack.Push(n1 * n2);
                            break;
                        case '/':
                            stack.Push(n2 / n1);
                            break;
                        case '+':
                            stack.Push(n1 + n2);
                            break;
                        case '-':
                            stack.Push(n2 - n1);
                            break;
                    }
                }

                operators.Push(ch);
            }

            return stack.Pop();
        }

        #endregion

        #region 134. 加油站

        //https://leetcode-cn.com/problems/gas-station/
        public int CanCompleteCircuit(int[] gas, int[] cost)
        {
            for (int i = 0; i < gas.Length; i++)
            {
                var cur = gas[i];
                int pre = i, ci = i + 1;
                while (true)
                {
                    if (ci >= cost.Length)
                    {
                        ci = 0;
                        pre = cost.Length - 1;
                    }

                    cur -= cost[pre];
                    if (cur < 0)
                    {
                        break;
                    }

                    if (ci == i)
                    {
                        return i;
                    }

                    cur += gas[ci];
                    pre = ci;
                    ci++;
                }
            }

            return -1;
        }

        public int CanCompleteCircuitON(int[] gas, int[] cost)
        {
            int index = -1, spare = 0, min = int.MaxValue;
            for (int i = 0; i < gas.Length; i++)
            {
                spare += gas[i] - cost[i];
                if (spare < min)
                {
                    min = spare;
                    index = i;
                }
            }

            return spare < 0 ? -1 : (index + 1) % gas.Length;
        }

        #endregion

        #region 212. 单词搜索 II

        //https://leetcode-cn.com/problems/word-search-ii/
        class TrieTree
        {
            public char Char;
            public bool IsWord;
            public TrieTree[] Trees;
        }

        void FindWords(int x, int y, char[][] board, TrieTree trieTree, bool[,] visited, ICollection<string> result,
            IList<char> sub)
        {
            if (x < 0 || x >= board.Length || y < 0 || y >= board[0].Length || visited[x, y])
            {
                return;
            }

            var ch = board[x][y];
            if (trieTree.Char != ch)
            {
                return;
            }

            sub.Add(ch);
            if (trieTree.IsWord)
            {
                result.Add(new string(sub.ToArray()));
            }

            visited[x, y] = true;
            if (trieTree.Trees != null)
            {
                foreach (var tree in trieTree.Trees)
                {
                    if (tree == null)
                    {
                        continue;
                    }

                    FindWords(x - 1, y, board, tree, visited, result, sub);
                    FindWords(x + 1, y, board, tree, visited, result, sub);
                    FindWords(x, y - 1, board, tree, visited, result, sub);
                    FindWords(x, y + 1, board, tree, visited, result, sub);
                }
            }

            visited[x, y] = false;
            sub.RemoveAt(sub.Count - 1);
        }

        public IList<string> FindWords(char[][] board, string[] words)
        {
            var treeList = new TrieTree[26];
            foreach (var word in words)
            {
                var currentTree = treeList;
                for (int i = 0; i < word.Length; i++)
                {
                    var ch = word[i];
                    var tree = currentTree[ch - 'a'];
                    if (tree == null)
                    {
                        tree = new TrieTree {Char = ch, Trees = new TrieTree[26]};
                        currentTree[ch - 'a'] = tree;
                    }

                    tree.IsWord = tree.IsWord || i == word.Length - 1;
                    currentTree = tree.Trees;
                }
            }

            var result = new HashSet<string>();
            var visited = new bool[board.Length, board[0].Length];
            for (int i = 0; i < board.Length; i++)
            {
                for (int j = 0; j < board[0].Length; j++)
                {
                    var trieTree = treeList[board[i][j] - 'a'];
                    if (trieTree == null)
                    {
                        continue;
                    }

                    FindWords(i, j, board, trieTree, visited, result, new List<char>());
                }
            }

            return result.ToArray();
        }

        #endregion

        #region 1306. 跳跃游戏 III

        //https://leetcode-cn.com/problems/jump-game-iii/
        public bool CanReach(int[] arr, int start)
        {
            var visited = new HashSet<int>();
            var queue = new Queue<int>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                start = queue.Dequeue();
                if (arr[start] == 0)
                {
                    return true;
                }

                if (!visited.Add(start))
                {
                    continue;
                }

                if (start + arr[start] < arr.Length)
                {
                    queue.Enqueue(start + arr[start]);
                }

                if (start - arr[start] > -1)
                {
                    queue.Enqueue(start - arr[start]);
                }
            }

            return false;
        }

        #endregion

        #region 1232. 缀点成线

        //https://leetcode-cn.com/problems/check-if-it-is-a-straight-line/
        public bool CheckStraightLine(int[][] coordinates)
        {
            if (coordinates.Length <= 2)
            {
                return true;
            }

            int x = coordinates[0][0] - coordinates[1][0], y = coordinates[0][1] - coordinates[1][1];
            for (int i = 2; i < coordinates.Length; i++)
            {
                //求斜率 x1/y1=x2/y2 可以转换成 x1*y2=x2*y1
                int x1 = coordinates[i - 1][0] - coordinates[i][0], y1 = coordinates[i - 1][1] - coordinates[i][1];
                if (x1 * y != y1 * x)
                {
                    return false;
                }
            }

            return true;
        }

        #endregion

        #region 468. 验证IP地址

        //https://leetcode-cn.com/problems/validate-ip-address/
        bool ValidIP4Address(string ip)
        {
            var arr = ip.Split('.');
            if (arr.Length != 4)
            {
                return false;
            }

            for (int i = 0; i < arr.Length; i++)
            {
                if (arr[i].Length <= 0 || arr[i][0] == '0' && (i == 0 || arr[i].Length > 1))
                {
                    return false;
                }

                if (!byte.TryParse(arr[i], out _))
                {
                    return false;
                }
            }

            return true;
        }

        bool ValidIP6Address(string ip)
        {
            var arr = ip.Split(':');
            if (arr.Length != 8)
            {
                return false;
            }

            foreach (var s in arr)
            {
                if (s.Length <= 0 || s.Length > 4)
                {
                    return false;
                }

                if (s.Any(ch => ('0' > ch || ch > '9') && ('a' > ch || ch > 'f') && ('A' > ch || ch > 'F')))
                {
                    return false;
                }
            }

            return true;
        }

        public string ValidIPAddress(string ip)
        {
            if (ip.IndexOf('.') > 0 && ValidIP4Address(ip))
            {
                return "IPv4";
            }

            if (ip.IndexOf(':') > 0 && ValidIP6Address(ip))
            {
                return "IPv6";
            }

            return "Neither";
        }

        #endregion

        #region 179. 最大数

        //https://leetcode-cn.com/problems/largest-number/
        public string LargestNumber(int[] nums)
        {
            Array.Sort(nums, Comparer<int>.Create((a, b) =>
            {
                string s1 = a + string.Empty + b, s2 = b + string.Empty + a;
                return s2.CompareTo(s1);
            }));
            if (nums[0] == 0)
            {
                return "0";
            }

            var res = new StringBuilder();
            foreach (var num in nums)
            {
                res.Append(num);
            }

            return res.ToString();
        }

        #endregion

        #region 1002. 查找常用字符

        //https://leetcode-cn.com/problems/find-common-characters/
        public IList<string> CommonChars(string[] a)
        {
            if (a.Length <= 0)
            {
                return new string[0];
            }

            var dict = new int[26];
            foreach (var ch in a[0])
            {
                dict[ch - 'a']++;
            }

            var temp = new int[26];
            for (int i = 1; i < a.Length; i++)
            {
                foreach (var ch in a[i])
                {
                    temp[ch - 'a']++;
                }

                for (int j = 0; j < dict.Length; j++)
                {
                    if (dict[j] != 0 && temp[j] != 0)
                    {
                        dict[j] = Math.Min(dict[j], temp[j]);
                    }
                    else
                    {
                        dict[j] = 0;
                    }

                    temp[j] = 0;
                }
            }

            var result = new List<string>();
            for (int i = 0; i < dict.Length; i++)
            {
                if (dict[i] == 0)
                {
                    continue;
                }

                result.AddRange(Enumerable.Repeat(((char) (i + 'a')).ToString(), dict[i]));
            }

            return result;
        }

        #endregion

        #region 面试题 16.11. 跳水板

        //https://leetcode-cn.com/problems/diving-board-lcci/

        #region 回溯(超时)

        void DivingBoard(int index, IList<int> lens, int k, int len, IList<int> result)
        {
            if (k == 0)
            {
                result.Add(len);
                return;
            }

            for (int i = index; i < lens.Count; i++)
            {
                if (i > index && lens[i] == lens[i - 1])
                {
                    continue;
                }

                DivingBoard(i + 1, lens, k - 1, len + lens[i], result);
            }
        }

        public int[] DivingBoard(int shorter, int longer, int k)
        {
            if (k <= 0)
            {
                return new int[0];
            }

            if (shorter == longer)
            {
                return new[] {longer * k};
            }

            var nums = new int[k * 2];
            for (int i = 0, j = k; i < k; i++, j++)
            {
                nums[i] = shorter;
                nums[j] = longer;
            }

            var result = new List<int>();
            DivingBoard(0, nums, k, 0, result);
            return result.ToArray();
        }

        #endregion

        #region 递归（超时）

        public int[] DivingBoardR(int shorter, int longer, int k)
        {
            if (k <= 0)
            {
                return new int[0];
            }

            if (shorter == longer)
            {
                return new[] {longer * k};
            }

            if (k == 1)
            {
                return new[] {shorter, longer};
            }

            var items = DivingBoardR(shorter, longer, k - 1);
            if (items.Length <= 0)
            {
                return new[] {shorter, longer};
            }

            var res = new List<int>();
            foreach (var item in items)
            {
                res.Add(item + shorter);
            }

            foreach (var item in items)
            {
                var n = item + longer;
                if (res[res.Count - 1] >= n)
                {
                    continue;
                }

                res.Add(n);
            }

            return res.ToArray();
        }

        #endregion

        //数学
        public int[] DivingBoardMath(int shorter, int longer, int k)
        {
            if (k <= 0)
            {
                return new int[0];
            }

            if (shorter == longer)
            {
                return new[] {longer * k};
            }

            if (k == 1)
            {
                return new[] {shorter, longer};
            }

            var res = new List<int>();
            for (int i = 0; i <= k; i++)
            {
                res.Add((k - i) * shorter + longer * i);
            }

            return res.ToArray();
        }

        #endregion

        #region 1072. 按列翻转得到最大值等行数

        //https://leetcode-cn.com/problems/flip-columns-for-maximum-number-of-equal-rows/
        //位运算(异或操作，不同位变1，找出异或后相同的数量，即为最大数)
        //题解：https://leetcode-cn.com/problems/flip-columns-for-maximum-number-of-equal-rows/solution/xun-zhao-ju-you-xiang-tong-de-te-zheng-de-xing-de-/
        public int MaxEqualRowsAfterFlips(int[][] matrix)
        {
            if (matrix.Length <= 0 || matrix[0].Length <= 0)
            {
                return 0;
            }

            var dict = new Dictionary<string, int>();
            var keyBuilder = new StringBuilder();
            var res = 0;
            foreach (var ints in matrix)
            {
                var flag = ints[0] == 0;
                foreach (var i in ints)
                {
                    if (flag)
                    {
                        keyBuilder.Append(i);
                    }
                    else
                    {
                        keyBuilder.Append(i ^ 1);
                    }
                }

                var key = keyBuilder.ToString();
                if (dict.TryGetValue(key, out var n))
                {
                    dict[key] = n + 1;
                }
                else
                {
                    dict[key] = 1;
                }

                res = Math.Max(res, dict[key]);
                keyBuilder.Clear();
            }

            return res;
        }

        #endregion

        #region 830. 较大分组的位置

        //https://leetcode-cn.com/problems/positions-of-large-groups/
        public IList<IList<int>> LargeGroupPositions(string s)
        {
            if (s.Length < 3)
            {
                return new IList<int>[0];
            }

            var result = new List<IList<int>>();
            var size = 0;
            for (int i = 0, j = 0; i < s.Length; i++)
            {
                if (s[i] == s[j])
                {
                    size++;
                    continue;
                }

                if (size >= 3)
                {
                    result.Add(new[] {j, i - 1});
                }

                j = i;
                size = 1;
            }

            if (size >= 3)
            {
                result.Add(new[] {s.Length - size, s.Length - 1});
            }

            return result.ToArray();
        }

        #endregion

        #region 1433. 检查一个字符串是否可以打破另一个字符串

        //https://leetcode-cn.com/problems/check-if-a-string-can-break-another-string/
        public bool CheckIfCanBreak(string s1, string s2)
        {
            char[] chars1 = s1.ToCharArray(), chars2 = s2.ToCharArray();
            Array.Sort(chars1);
            Array.Sort(chars2);
            bool flag1 = true, flag2 = true;
            for (int i = 0; i < chars1.Length; i++)
            {
                if (flag1 && chars1[i] < chars2[i])
                {
                    flag1 = false;
                }

                if (flag2 && chars1[i] > chars2[i])
                {
                    flag2 = false;
                }

                if (!flag1 && !flag2)
                {
                    return false;
                }
            }

            return true;
        }

        #endregion

        #region 1262. 可被三整除的最大和

        //https://leetcode-cn.com/problems/greatest-sum-divisible-by-three/
        public int MaxSumDivThree(int[] nums)
        {
            if (nums.Length <= 0)
            {
                return 0;
            }

            if (nums.Length == 1)
            {
                return nums[0] % 3 == 0 ? nums[0] : 0;
            }

            var dp = new int[3];
            var res = new int[3];
            for (int i = 0; i < nums.Length; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    var num = res[j] + nums[i];
                    var mod = num % 3;
                    dp[mod] = Math.Max(dp[mod], num);
                }

                for (int j = 0; j < 3; j++)
                {
                    res[j] = Math.Max(res[j], dp[j]);
                }
            }

            return dp[0];
        }

        #endregion

        #region 1185. 一周中的第几天

        //https://leetcode-cn.com/problems/day-of-the-week/
        public string DayOfTheWeek(int day, int month, int year)
        {
            var daysToMonth365 = new[]
            {
                0,
                31,
                59,
                90,
                120,
                151,
                181,
                212,
                243,
                273,
                304,
                334,
                365
            };
            var daysToMonth366 = new[]
            {
                0,
                31,
                60,
                91,
                121,
                152,
                182,
                213,
                244,
                274,
                305,
                335,
                366
            };
            var weeks = new[] {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
            var numArray = DateTime.IsLeapYear(year) ? daysToMonth366 : daysToMonth365;
            var num = year - 1;
            var days = num * 365 + num / 4 - num / 100 + num / 400 + numArray[month - 1] + day;
            return weeks[days % weeks.Length];
        }

        #endregion

        #region 1008. 先序遍历构造二叉树

        //https://leetcode-cn.com/problems/construct-binary-search-tree-from-preorder-traversal/
        TreeNode BstFromPreorder(int[] preorder, int start, int end)
        {
            if (start > end)
            {
                return null;
            }

            var root = new TreeNode(preorder[start]);
            for (int i = start + 1; i <= end; i++)
            {
                if (preorder[i] > preorder[start])
                {
                    root.left = BstFromPreorder(preorder, start + 1, i - 1);
                    root.right = BstFromPreorder(preorder, i, end);
                    return root;
                }
            }

            root.left = BstFromPreorder(preorder, start + 1, end);
            return root;
        }

        TreeNode BstFromPreorder(int[] preorder, int start, int end, ref int index)
        {
            if (index >= preorder.Length)
            {
                return null;
            }

            var val = preorder[index];
            if (val < start || val > end)
            {
                return null;
            }

            index++;
            var root = new TreeNode(val);
            root.left = BstFromPreorder(preorder, start, val, ref index);
            root.right = BstFromPreorder(preorder, val, end, ref index);
            return root;
        }

        public TreeNode BstFromPreorder(int[] preorder)
        {
            var ignore = 0;
            return BstFromPreorder(preorder, int.MinValue, int.MaxValue, ref ignore);
        }

        #endregion

        #region 187. 重复的DNA序列

        //https://leetcode-cn.com/problems/repeated-dna-sequences/
        public IList<string> FindRepeatedDnaSequences(string s)
        {
            ISet<string> tmp = new HashSet<string>(), res = new HashSet<string>();
            for (int i = 0, end = s.Length - 10; i <= end; i++)
            {
                var key = s.Substring(i, 10);
                if (!tmp.Add(key))
                {
                    res.Add(key);
                }
            }

            return res.ToArray();
        }

        #endregion

        #region 1380. 矩阵中的幸运数

        //https://leetcode-cn.com/problems/lucky-numbers-in-a-matrix/
        public IList<int> LuckyNumbers(int[][] matrix)
        {
            var result = new List<int>();
            for (int i = 0; i < matrix.Length; i++)
            {
                var nums = matrix[i];
                var min = 0;
                for (int j = 1; j < nums.Length; j++)
                {
                    if (nums[j] < nums[min])
                    {
                        min = j;
                    }
                }

                var flag = true;
                for (int r = 0; r < matrix.Length; r++)
                {
                    if (matrix[r][min] > nums[min])
                    {
                        flag = false;
                        break;
                    }
                }

                if (flag)
                {
                    result.Add(nums[min]);
                }
            }

            return result;
        }

        public IList<int> LuckyNumbersN(int[][] matrix)
        {
            if (matrix.Length <= 0 || matrix[0].Length <= 0)
            {
                return new int[0];
            }

            var result = new List<int>();
            var mins = new int[matrix.Length];
            var maxs = new int[matrix[0].Length];
            Array.Fill(mins, int.MaxValue);
            for (int i = 0; i < matrix.Length; i++)
            {
                var nums = matrix[i];
                for (int j = 0; j < nums.Length; j++)
                {
                    mins[i] = Math.Min(mins[i], nums[j]);
                    maxs[j] = Math.Max(maxs[j], nums[j]);
                }
            }

            for (int i = 0; i < matrix.Length; i++)
            {
                var nums = matrix[i];
                for (int j = 0; j < nums.Length; j++)
                {
                    if (nums[j] == mins[i] && nums[j] == maxs[j])
                    {
                        result.Add(nums[j]);
                    }
                }
            }

            return result;
        }

        #endregion

        #region 面试题 17.13. 恢复空格

        //https://leetcode-cn.com/problems/re-space-lcci/
        //思路：求出以s[i]结尾的字符串匹配的最多字符数 s.length-dp[i]即为最小变更数
        public int Respace(string[] dictionary, string sentence)
        {
            var treeList = new TrieTree[26];
            foreach (var word in dictionary)
            {
                var currentTree = treeList;
                for (int i = 0; i < word.Length; i++)
                {
                    var ch = word[i];
                    var tree = currentTree[ch - 'a'];
                    if (tree == null)
                    {
                        tree = new TrieTree {Char = ch, Trees = new TrieTree[26]};
                        currentTree[ch - 'a'] = tree;
                    }

                    tree.IsWord = tree.IsWord || i == word.Length - 1;
                    currentTree = tree.Trees;
                }
            }

            var dp = new int[sentence.Length + 1];
            for (int i = 0; i < sentence.Length; i++)
            {
                var list = treeList;
                dp[i + 1] = Math.Max(dp[i + 1], dp[i]);
                for (int j = i; j < sentence.Length; j++)
                {
                    var ch = sentence[j] - 'a';
                    var tree = list[ch];
                    if (tree == null)
                    {
                        break;
                    }

                    if (tree.IsWord)
                    {
                        dp[j + 1] = Math.Max(dp[j + 1], dp[i] + j - i + 1);
                    }

                    if (tree.Trees == null)
                    {
                        break;
                    }

                    list = tree.Trees;
                }
            }

            return sentence.Length - dp[dp.Length - 1];
        }

        #endregion

        #region 887. 鸡蛋掉落

        //https://leetcode-cn.com/problems/super-egg-drop/
        public int SuperEggDrop(int k, int n)
        {
            var cache = new int[k + 1, n + 1];

            int Dp(int eggs, int levels)
            {
                //只有一个鸡蛋，只能够从第一层开始一层一层扔
                if (eggs == 1)
                {
                    return levels;
                }

                if (levels == 0)
                {
                    return 0;
                }

                if (cache[eggs, levels] != 0)
                {
                    return cache[eggs, levels];
                }

                int res = n;
                int l = 1, r = levels;
                while (l <= r)
                {
                    var mid = (l + r) / 2;
                    var brokens = Dp(eggs - 1, mid - 1); //鸡蛋碎了，需要从下一层开始eggs-1,level-1
                    var notBrokens = Dp(eggs, levels - mid); //鸡蛋没碎，继续试，还是levels-level层要试
                    if (brokens > notBrokens)
                    {
                        res = Math.Min(res, brokens + 1);
                        r = mid - 1; //鸡蛋碎了，需要从mid层下找
                    }
                    else
                    {
                        res = Math.Min(res, notBrokens + 1);
                        l = mid + 1; //鸡蛋没碎，从mid层1上找
                    }
                }

                cache[eggs, levels] = res;
                return res;
            }

            return Dp(k, n);
        }

        #endregion

        #region 18. 四数之和

        //https://leetcode-cn.com/problems/4sum/
        public IList<IList<int>> FourSum(int[] nums, int target)
        {
            var result = new List<IList<int>>();
            Array.Sort(nums);
            for (int i = 0; i < nums.Length; i++)
            {
                if (i > 0 && nums[i] == nums[i - 1])
                {
                    continue;
                }

                for (int j = i + 1; j < nums.Length; j++)
                {
                    if (j > i + 1 && nums[j] == nums[j - 1])
                    {
                        continue;
                    }

                    int start = j + 1, end = nums.Length - 1;
                    var baseNum = nums[i] + nums[j];
                    while (start < end)
                    {
                        var num = baseNum + nums[start] + nums[end];
                        if (num == target)
                        {
                            result.Add(new int[] {nums[i], nums[j], nums[start], nums[end]});
                            while (start < end && nums[start] == nums[start + 1])
                            {
                                start++;
                            }

                            while (start < end && nums[end] == nums[end - 1])
                            {
                                end--;
                            }

                            start++;
                            end--;
                        }
                        else if (num < target)
                        {
                            start++;
                        }
                        else
                        {
                            end--;
                        }
                    }
                }
            }

            return result;
        }

        #endregion
    }
}