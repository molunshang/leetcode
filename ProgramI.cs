﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Json;
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

        #region 42. 接雨水

        //https://leetcode-cn.com/problems/trapping-rain-water/
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
            for (int i = 1; i < height.Length; i++)
            {
                leftMax[i] = Math.Max(leftMax[i - 1], height[i]);
            }

            for (int i = height.Length - 2; i >= 0; i--)
            {
                rightMax[i] = Math.Max(rightMax[i + 1], height[i]);
            }

            for (int i = 1; i < height.Length - 1; i++)
            {
                res += Math.Min(leftMax[i], rightMax[i]) - height[i];
            }

            return res;
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
            bool Dfs(Dictionary<int, ISet<int>> dict, int key, ISet<int> paths)
            {
                if (!paths.Add(key))
                {
                    return false;
                }

                if (dict.TryGetValue(key, out var next))
                {
                    if (next.Any(k => !Dfs(dict, k, paths)))
                    {
                        return false;
                    }
                }

                paths.Remove(key);
                return true;
            }

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

            var path = new HashSet<int>();
            return depend.All(kv => Dfs(depend, kv.Key, path));
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
            var martix = new bool[3, 3][];

            bool Set(int i, int j, int index)
            {
                if (index >= 81)
                {
                    return true;
                }

                if (j >= 9)
                {
                    j = 0;
                    i++;
                }

                var row = board[i];
                if (row[j] == '.')
                {
                    int rIndex = i / 3, cIndex = j / 3;
                    for (int num = 0; num < 9; num++)
                    {
                        if (rows[i, num] || cols[j, num] || martix[rIndex, cIndex][num])
                        {
                            continue;
                        }

                        row[j] = (char)('1' + num);
                        rows[i, num] = cols[j, num] = martix[rIndex, cIndex][num] = true;
                        if (Set(i, j + 1, index + 1))
                        {
                            return true;
                        }

                        rows[i, num] = cols[j, num] = martix[rIndex, cIndex][num] = false;
                        row[j] = '.';
                    }

                    return false;
                }

                return Set(i, j + 1, index + 1);
            }

            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    var row = board[i];
                    int rIndex = i / 3, cIndex = j / 3;
                    if (martix[rIndex, cIndex] == null)
                    {
                        martix[rIndex, cIndex] = new bool[9];
                    }

                    if (row[j] == '.')
                    {
                        continue;
                    }

                    var n = row[j] - '1';
                    rows[i, n] = cols[j, n] = martix[rIndex, cIndex][n] = true;
                }
            }

            Set(0, 0, 0);
        }

        #endregion

        #region 209. 长度最小的子数组

        //https://leetcode-cn.com/problems/minimum-size-subarray-sum/
        public int MinSubArrayLen(int s, int[] nums)
        {
            int len = nums.Length, sum = 0;
            var found = false;
            for (int i = 0, j = 0; i < nums.Length; i++)
            {
                sum += nums[i];
                while (sum >= s && j <= i)
                {
                    found = true;
                    len = Math.Min(len, i - j + 1);
                    sum -= nums[j];
                    j++;
                }
            }

            return found ? len : 0;
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
                    return new TreeNode[] { null };
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

        #region 72. 编辑距离

        //https://leetcode-cn.com/problems/edit-distance/
        public int MinDistance(string word1, string word2)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}