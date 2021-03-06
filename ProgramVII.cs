﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 304. 二维区域和检索 - 矩阵不可变

        //https://leetcode-cn.com/problems/range-sum-query-2d-immutable/
        class NumMatrix
        {
            private int[,] prefixSums;
            private int[][] matrix;

            public NumMatrix(int[][] matrix)
            {
                this.matrix = matrix;
                if (matrix.Length <= 0 || matrix[0].Length <= 0)
                {
                    return;
                }

                int m = matrix.Length, n = matrix[0].Length;
                prefixSums = new int[m, n];
                for (int i = 0; i < m; i++)
                {
                    var arr = matrix[i];
                    prefixSums[i, 0] = arr[0];
                    for (int j = 1; j < n; j++)
                    {
                        prefixSums[i, j] = prefixSums[i, j - 1] + arr[j];
                    }
                }
            }

            public int SumRegion(int row1, int col1, int row2, int col2)
            {
                if (prefixSums == null)
                {
                    return 0;
                }

                var sum = 0;
                for (int i = row1; i <= row2; i++)
                {
                    sum += prefixSums[i, col2] - prefixSums[i, col1] + matrix[i][col1];
                }

                return sum;
            }
        }

        #endregion

        #region 354. 俄罗斯套娃信封问题

        //https://leetcode-cn.com/problems/russian-doll-envelopes/
        public int MaxEnvelopes(int[][] envelopes)
        {
            if (envelopes.Length < 2)
            {
                return envelopes.Length;
            }

            Array.Sort(envelopes, Comparer<int[]>.Create((a, b) => a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]));
            var dp = new int[envelopes.Length];
            dp[0] = 1;
            var ans = 1;
            for (int i = 1; i < dp.Length; i++)
            {
                var count = 0;
                var big = envelopes[i];
                for (int j = 0; j < i; j++)
                {
                    if (envelopes[j][0] < big[0] && envelopes[j][1] < big[1])
                    {
                        count = Math.Max(count, dp[j]);
                    }
                }

                dp[i] = count + 1;
                ans = Math.Max(ans, dp[i]);
            }

            return ans;
        }

        #endregion

        #region 503. 下一个更大元素 II

        //https://leetcode-cn.com/problems/next-greater-element-ii/
        public int[] NextGreaterElements(int[] nums)
        {
            var res = new int[nums.Length];
            Array.Fill(res, -1);
            var stack = new Stack<int>();
            for (int i = 0, n = nums.Length; i < nums.Length * 2 - 1; i++)
            {
                var num = nums[i % n];
                while (stack.TryPeek(out var j) && nums[j] < num)
                {
                    res[stack.Pop()] = num;
                }

                stack.Push(i % n);
            }

            return res;
        }

        #endregion

        #region 132. 分割回文串 II

        //https://leetcode-cn.com/problems/palindrome-partitioning-ii/
        public int MinCut(string s)
        {
            var dp = new bool[s.Length, s.Length];
            for (int l = 1; l <= s.Length; l++)
            {
                for (int i = 0, j = i + l - 1; j < s.Length; i++, j++)
                {
                    switch (l)
                    {
                        case 1:
                            dp[i, j] = true;
                            break;
                        case 2:
                            dp[i, j] = s[i] == s[j];
                            break;
                        default:
                            dp[i, j] = s[i] == s[j] && dp[i + 1, j - 1];
                            break;
                    }
                }
            }

            var ans = new int[s.Length];
            //判断以i结尾的字符串分割次数
            for (int i = 0; i < s.Length; i++)
            {
                if (dp[0, i])
                {
                    ans[i] = 0;
                }
                else
                {
                    ans[i] = int.MaxValue;
                    for (int j = 0; j < i; j++)
                    {
                        if (dp[j + 1, i])
                        {
                            ans[i] = Math.Min(ans[i], ans[j] + 1);
                        }
                    }
                }
            }

            return ans[s.Length - 1];
        }

        #endregion

        #region 1047. 删除字符串中的所有相邻重复项

        //https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/
        public string RemoveDuplicates(string s)
        {
            var stack = new Stack<char>();
            foreach (var ch in s)
            {
                if (stack.TryPeek(out var h) && h == ch)
                {
                    stack.Pop();
                    continue;
                }

                stack.Push(ch);
            }

            var chars = new char[stack.Count];
            for (int i = chars.Length - 1; i >= 0; i--)
            {
                chars[i] = stack.Pop();
            }

            return new string(chars);
        }

        #endregion

        #region 331. 验证二叉树的前序序列化

        //https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/
        public bool IsValidSerialization(string preorder)
        {
            if (!preorder.EndsWith("#"))
            {
                return false;
            }

            var arr = preorder.Split(',');
            var stack = new Stack<string>();
            for (int i = 0; i < arr.Length; i++)
            {
                var str = arr[i];
                if (str == "#")
                {
                    if (stack.TryPop(out _))
                    {
                        //最后一个栈能弹出说明格式有问题
                        if (i == arr.Length - 1)
                        {
                            return false;
                        }
                    }
                    else if (i != arr.Length - 1) //最后一个栈应该已经空了
                    {
                        return false;
                    }
                }
                else
                {
                    stack.Push(str);
                }
            }

            return stack.Count <= 0;
        }

        public bool IsValidSerializationByLeetcode(string preorder)
        {
            if (!preorder.EndsWith("#"))
            {
                return false;
            }

            var slots = 1;
            for (int i = 0; i < preorder.Length; i++)
            {
                if (slots == 0)
                {
                    return false;
                }

                if (preorder[i] == ',')
                {
                    continue;
                }

                if (preorder[i] == '#')
                {
                    slots--;
                }
                else
                {
                    while (i < preorder.Length - 1 && char.IsDigit(preorder[i + 1]))
                    {
                        i++;
                    }

                    slots++;
                }
            }

            return slots == 0;
        }

        #endregion

        #region 456. 132 模式

        //https://leetcode-cn.com/problems/132-pattern/
        //leetcode题解，单调栈
        public bool Find132pattern(int[] nums)
        {
            var stack = new Stack<int>();
            var num2 = int.MinValue;
            for (int i = nums.Length - 1; i >= 0; i--)
            {
                if (nums[i] < num2)
                {
                    return true;
                }

                while (stack.TryPeek(out var h) && nums[i] > h)
                {
                    num2 = stack.Pop();
                }

                if (nums[i] > num2)
                {
                    stack.Push(nums[i]);
                }
            }

            return false;
        }

        #endregion

        #region 1006. 笨阶乘
        //https://leetcode-cn.com/problems/clumsy-factorial/
        public int Clumsy(int n)
        {
            if (n <= 1)
            {
                return n;
            }
            var stack = new Stack<int>();
            stack.Push(n);
            for (int i = n - 1, t = 0; i > 0; i--)
            {
                if (t == 0)
                {
                    stack.Push(stack.Pop() * i);
                }
                else if (t == 1)
                {
                    stack.Push(stack.Pop() / i);
                }
                else if (t == 2)
                {
                    stack.Push(i);
                }
                else
                {
                    stack.Push(-i);
                }
                t = (t + 1) % 4;
            }
            var result = 0;
            while (stack.TryPop(out var num))
            {
                result += num;
            }
            return result;
        }

        //数学解法（Leetcode）
        public int ClumsyByLeetcode(int n)
        {
            switch (n)
            {
                case 1:
                    return 1;
                case 2:
                    return 2;
                case 3:
                    return 6;
                case 4:
                    return 7;
            }
            if (n % 4 == 0)
            {
                return n + 1;
            }
            else if (n % 4 <= 2)
            {
                return n + 2;
            }
            else
            {
                return n - 1;
            }
        }
        #endregion

        #region 781. 森林中的兔子
        //https://leetcode-cn.com/problems/rabbits-in-forest/
        public int NumRabbits(int[] answers)
        {
            if (answers.Length <= 0)
            {
                return 0;
            }
            var count = 0;
            var dict = new Dictionary<int, int>();
            foreach (var ans in answers)
            {
                var n = ans + 1;
                if (dict.TryGetValue(n, out var c))
                {
                    c++;
                    if (c > n)
                    {
                        c = 1;
                    }
                }
                else
                {
                    c = 1;
                }
                if (c == 1)
                {
                    count += n;
                }
                dict[n] = c;
            }
            return count;
        }
        #endregion

        #region 783. 二叉搜索树节点最小距离
        //https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/
        public int MinDiffInBST(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }
            var diff = int.MaxValue;
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
                if (prev != null)
                {
                    diff = Math.Min(diff, root.val - prev.val);
                }
                prev = root;
                root = root.right;
            }
            return diff == int.MaxValue ? 0 : diff;
        }
        #endregion

        #region 220. 存在重复元素 III
        //https://leetcode-cn.com/problems/contains-duplicate-iii/
        public bool ContainsNearbyAlmostDuplicate(int[] nums, int k, int t)
        {
            if (k < 0 || t < 0)
            {
                return false;
            }
            var sortedDict = new SortedDictionary<long, IList<int>>();
            for (int i = 0; i < nums.Length; i++)
            {
                var n = nums[i];
                if (!sortedDict.TryGetValue(n, out var indexs))
                {
                    indexs = sortedDict[n] = new List<int>();
                }
                indexs.Add(i);
            }
            var keys = sortedDict.Keys.ToList();
            foreach (var kv in sortedDict)
            {
                var key = kv.Key;
                int s = keys.BinarySearch(key - t), e = keys.BinarySearch(key + t);
                s = s < 0 ? ~s : s;
                e = e < 0 ? (~e - 1) : e;
                while (s <= e && s < keys.Count)
                {
                    var ki = keys[s];
                    var items = sortedDict[ki];
                    foreach (var i in kv.Value)
                    {
                        foreach (var j in items)
                        {
                            if (i == j)
                            {
                                continue;
                            }
                            if (Math.Abs(i - j) <= k)
                            {
                                return true;
                            }
                        }
                    }
                    s++;
                }
            }
            return false;
        }

        //todo 分桶法
        #endregion


        #region 363. 矩形区域不超过 K 的最大数值和(未完成)
        //todo 
        //https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/
        #endregion

        #region 368. 最大整除子集(未完成)
        //todo 
        //https://leetcode-cn.com/problems/largest-divisible-subset/
        #endregion


        #region 897. 递增顺序搜索树
        //https://leetcode-cn.com/problems/increasing-order-search-tree/
        public TreeNode IncreasingBST(TreeNode root)
        {
            if (root == null)
            {
                return root;
            }
            var stack = new Stack<TreeNode>();
            TreeNode head = null, prev = new TreeNode(-1);
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }
                root = stack.Pop();
                if (head == null)
                {
                    head = root;
                }
                root.left = null;
                prev.right = root;
                prev = root;
                root = root.right;
            }
            return head;
        }
        #endregion

        #region 1011. 在 D 天内送达包裹的能力
        //https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/
        public int ShipWithinDays(int[] weights, int d)
        {
            int min = 0, max = 0;
            foreach (var w in weights)
            {
                max += w;
                min = Math.Max(min, w);
            }
            while (min < max)
            {
                var mid = min + (max - min) / 2;
                int days = 1, weight = 0;
                foreach (var w in weights)
                {
                    weight += w;
                    if (weight > mid)
                    {
                        weight = w;
                        days++;
                    }
                }
                if (days <= d)
                {
                    max = mid;
                }
                else
                {
                    min = mid + 1;
                }
            }
            return min;
        }
        #endregion

        #region 938. 二叉搜索树的范围和
        //https://leetcode-cn.com/problems/range-sum-of-bst/
        public int RangeSumBST(TreeNode root, int low, int high)
        {
            var sum = 0;
            void InOrder(TreeNode node)
            {
                if (node == null)
                {
                    return;
                }
                InOrder(node.left);
                if (node.val >= low && node.val <= high)
                {
                    sum += node.val;
                }
                else if (node.val > high)
                {
                    return;
                }
                InOrder(node.right);
            }
            InOrder(root);
            return sum;
        }
        #endregion

        #region 633. 平方数之和
        //https://leetcode-cn.com/problems/sum-of-square-numbers/
        public bool JudgeSquareSum(int c)
        {
            long l = 0, r = (long)Math.Sqrt(c);
            while (l <= r)
            {
                var n = l * l + r * r;
                if (n == c)
                {
                    return true;
                }
                if (n < c)
                {
                    l++;
                }
                else
                {
                    r--;
                }
            }
            return false;
        }
        #endregion

        #region 1720. 解码异或后的数组
        //https://leetcode-cn.com/problems/decode-xored-array/
        public int[] Decode(int[] encoded, int first)
        {
            var res = new int[encoded.Length + 1];
            res[0] = first;
            for (int i = 1; i < res.Length; i++)
            {
                res[i] = res[i - 1] ^ encoded[i - 1];
            }
            return res;
        }
        #endregion

        #region 690. 员工的重要性
        //https://leetcode-cn.com/problems/employee-importance/
        public class Employee
        {
            public int id;
            public int importance;
            public IList<int> subordinates;
        }
        public int GetImportance(IList<Employee> employees, int id)
        {
            var res = 0;
            var dict = new Dictionary<int, Employee>();
            foreach (var em in employees)
            {
                dict[em.id] = em;
            }
            var queue = new Queue<int>();
            queue.Enqueue(id);
            while (queue.TryDequeue(out id))
            {
                var em = dict[id];
                res += em.importance;
                foreach (var sid in em.subordinates)
                {
                    queue.Enqueue(sid);
                }
            }
            return res;
        }
        #endregion

        #region 1723. 完成所有工作的最短时间
        //https://leetcode-cn.com/problems/find-minimum-time-to-finish-all-jobs/
        public int MinimumTimeRequired(int[] jobs, int k)
        {
            //todo 待完成
            throw new NotImplementedException();
        }
        #endregion

        #region 872. 叶子相似的树
        //https://leetcode-cn.com/problems/leaf-similar-trees/
        public bool LeafSimilar(TreeNode root1, TreeNode root2)
        {
            void Dfs(TreeNode root, IList<int> list)
            {
                if (root == null)
                {
                    return;
                }
                if (root.left == null && root.right == null)
                {
                    list.Add(root.val);
                    return;
                }
                Dfs(root.left, list);
                Dfs(root.right, list);
            }
            IList<int> l1 = new List<int>(), l2 = new List<int>();
            Dfs(root1, l1);
            Dfs(root2, l2);
            return l1.Count == l2.Count && l1.SequenceEqual(l2);
        }
        #endregion

        #region 740. 删除并获得点数
        //https://leetcode-cn.com/problems/delete-and-earn/
        public int DeleteAndEarn(int[] nums)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region 1482. 制作 m 束花所需的最少天数
        //https://leetcode-cn.com/problems/minimum-number-of-days-to-make-m-bouquets/
        public int MinDays(int[] bloomDay, int m, int k)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region 554. 砖墙
        //https://leetcode-cn.com/problems/brick-wall/
        public int LeastBricks(IList<IList<int>> wall)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region 1473. 粉刷房子 III
        //https://leetcode-cn.com/problems/paint-house-iii/
        public int MinCost(int[] houses, int[][] cost, int m, int n, int target)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region 1310. 子数组异或查询
        //https://leetcode-cn.com/problems/xor-queries-of-a-subarray/
        public int[] XorQueries(int[] arr, int[][] queries)
        {
            var prefix = new int[arr.Length + 1];
            for (int i = 0; i < arr.Length; i++)
            {
                prefix[i + 1] = prefix[i] ^ arr[i];
            }
            var result = new int[queries.Length];
            for (int i = 0; i < queries.Length; i++)
            {
                var query = queries[i];
                result[i] = prefix[query[0]] ^ prefix[query[1] + 1];
            }
            return result;
        }
        #endregion

        #region 1734. 解码异或后的排列
        //https://leetcode-cn.com/problems/decode-xored-permutation/
        public int[] Decode(int[] encoded)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region 1269. 停在原地的方案数
        //https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/
        public int NumWays(int steps, int arrLen)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region 421. 数组中两个数的最大异或值
        //https://leetcode-cn.com/problems/maximum-xor-of-two-numbers-in-an-array/
        public int FindMaximumXOR(int[] nums)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region 993. 二叉树的堂兄弟节点
        //https://leetcode-cn.com/problems/cousins-in-binary-tree/
        public bool IsCousins(TreeNode root, int x, int y)
        {
            var queue = new Queue<TreeNode>();
            Dictionary<int, int> depthDict = new Dictionary<int, int>(), parent = new Dictionary<int, int>();
            var depth = 0;
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                for (int i = 0, len = queue.Count; i < len; i++)
                {
                    root = queue.Dequeue();
                    depthDict[root.val] = depth;
                    if (root.left != null)
                    {
                        queue.Enqueue(root.left);
                        parent[root.left.val] = root.val;
                    }
                    if (root.right != null)
                    {
                        queue.Enqueue(root.right);
                        parent[root.right.val] = root.val;
                    }
                }
                depth++;
            }
            return depthDict[x] == depthDict[y] && parent[x] != parent[y];
        }
        #endregion
    }
}