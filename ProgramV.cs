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
        #region 834. 树中距离之和

        //https://leetcode-cn.com/problems/sum-of-distances-in-tree/
        public int[] SumOfDistancesInTree(int N, int[][] edges)
        {
            #region 暴力解

            int[] Force()
            {
                var graph = new List<int>[N];
                var cache = new int[N, N];
                var visited = new bool[N];
                foreach (var edge in edges)
                {
                    int n1 = edge[0], n2 = edge[1];
                    if (graph[n1] == null)
                    {
                        graph[n1] = new List<int>();
                    }

                    graph[n1].Add(n2);

                    if (graph[n2] == null)
                    {
                        graph[n2] = new List<int>();
                    }

                    graph[n2].Add(n1);
                }

                int Distance(int start, int end)
                {
                    if (start == end)
                    {
                        return 0;
                    }

                    if (cache[start, end] != 0)
                    {
                        return cache[start, end];
                    }

                    var distance = -1;
                    var next = graph[start];
                    visited[start] = true;
                    foreach (var n in next.Where(n => !visited[n]))
                    {
                        distance = Distance(n, end);
                        if (distance == -1)
                        {
                            continue;
                        }

                        distance++;
                        break;
                    }

                    visited[start] = false;
                    if (distance != -1)
                    {
                        cache[start, end] = cache[end, start] = distance;
                    }

                    return distance;
                }

                var result = new int[N];
                for (int i = 0; i < result.Length; i++)
                {
                    var sum = 0;
                    for (int j = 0; j < N; j++)
                    {
                        sum += Distance(i, j);
                    }

                    result[i] = sum;
                }

                return result;
            }

            #endregion

            var treeGraph = new List<int>[N];
            for (var i = 0; i < treeGraph.Length; i++)
            {
                treeGraph[i] = new List<int>();
            }

            foreach (var edge in edges)
            {
                int n1 = edge[0], n2 = edge[1];
                treeGraph[n1].Add(n2);
                treeGraph[n2].Add(n1);
            }

            int[] distanceSum = new int[N], childNum = new int[N];
            Array.Fill(childNum, 1);

            void PostDfs(int root, int parent)
            {
                foreach (var n in treeGraph[root])
                {
                    if (n == parent)
                    {
                        continue;
                    }

                    PostDfs(n, root);
                    childNum[root] += childNum[n];
                    distanceSum[root] = distanceSum[root] + childNum[n] + distanceSum[n];
                }
            }

            void PreDfs(int root, int parent)
            {
                foreach (var n in treeGraph[root])
                {
                    if (n == parent)
                    {
                        continue;
                    }

                    distanceSum[n] = distanceSum[root] - childNum[n] + (N - childNum[n]);
                    PreDfs(n, root);
                }
            }

            PostDfs(0, -1);
            PreDfs(0, -1);
            return distanceSum;
        }

        #endregion

        #region 面试题 17.12. BiNode

        //https://leetcode-cn.com/problems/binode-lcci/
        public TreeNode ConvertBiNode(TreeNode root)
        {
            if (root == null)
            {
                return null;
            }

            var stack = new Stack<TreeNode>();
            var result = new TreeNode(-1);
            var head = result;
            while (stack.Count > 0 || root != null)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                root.left = null;
                head.right = root;
                head = head.right;
                root = root.right;
            }

            return result.right;
        }

        #endregion

        #region 508. 出现次数最多的子树元素和

        //https://leetcode-cn.com/problems/most-frequent-subtree-sum/
        public int[] FindFrequentTreeSum(TreeNode root)
        {
            var counter = new Dictionary<int, int>();
            var max = 0;

            int TreeSum(TreeNode node)
            {
                if (node == null)
                {
                    return 0;
                }

                var sum = node.val + TreeSum(node.left) + TreeSum(node.right);
                if (counter.TryGetValue(sum, out var count))
                {
                    count++;
                }
                else
                {
                    count = 1;
                }

                max = Math.Max(count, max);
                counter[sum] = count;
                return sum;
            }

            TreeSum(root);
            return counter.Where(kv => kv.Value == max).Select(kv => kv.Key).ToArray();
        }

        #endregion

        #region 589. N叉树的前序遍历

        //https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/
        public IList<int> Preorder(Node root)
        {
            var result = new List<int>();

            void Dfs(Node node)
            {
                if (node == null)
                {
                    return;
                }

                result.Add(node.val);
                if (node.children != null && node.children.Count > 0)
                {
                    foreach (var child in node.children)
                    {
                        Dfs(child);
                    }
                }
            }

            Dfs(root);
            return result;
        }

        #endregion

        #region 590. N叉树的后序遍历

        //https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/
        public IList<int> Postorder(Node root)
        {
            var result = new List<int>();

            void Dfs(Node node)
            {
                if (node == null)
                {
                    return;
                }

                if (node.children != null && node.children.Count > 0)
                {
                    foreach (var child in node.children)
                    {
                        Dfs(child);
                    }
                }

                result.Add(node.val);
            }

            Dfs(root);
            return result;
        }

        #endregion

        #region 513. 找树左下角的值

        //https://leetcode-cn.com/problems/find-bottom-left-tree-value/
        public int FindBottomLeftValue(TreeNode root)
        {
            var res = root.val;
            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                res = queue.Peek().val;
                for (int i = 0, j = queue.Count; i < j; i++)
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
                }
            }

            return res;
        }

        #endregion

        #region 645. 错误的集合

        //https://leetcode-cn.com/problems/set-mismatch/
        public int[] FindErrorNums(int[] nums)
        {
            var set = new HashSet<int>();
            int duplicate = 0, miss = 0;
            foreach (var num in nums)
            {
                if (!set.Add(num))
                {
                    duplicate = num;
                }
            }

            for (var i = 1; i <= nums.Length; i++)
            {
                if (set.Contains(i))
                    continue;
                miss = i;
                break;
            }

            return new[] {duplicate, miss};
        }

        #endregion

        #region 416. 分割等和子集

        //https://leetcode-cn.com/problems/partition-equal-subset-sum/

        public bool CanPartition(int[] nums)
        {
            if (nums.Length <= 1)
            {
                return false;
            }

            int sum = 0, max = int.MinValue;
            foreach (var num in nums)
            {
                sum += num;
                max = Math.Max(max, num);
            }

            if (sum % 2 == 1)
            {
                return false;
            }

            var target = sum / 2;
            if (target < max)
            {
                return false;
            }

            //回溯
            var cache = new Dictionary<string, bool>();

            bool Dfs(int i, int prev)
            {
                if (prev <= 0 || i >= nums.Length)
                {
                    return prev == 0;
                }

                var key = i + "," + prev;
                if (cache.TryGetValue(key, out var res))
                {
                    return res;
                }

                res = Dfs(i + 1, prev) || Dfs(i + 1, prev - nums[i]);
                cache[key] = res;
                return res;
            }

            //动态规划
            bool Dp()
            {
                var flag = new bool[nums.Length, target + 1];
                //任意区间不选择数字，结果是0
                for (int i = 0; i < nums.Length; i++)
                {
                    flag[i, 0] = true;
                }

                //只能选择1个，结果是nums[0]
                flag[0, nums[0]] = true;
                for (int i = 1; i <= target; i++)
                {
                    for (int j = 1; j < nums.Length; j++)
                    {
                        if (nums[j] > i) //不能选择
                        {
                            flag[j, i] = flag[j - 1, i];
                        }
                        else
                        {
                            flag[j, i] = flag[j - 1, i] || flag[j - 1, i - nums[j]];
                        }
                    }
                }

                return flag[nums.Length - 1, target];
            }

            return Dfs(0, target);
        }

        #endregion

        #region 530. 二叉搜索树的最小绝对差

        //https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/
        public int GetMinimumDifference(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            int ans = int.MaxValue, prev = -1;
            var stack = new Stack<TreeNode>();
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                if (prev < 0)
                {
                    prev = root.val;
                }
                else
                {
                    ans = Math.Min(ans, root.val - prev);
                    prev = root.val;
                }

                root = root.right;
            }

            return ans;
        }

        #endregion

        #region 532. 数组中的 k-diff 数对

        //https://leetcode-cn.com/problems/k-diff-pairs-in-an-array/
        class ArrayEqualityComparer : IEqualityComparer<int[]>
        {
            public bool Equals(int[] x, int[] y)
            {
                return x[0] == y[0] && x[1] == y[1];
            }

            public int GetHashCode(int[] obj)
            {
                return obj[0] << 16 | obj[1];
            }
        }


        public int FindPairs(int[] nums, int k)
        {
            var set = new HashSet<int>();
            var numSet = new HashSet<int[]>(new ArrayEqualityComparer());
            foreach (var num in nums)
            {
                int t1 = num + k, t2 = num - k;
                if (set.Contains(t1))
                {
                    numSet.Add(new[] {num, t1});
                }

                if (set.Contains(t2))
                {
                    numSet.Add(new[] {t2, num});
                }

                set.Add(num);
            }

            return numSet.Count;
        }

        #endregion

        #region 652. 寻找重复的子树

        //https://leetcode-cn.com/problems/find-duplicate-subtrees/
        public IList<TreeNode> FindDuplicateSubtrees(TreeNode root)
        {
            var res = new List<TreeNode>();
            if (root == null)
            {
                return res;
            }

            var cache = new Dictionary<string, int>();

            string Dfs(TreeNode node)
            {
                if (node == null)
                {
                    return "#";
                }

                var left = Dfs(node.left);
                var right = Dfs(node.right);
                var tree = left + "," + right + "," + node.val;
                if (cache.TryGetValue(tree, out var count) && count == 1)
                {
                    res.Add(node);
                }

                cache[tree] = count + 1;
                return tree;
            }

            Dfs(root);
            return res;
        }

        #endregion

        #region 606. 根据二叉树创建字符串

        //https://leetcode-cn.com/problems/construct-string-from-binary-tree/
        public string Tree2str(TreeNode t)
        {
            var sub = new StringBuilder();

            void Dfs(TreeNode node)
            {
                if (node == null)
                {
                    return;
                }

                sub.Append(node.val);
                if (node.left != null || node.right != null)
                {
                    sub.Append('(');
                    Dfs(node.left);
                    sub.Append(')');
                }

                if (node.right != null)
                {
                    sub.Append('(');
                    Dfs(node.right);
                    sub.Append(')');
                }
            }

            Dfs(t);
            return sub.ToString();
        }

        #endregion

        #region 115. 不同的子序列

        //https://leetcode-cn.com/problems/distinct-subsequences/
        public int NumDistinct(string s, string t)
        {
            var cache = new int?[s.Length, t.Length];

            int Dfs(int i, int j)
            {
                if (j >= t.Length)
                {
                    return 1;
                }

                if (i >= s.Length)
                {
                    return 0;
                }

                if (cache[i, j].HasValue)
                {
                    return cache[i, j].Value;
                }

                var count = 0;
                if (s[i] == t[j])
                {
                    count += Dfs(i + 1, j + 1);
                }

                count += Dfs(i + 1, j);
                cache[i, j] = count;
                return count;
            }

            return Dfs(0, 0);
        }

        #endregion

        #region 403. 青蛙过河

        //https://leetcode-cn.com/problems/frog-jump/
        public bool CanCross(int[] stones)
        {
            var cache = new Dictionary<string, bool>();

            bool Can(int i, int k)
            {
                if (i == stones.Length - 1)
                {
                    return true;
                }

                var key = i + "," + k;
                if (cache.TryGetValue(key, out var flag))
                {
                    return flag;
                }

                for (int j = i + 1; j < stones.Length; j++)
                {
                    var gap = stones[j] - stones[i];
                    if (gap >= k - 1 && gap <= k + 1)
                    {
                        if (Can(j, gap))
                        {
                            flag = true;
                            break;
                        }
                    }
                }

                cache[key] = flag;
                return flag;
            }

            return Can(0, 0);
        }

        #endregion

        #region 856. 括号的分数

        //https://leetcode-cn.com/problems/score-of-parentheses/
        public int ScoreOfParentheses(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return 0;
            }

            var stack = new Stack<int>();
            foreach (var ch in s)
            {
                if (ch == '(')
                {
                    stack.Push(0);
                }
                else
                {
                    var top = stack.Pop();
                    if (top == 0)
                    {
                        stack.Push(1);
                    }
                    else
                    {
                        var num = top;
                        while (stack.TryPop(out top) && top != 0)
                        {
                            num += top;
                        }

                        stack.Push(num * 2);
                    }
                }
            }

            var result = 0;
            while (stack.TryPop(out var n))
            {
                result += n;
            }

            return result;
        }

        #endregion

        #region 377. 组合总和 Ⅳ

        //https://leetcode-cn.com/problems/combination-sum-iv/
        public int CombinationSum4(int[] nums, int target)
        {
            var cache = new Dictionary<int, int>();

            int Helper(int num)
            {
                if (num <= 0)
                {
                    return num == 0 ? 1 : 0;
                }

                if (cache.TryGetValue(num, out var count))
                {
                    return count;
                }

                for (int i = 0; i < nums.Length; i++)
                {
                    count += Helper(num - nums[i]);
                }

                cache[num] = count;
                return count;
            }

            return Helper(target);
        }

        #endregion

        #region 1475. 商品折扣后的最终价格

        //https://leetcode-cn.com/problems/final-prices-with-a-special-discount-in-a-shop/
        public int[] FinalPrices(int[] prices)
        {
            var vals = new int[prices.Length];

            //暴力解
            void Force()
            {
                for (int i = 0; i < prices.Length; i++)
                {
                    var price = prices[i];
                    for (int j = i + 1; j < prices.Length; j++)
                    {
                        if (price < prices[j]) continue;
                        price -= prices[j];
                        break;
                    }

                    vals[i] = price;
                }
            }

            void ByStack()
            {
                var stack = new Stack<int>();
                for (int i = 0; i < prices.Length; i++)
                {
                    while (stack.TryPeek(out var j) && prices[j] >= prices[i])
                    {
                        vals[stack.Peek()] = prices[stack.Pop()] - prices[i];
                    }

                    stack.Push(i);
                }

                while (stack.TryPop(out var i))
                {
                    vals[i] = prices[i];
                }
            }

            return vals;
        }

        #endregion

        #region 914. 卡牌分组

        //https://leetcode-cn.com/problems/x-of-a-kind-in-a-deck-of-cards/
        public bool HasGroupsSizeX(int[] deck)
        {
            int Gcd(int x, int y)
            {
                while (true)
                {
                    if (x == 0)
                    {
                        return y;
                    }

                    var x1 = x;
                    x = y % x;
                    y = x1;
                }
            }

            var dict = deck.GroupBy(d => d).ToDictionary(g => g.Key, g => g.Count());
            if (dict.Count == 1)
            {
                return dict.First().Value > 1;
            }

            var gcd = dict.Aggregate(-1, (current, kv) => current == -1 ? kv.Value : Gcd(current, kv.Value));
            return gcd > 1;
        }

        #endregion

        #region 1177. 构建回文串检测

        //https://leetcode-cn.com/problems/can-make-palindrome-from-substring/
        public IList<bool> CanMakePaliQueries(string s, int[][] queries)
        {
            if (string.IsNullOrEmpty(s) || queries.Length <= 0)
            {
                return new bool[0];
            }

            var diffs = new int[s.Length][];
            for (var i = 0; i < s.Length; i++)
            {
                diffs[i] = new int[26];
                if (i != 0)
                {
                    Array.Copy(diffs[i - 1], diffs[i], 26);
                }

                diffs[i][s[i] - 'a']++;
            }


            bool Diff(int l, int r, int size)
            {
                if (r - l <= size)
                {
                    return true;
                }
                var diff = 0;
                for (int i = 0; i < 26; i++)
                {
                    var count = l == 0 ? diffs[r][i] : diffs[r][i] - diffs[l - 1][i];
                    if (count % 2 == 1)
                    {
                        diff++;
                    }
                }

                return diff <= size * 2 + 1;
            }

            var result = new bool[queries.Length];
            for (var i = 0; i < queries.Length; i++)
            {
                int l = queries[i][0], r = queries[i][1], size = queries[i][2];
                result[i] = Diff(l, r, size);
            }

            return result;
        }

        #endregion
    }
}