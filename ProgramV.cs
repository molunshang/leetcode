using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
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

            return new[] { duplicate, miss };
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
                    numSet.Add(new[] { num, t1 });
                }

                if (set.Contains(t2))
                {
                    numSet.Add(new[] { t2, num });
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

        #region 52. N皇后 II

        //https://leetcode-cn.com/problems/n-queens-ii/
        public int TotalNQueens(int n)
        {
            var count = 0;
            bool[] cols = new bool[n];
            var flags = new bool[n, n];

            bool Can(int x, int y)
            {
                //左上角
                for (int i = x - 1, j = y - 1; i >= 0 && j >= 0; i--, j--)
                {
                    if (flags[i, j])
                    {
                        return false;
                    }
                }

                //右上角
                for (int i = x - 1, j = y + 1; i >= 0 && j < n; i--, j++)
                {
                    if (flags[i, j])
                    {
                        return false;
                    }
                }

                return true;
            }

            void Dfs(int row)
            {
                if (row >= n)
                {
                    count++;
                    return;
                }

                for (int i = 0; i < n; i++)
                {
                    if (cols[i] || Can(row, i))
                    {
                        continue;
                    }

                    cols[i] = flags[row, i] = true;
                    Dfs(row + 1);
                    cols[i] = flags[row, i] = false;
                }
            }

            Dfs(0);
            return count;
        }

        #endregion

        #region 844. 比较含退格的字符串

        //https://leetcode-cn.com/problems/backspace-string-compare/
        public bool BackspaceCompare(string S, string T)
        {
            Stack<char> ss = new Stack<char>(), ts = new Stack<char>();
            for (int i = 0; i < S.Length; i++)
            {
                if (S[i] == '#')
                {
                    ss.TryPop(out _);
                }
                else
                {
                    ss.Push(S[i]);
                }
            }

            for (int i = 0; i < T.Length; i++)
            {
                if (T[i] == '#')
                {
                    ts.TryPop(out _);
                }
                else
                {
                    ts.Push(T[i]);
                }
            }

            if (ts.Count != ss.Count)
            {
                return false;
            }

            while (ts.Count > 0)
            {
                if (ts.Pop() != ss.Pop())
                {
                    return false;
                }
            }

            return true;
        }

        #endregion

        #region 310. 最小高度树

        //https://leetcode-cn.com/problems/minimum-height-trees/
        public IList<int> FindMinHeightTrees(int n, int[][] edges)
        {
            if (n == 1)
            {
                return new[] { 0 };
            }

            var graph = new Dictionary<int, List<int>>();
            var degree = new int[n];
            foreach (var edge in edges)
            {
                degree[edge[0]]++;
                degree[edge[1]]++;
                if (!graph.TryGetValue(edge[0], out var points))
                {
                    points = new List<int>();
                    graph[edge[0]] = points;
                }

                points.Add(edge[1]);
                if (!graph.TryGetValue(edge[1], out points))
                {
                    points = new List<int>();
                    graph[edge[1]] = points;
                }

                points.Add(edge[0]);
            }

            var result = new List<int>();
            var queue = new Queue<int>();
            for (var i = 0; i < n; i++)
            {
                if (degree[i] == 1)
                {
                    queue.Enqueue(i);
                }
            }

            while (queue.Count > 0)
            {
                result.Clear();
                for (int s = 0, l = queue.Count; s < l; s++)
                {
                    var start = queue.Dequeue();
                    result.Add(start);
                    if (graph.TryGetValue(start, out var points))
                    {
                        foreach (var point in points)
                        {
                            degree[point]--;
                            if (degree[point] == 1)
                            {
                                queue.Enqueue(point);
                            }
                        }
                    }
                }
            }

            return result;
        }

        #endregion

        #region 143. 重排链表

        //https://leetcode-cn.com/problems/reorder-list/
        public void ReorderList(ListNode head)
        {
            if (head == null || head.next == null)
            {
                return;
            }

            var list = new List<ListNode>();
            var node = head;
            while (node != null)
            {
                list.Add(node);
                node = node.next;
            }

            int s = 0, e = list.Count - 1;
            var prev = new ListNode(-1);
            while (s <= e)
            {
                prev.next = list[s];
                if (s > e)
                {
                    break;
                }

                prev = prev.next;
                prev.next = list[e];
                prev = prev.next;
                s++;
                e--;
            }

            prev.next = null;
        }

        #endregion

        #region 剑指 Offer 19. 正则表达式匹配

        //https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/
        public bool IsMatchOffer(string s, string p)
        {
            if (string.IsNullOrEmpty(p))
            {
                return string.IsNullOrEmpty(s);
            }

            if (string.IsNullOrEmpty(s))
            {
                return false;
            }

            bool Dfs(int si, int pi)
            {
                if (pi >= p.Length)
                {
                    return si >= s.Length;
                }

                if (si >= s.Length)
                {
                    if (p[pi] == '*')
                    {
                        pi++;
                    }

                    var count = 0;
                    for (int i = pi; i < p.Length; i++)
                    {
                        if (p[i] == '*')
                        {
                            count++;
                        }
                        else
                        {
                            count--;
                        }
                    }

                    return count == 0;
                }

                if (pi < p.Length - 1 && p[pi + 1] == '*')
                {
                    //1个或n个 s[si]
                    for (int i = si; i < s.Length; i++)
                    {
                        if (s[i] == p[pi] || p[pi] == '.')
                        {
                            if (Dfs(i + 1, pi + 2))
                            {
                                return true;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }

                    //0 个 s[si]
                    return Dfs(si, pi + 2);
                }

                return (s[si] == p[pi] || p[pi] == '.') && Dfs(si + 1, pi + 1);
            }

            return Dfs(0, 0);
        }

        #endregion

        #region 925. 长按键入

        //https://leetcode-cn.com/problems/long-pressed-name/
        public bool IsLongPressedName(string name, string typed)
        {
            bool Simple()
            {
                if (typed.Length < name.Length)
                {
                    return false;
                }

                int ni = 0, ti = 0;
                while (ti < typed.Length)
                {
                    if (ni < name.Length && name[ni] == typed[ti])
                    {
                        ni++;
                        ti++;
                    }
                    else if (ti > 0 && typed[ti] == typed[ti - 1])
                    {
                        ti++;
                    }
                    else
                    {
                        return false;
                    }
                }

                return ni == name.Length;
            }

            if (string.IsNullOrEmpty(name))
            {
                return string.IsNullOrEmpty(typed);
            }

            if (string.IsNullOrEmpty(typed) || name.Length > typed.Length)
            {
                return false;
            }

            int i = 0, j = 0;
            while (i < name.Length && j < typed.Length)
            {
                var count = 0;
                var ch = name[i];
                while (i < name.Length && name[i] == ch)
                {
                    count++;
                    i++;
                }

                while (j < typed.Length && typed[j] == ch)
                {
                    count--;
                    j++;
                }

                if (count > 0)
                {
                    return false;
                }
            }

            return i == name.Length && j == typed.Length;
        }

        #endregion

        #region 659. 分割数组为连续子序列

        //https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/
        public bool IsPossible(int[] nums)
        {
            //贪心算法：尽可能的将当前数字添加到上一个序列中
            var seq = new Dictionary<int, int>();
            var count = nums.GroupBy(n => n).ToDictionary(g => g.Key, g => g.Count());
            foreach (var num in nums)
            {
                if (count[num] == 0)
                {
                    continue;
                }

                int c, c1, c2;
                if ((c = seq.GetValueOrDefault(num - 1)) > 0)
                {
                    seq[num - 1] = c - 1;
                    if (seq.TryGetValue(num, out c))
                    {
                        seq[num] = c + 1;
                    }
                    else
                    {
                        seq[num] = 1;
                    }
                }
                else if ((c1 = count.GetValueOrDefault(num + 1)) > 0 && (c2 = count.GetValueOrDefault(num + 2)) > 0)
                {
                    count[num + 1] = c1 - 1;
                    count[num + 2] = c2 - 1;
                    if (seq.TryGetValue(num + 2, out c))
                    {
                        seq[num + 2] = c + 1;
                    }
                    else
                    {
                        seq[num + 2] = 1;
                    }
                }
                else
                {
                    return false;
                }

                count[num]--;
            }

            return true;
        }

        #endregion

        #region 763. 划分字母区间

        //https://leetcode-cn.com/problems/partition-labels/
        public IList<int> PartitionLabels(string s)
        {
            var indexs = new int[26];
            for (int i = 0; i < s.Length; i++)
            {
                var c = s[i] - 'a';
                indexs[c] = i;
            }

            var result = new List<int>();
            for (int i = 0, l = 0, r = 0; i < s.Length; i++)
            {
                var c = s[i] - 'a';
                r = Math.Max(r, indexs[c]);
                if (i == r)
                {
                    result.Add(r - l + 1);
                    l = i + 1;
                }
            }

            return result;
        }

        #endregion

        #region 57. 插入区间

        //https://leetcode-cn.com/problems/insert-interval/
        public int[][] Insert(int[][] intervals, int[] newInterval)
        {
            var result = new List<int[]>();
            var add = false;
            foreach (var interval in intervals)
            {
                if (interval[0] > newInterval[1])
                {
                    if (!add)
                    {
                        result.Add(newInterval);
                        add = true;
                    }

                    result.Add(interval);
                }
                else if (interval[1] < newInterval[0])
                {
                    result.Add(interval);
                }
                else
                {
                    newInterval[0] = Math.Min(newInterval[0], interval[0]);
                    newInterval[1] = Math.Max(newInterval[1], interval[1]);
                    if (!add)
                    {
                        result.Add(newInterval);
                        add = true;
                    }
                }
            }

            if (!add)
            {
                result.Add(newInterval);
            }

            return result.ToArray();
        }

        #endregion

        #region 722. 删除注释

        //https://leetcode-cn.com/problems/remove-comments/
        public IList<string> RemoveComments(string[] source)
        {
            var result = new List<string>();
            var line = new StringBuilder();
            var skip = false;
            foreach (var s in source)
            {
                for (var j = 0; j < s.Length; j++)
                {
                    if (!skip && s[j] == '/')
                    {
                        if (j < s.Length - 1)
                        {
                            if (s[j + 1] == '/')
                            {
                                break;
                            }

                            if (s[j + 1] == '*')
                            {
                                skip = true;
                                j++;
                            }
                        }
                    }
                    else if (skip && s[j] == '*')
                    {
                        if (j < s.Length - 1 && s[j + 1] == '/')
                        {
                            skip = false;
                            j++;
                            continue;
                        }
                    }

                    if (!skip)
                    {
                        line.Append(s[j]);
                    }
                }

                if (!skip && line.Length > 0)
                {
                    result.Add(line.ToString());
                    line.Clear();
                }
            }


            return result;
        }

        #endregion

        #region 1024. 视频拼接

        //https://leetcode-cn.com/problems/video-stitching/
        public int VideoStitching(int[][] clips, int t)
        {
            int ByBackTrack()
            {
                Array.Sort(clips,
                    Comparer<int[]>.Create((x, y) => { return x[1] == y[1] ? x[0] - y[0] : x[1] - y[1]; }));
                var cache = new int[clips.Length, t + 1];

                int Dfs(int i, int range)
                {
                    if (i <= 0)
                    {
                        return i == 0 && clips[i][0] == 0 && clips[i][1] >= range ? 1 : -1;
                    }

                    if (cache[i, range] != 0)
                    {
                        return cache[i, range];
                    }

                    var res = int.MaxValue;
                    for (int j = i; j >= 0; j--)
                    {
                        var clip = clips[j];
                        if (clip[1] < range)
                        {
                            break;
                        }

                        if (clip[0] > range)
                        {
                            continue;
                        }

                        if (clip[0] == 0)
                        {
                            res = 1;
                            break;
                        }

                        var n = Dfs(j - 1, clip[0]);
                        if (n != -1)
                        {
                            res = Math.Min(res, n + 1);
                        }
                    }

                    res = cache[i, range] = res == int.MaxValue ? -1 : res;
                    return res;
                }

                return Dfs(clips.Length - 1, t);
            }

            //动态规划
            int[] dp = new int[t + 1];
            Array.Fill(dp, int.MaxValue - 1);
            dp[0] = 0;
            for (int i = 1; i <= t; i++)
            {
                foreach (var clip in clips)
                {
                    if (clip[0] < i && i <= clip[1])
                    {
                        dp[i] = Math.Min(dp[i], dp[clip[0]] + 1);
                    }
                }
            }

            return dp[t] > clips.Length ? -1 : dp[t];
        }

        #endregion

        #region 845. 数组中的最长山脉

        //https://leetcode-cn.com/problems/longest-mountain-in-array/
        public int LongestMountain(int[] A)
        {
            var res = 0;
            bool left = false, right = false;
            for (int i = 1, j = 0; i < A.Length; i++)
            {
                if (A[i] > A[i - 1])
                {
                    left = true;
                    if (right)
                    {
                        j = i - 1;
                        right = false;
                    }
                }
                else if (A[i] == A[i - 1])
                {
                    j = i;
                    left = right = false;
                }
                else
                {
                    if (left)
                    {
                        res = Math.Max(res, i - j + 1);
                        right = true;
                    }
                    else
                    {
                        j = i;
                    }
                }
            }

            return res;
        }

        #endregion

        #region 345. 反转字符串中的元音字母

        //https://leetcode-cn.com/problems/reverse-vowels-of-a-string/
        //元音 a、e、i、o、u
        public string ReverseVowels(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return s;
            }

            var set = new HashSet<char>
            {
                'a', 'e', 'i', 'o', 'u',
                'A', 'E', 'I', 'O', 'U'
            };
            var chars = new char[s.Length];
            int l = 0, r = s.Length - 1;
            while (l < r)
            {
                if (!set.Contains(s[l]))
                {
                    chars[l] = s[l];
                    l++;
                }
                else if (!set.Contains(s[r]))
                {
                    chars[r] = s[r];
                    r--;
                }
                else
                {
                    chars[l] = s[r];
                    chars[r] = s[l];
                    l++;
                    r--;
                }
            }

            if (l == r)
            {
                chars[l] = s[l];
            }

            return new string(chars);
        }

        #endregion

        #region 383. 赎金信

        //https://leetcode-cn.com/problems/ransom-note/
        public bool CanConstruct(string ransomNote, string magazine)
        {
            if (string.IsNullOrEmpty(magazine))
            {
                return string.IsNullOrEmpty(ransomNote);
            }

            var dict = magazine.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
            foreach (var ch in ransomNote)
            {
                if (!dict.TryGetValue(ch, out var count) || count <= 0)
                {
                    return false;
                }

                dict[ch] = count - 1;
            }

            return true;
        }

        #endregion


        #region 434. 字符串中的单词数

        //https://leetcode-cn.com/problems/number-of-segments-in-a-string/
        public int CountSegments(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return 0;
            }

            var count = 0;
            var flag = true;
            foreach (var ch in s)
            {
                if (ch == ' ')
                {
                    flag = true;
                    continue;
                }

                if (flag)
                {
                    count++;
                    flag = false;
                }
            }

            return count;
        }

        #endregion

        #region 520. 检测大写字母

        //https://leetcode-cn.com/problems/detect-capital/
        public bool DetectCapitalUse(string word)
        {
            if (string.IsNullOrEmpty(word))
            {
                return true;
            }

            var count = 0;
            foreach (var ch in word)
            {
                if (ch >= 'A' && ch <= 'Z')
                {
                    count++;
                }
            }

            return word.Length == count || count == 0 || (count == 1 && 'A' <= word[0] && word[0] <= 'Z');
        }

        #endregion

        #region 521. 最长特殊序列 Ⅰ

        //https://leetcode-cn.com/problems/longest-uncommon-subsequence-i/
        public int FindLUSlength(string a, string b)
        {
            if (a.Length != b.Length)
            {
                return Math.Max(a.Length, b.Length);
            }

            return a == b ? -1 : a.Length;
        }

        #endregion

        #region 144. 二叉树的前序遍历

        //https://leetcode-cn.com/problems/binary-tree-preorder-traversal/
        public IList<int> PreorderTraversal(TreeNode root)
        {
            if (root == null)
            {
                return new int[0];
            }

            var seq = new List<int>();
            var stack = new Stack<TreeNode>();
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    seq.Add(root.val);
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                root = root.right;
            }

            return seq;
        }

        #endregion

        #region 1207. 独一无二的出现次数

        //https://leetcode-cn.com/problems/unique-number-of-occurrences/
        public bool UniqueOccurrences(int[] arr)
        {
            var bucket = new Dictionary<int, int>();
            foreach (var n in arr)
            {
                if (bucket.TryGetValue(n, out var c))
                {
                    c++;
                }
                else
                {
                    c = 1;
                }

                bucket[n] = c;
            }

            var set = new HashSet<int>();
            foreach (var count in bucket.Values)
            {
                if (count == 0)
                {
                    continue;
                }

                if (!set.Add(count))
                {
                    return false;
                }
            }

            return true;
        }

        #endregion

        #region 522. 最长特殊序列 II

        //https://leetcode-cn.com/problems/longest-uncommon-subsequence-ii/
        public int FindLUSlength(string[] strs)
        {
            bool IsSub(string parent, string child)
            {
                int i = 0, j = 0;
                while (i < parent.Length && j < child.Length)
                {
                    if (parent[i] == child[j])
                    {
                        j++;
                    }

                    i++;
                }

                return j == child.Length;
            }

            var res = -1;
            for (int i = 0; i < strs.Length; i++)
            {
                var j = 0;
                for (; j < strs.Length; j++)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    if (IsSub(strs[j], strs[i]))
                    {
                        break;
                    }
                }

                if (j == strs.Length)
                {
                    res = Math.Max(res, strs[i].Length);
                }
            }

            return res;
        }

        #endregion

        #region 682. 棒球比赛

        //https://leetcode-cn.com/problems/baseball-game/
        public int CalPoints(string[] ops)
        {
            var num = new Stack<int>();
            foreach (var op in ops)
            {
                switch (op)
                {
                    case "C":
                        num.Pop();
                        break;
                    case "D":
                        num.Push(num.Peek() * 2);
                        break;
                    case "+":
                        int n1 = num.Pop(), n2 = num.Pop();
                        num.Push(n2);
                        num.Push(n1);
                        num.Push(n1 + n2);
                        break;
                    default:
                        num.Push(int.Parse(op));
                        break;
                }
            }

            var res = 0;
            while (num.TryPop(out var n))
            {
                res += n;
            }

            return res;
        }

        #endregion

        #region 1496. 判断路径是否相交

        //https://leetcode-cn.com/problems/path-crossing/
        public bool IsPathCrossing(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                return false;
            }

            //'N'、'S'、'E' 或者 'W'
            var visited = new HashSet<string>();
            int x = 0, y = 0;
            visited.Add(x + "," + y);
            foreach (var t in path)
            {
                switch (t)
                {
                    case 'N':
                        x++;
                        break;
                    case 'S':
                        x--;
                        break;
                    case 'E':
                        y++;
                        break;
                    case 'W':
                        y--;
                        break;
                }

                if (!visited.Add(x + "," + y))
                {
                    return true;
                }
            }

            return false;
        }

        #endregion

        #region 931. 下降路径最小和

        //https://leetcode-cn.com/problems/minimum-falling-path-sum/
        public int MinFallingPathSum(int[][] a)
        {
            var dp = new int[a.Length, a[0].Length];
            var res = int.MaxValue;
            for (var i = 0; i < a.Length; i++)
            {
                for (var j = 0; j < a[i].Length; j++)
                {
                    if (i == 0)
                    {
                        dp[i, j] = a[i][j];
                    }
                    else if (j == 0)
                    {
                        dp[i, j] = (a[i].Length == 1 ? dp[i - 1, j] : Math.Min(dp[i - 1, j], dp[i - 1, j + 1])) +
                                   a[i][j];
                    }
                    else if (j == a[i].Length - 1)
                    {
                        dp[i, j] = Math.Min(dp[i - 1, j], dp[i - 1, j - 1]) + a[i][j];
                    }
                    else
                    {
                        dp[i, j] = Math.Min(Math.Min(dp[i - 1, j], dp[i - 1, j - 1]), dp[i - 1, j + 1]) + a[i][j];
                    }

                    if (i == a.Length - 1)
                    {
                        res = Math.Min(res, dp[i, j]);
                    }
                }
            }

            return res;
        }

        #endregion

        #region 919. 完全二叉树插入器

        //https://leetcode-cn.com/problems/complete-binary-tree-inserter/
        class CBTInserter
        {
            private TreeNode _root;
            private Queue<TreeNode> queue = new Queue<TreeNode>();

            public CBTInserter(TreeNode root)
            {
                _root = root;
                queue.Enqueue(root);
                while (queue.Count > 0)
                {
                    var head = queue.Peek();
                    if (head.left == null)
                    {
                        break;
                    }

                    queue.Enqueue(head.left);
                    if (head.right == null)
                    {
                        break;
                    }

                    queue.Enqueue(head.right);
                    queue.Dequeue();
                }
            }

            public int Insert(int v)
            {
                var root = queue.Peek();
                if (root.left == null)
                {
                    root.left = new TreeNode(v);
                    queue.Enqueue(root.left);
                }
                else
                {
                    root.right = new TreeNode(v);
                    queue.Enqueue(root.right);
                    queue.Dequeue();
                }

                return root.val;
            }

            public TreeNode Get_root()
            {
                return _root;
            }
        }

        #endregion

        #region 463. 岛屿的周长

        //https://leetcode-cn.com/problems/island-perimeter/
        public int IslandPerimeter(int[][] grid)
        {
            var res = 0;
            for (int i = 0; i < grid.Length; i++)
            {
                for (int j = 0; j < grid[i].Length; j++)
                {
                    if (grid[i][j] == 0)
                    {
                        continue;
                    }

                    //上
                    if (i == 0 || grid[i - 1][j] == 0)
                    {
                        res++;
                    }

                    //下
                    if (i == grid.Length - 1 || grid[i + 1][j] == 0)
                    {
                        res++;
                    }

                    //左
                    if (j == 0 || grid[i][j - 1] == 0)
                    {
                        res++;
                    }

                    //右
                    if (j == grid[i].Length - 1 || grid[i][j + 1] == 0)
                    {
                        res++;
                    }
                }
            }

            return res;
        }

        #endregion

        #region 583. 两个字符串的删除操作

        //https://leetcode-cn.com/problems/delete-operation-for-two-strings/
        public int MinDistanceDelete(string word1, string word2)
        {
            var cache = new int[word1.Length, word2.Length];

            int Dfs(int i, int j)
            {
                if (i >= word1.Length && j >= word2.Length)
                {
                    return 0;
                }

                if (i >= word1.Length)
                {
                    return word2.Length - j;
                }

                if (j >= word2.Length)
                {
                    return word1.Length - i;
                }

                if (cache[i, j] != 0)
                {
                    return cache[i, j];
                }

                var step = 0;
                if (word1[i] == word2[j])
                {
                    step = Dfs(i + 1, j + 1);
                }
                else
                {
                    step = Math.Min(Dfs(i + 1, j), Dfs(i, j + 1)) + 1;
                }

                cache[i, j] = step;
                return step;
            }

            return Dfs(0, 0);
        }

        #endregion

        #region 712. 两个字符串的最小ASCII删除和

        //https://leetcode-cn.com/problems/minimum-ascii-delete-sum-for-two-strings/
        public int MinimumDeleteSum(string s1, string s2)
        {
            var cache = new int[s1.Length + 1, s2.Length + 1];

            int Dfs(int i, int j)
            {
                if (i >= s1.Length && j >= s2.Length)
                {
                    return 0;
                }

                if (cache[i, j] != 0)
                {
                    return cache[i, j];
                }

                var step = 0;
                if (i >= s1.Length)
                {
                    while (j < s2.Length)
                    {
                        step += s2[j];
                        j++;
                    }
                }
                else if (j >= s2.Length)
                {
                    while (i < s1.Length)
                    {
                        step += s1[i];
                        i++;
                    }
                }
                else
                {
                    step = s1[i] == s2[j] ? Dfs(i + 1, j + 1) : Math.Min(Dfs(i + 1, j) + s1[i], Dfs(i, j + 1) + s2[j]);
                }

                cache[i, j] = step;
                return step;
            }

            return Dfs(0, 0);
        }

        #endregion


        #region 385. 迷你语法分析器

        //https://leetcode-cn.com/problems/mini-parser/
        public NestedInteger Deserialize(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return null;
            }


            var reader = new StringReader(s);
            var numStr = new StringBuilder();

            NestedInteger Read()
            {
                var nestedInteger = new NestedInteger();
                while (reader.Peek() > -1)
                {
                    //read int
                    var ch = (char)reader.Read();
                    switch (ch)
                    {
                        case ']':
                            return nestedInteger;
                        case ',':
                            nestedInteger.Add(Read());
                            break;
                        case '[':
                            if (']' == (char)reader.Peek())
                            {
                                reader.Read();
                                return nestedInteger;
                            }
                            nestedInteger.Add(Read());
                            break;
                        default:
                            numStr.Append(ch);
                            while (reader.Peek() > -1)
                            {
                                ch = (char)reader.Peek();
                                if (ch != '-' && !char.IsDigit(ch))
                                {
                                    break;
                                }
                                numStr.Append(ch);
                                reader.Read();
                            }
                            nestedInteger.SetInteger(int.Parse(numStr.ToString()));
                            numStr.Clear();
                            return nestedInteger;
                    }
                }

                return nestedInteger;
            }
            return Read();
        }

        #endregion

        #region 381. O(1) 时间插入、删除和获取随机元素 - 允许重复
        //https://leetcode-cn.com/problems/insert-delete-getrandom-o1-duplicates-allowed/
        public class RandomizedCollection
        {
            private List<int> data = new List<int>();
            private Dictionary<int, ISet<int>> indexDict = new Dictionary<int, ISet<int>>();
            private Random random = new Random();
            /** Initialize your data structure here. */
            public RandomizedCollection()
            {

            }

            /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
            public bool Insert(int val)
            {
                var flag = indexDict.TryGetValue(val, out var index);
                if (!flag)
                {
                    index = new HashSet<int>();
                    indexDict[val] = index;
                }
                index.Add(data.Count);
                data.Add(val);
                return !flag;
            }

            /** Removes a value from the collection. Returns true if the collection contained the specified element. */
            public bool Remove(int val)
            {
                if (!indexDict.TryGetValue(val, out var rmIndexs))
                {
                    return false;
                }
                var lastIndex = data.Count - 1;
                var last = data[lastIndex];
                var lastIndexs = indexDict[last];

                var rmIndex = rmIndexs.First();
                if (last == val)
                {
                    rmIndexs.Remove(lastIndex);
                    data.RemoveAt(lastIndex);
                    if (rmIndexs.Count <= 0)
                    {
                        indexDict.Remove(val);
                    }
                    return true;
                }
                lastIndexs.Add(rmIndex);
                lastIndexs.Remove(lastIndex);
                if (lastIndexs.Count <= 0)
                {
                    indexDict.Remove(last);
                }

                rmIndexs.Remove(rmIndex);
                if (rmIndexs.Count <= 0)
                {
                    indexDict.Remove(val);
                }

                data[rmIndex] = last;
                data.RemoveAt(lastIndex);
                return true;
            }

            /** Get a random element from the collection. */
            public int GetRandom()
            {
                return data[(int)(random.NextDouble() * data.Count)];
            }
        }
        #endregion
    }
}