using System;
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

        #region 1035. 不相交的线
        //https://leetcode-cn.com/problems/uncrossed-lines/
        public int MaxUncrossedLines(int[] nums1, int[] nums2)
        {
            var cache = new Dictionary<string, int>();
            int Recursion(int i, int j)
            {
                if (i >= nums1.Length || j >= nums2.Length)
                {
                    return 0;
                }
                var k = i + "," + j;
                if (cache.TryGetValue(k, out var count))
                {
                    return count;
                }
                if (nums1[i] == nums2[j])
                {
                    count = Recursion(i + 1, j + 1) + 1;
                }
                else
                {
                    count = Math.Max(Recursion(i + 1, j), Recursion(i, j + 1));
                }
                cache[k] = count;
                return count;
            }
            return Recursion(0, 0);
        }
        #endregion

        #region 1190. 反转每对括号间的子串
        //https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/
        public string ReverseParentheses(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return s;
            }
            var list = new List<char>();
            var tmp = new List<char>();
            foreach (var ch in s)
            {
                if (ch == ')')
                {
                    while (list.Count > 0)
                    {
                        var last = list.Count - 1;
                        var nc = list[last];
                        list.RemoveAt(list.Count - 1);
                        if (nc == '(')
                        {
                            break;
                        }
                        tmp.Add(nc);
                    }
                    list.AddRange(tmp);
                    tmp.Clear();
                }
                else
                {
                    list.Add(ch);
                }
            }
            return new string(list.ToArray());
        }
        #endregion

        #region 461. 汉明距离
        //https://leetcode-cn.com/problems/hamming-distance/
        public int HammingDistance(int x, int y)
        {
            var result = 0;
            for (int i = 0; i < 31 && (x > 0 || y > 0); i++)
            {
                if ((x & 1) != (y & 1))
                {
                    result++;
                }
                x = x >> 1;
                y = y >> 1;
            }
            return result;
        }
        #endregion

        #region 518. 零钱兑换 II
        //https://leetcode-cn.com/problems/coin-change-2/
        public int Change(int amount, int[] coins)
        {
            if (amount <= 0)
            {
                return 0;
            }
            var cache = new Dictionary<int, int>();
            int Loop(int price)
            {
                if (cache.TryGetValue(price, out var count))
                {
                    return count;
                }
                for (int j = coins.Length - 1; j >= 0; j--)
                {
                    var coin = coins[j];
                    if (price == coin)
                    {
                        count++;
                    }
                    else
                    {
                        count += Loop(price - coin);
                    }
                }
                cache[price] = count;
                return count;
            }
            Array.Sort(coins);
            return Loop(amount);
        }
        #endregion

        #region 2006. 差的绝对值为 K 的数对数目
        //https://leetcode-cn.com/problems/count-number-of-pairs-with-absolute-difference-k/
        public int CountKDifference(int[] nums, int k)
        {
            if (nums.Length < 2)
            {
                return 0;
            }
            var result = 0;
            var dict = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                var num = nums[i];
                int small = num + k, large = num - k;
                if (dict.TryGetValue(small, out var count))
                {
                    result += count;
                }
                if (dict.TryGetValue(large, out count))
                {
                    result += count;
                }
                if (!dict.TryGetValue(num, out count))
                {
                    count = 0;
                }
                dict[num] = count + 1;
            }
            return result;
        }

        #endregion

        #region 1447. 最简分数
        //https://leetcode-cn.com/problems/simplified-fractions/
        public IList<string> SimplifiedFractions(int n)
        {
            int Gcd(int a, int b)
            {
                if (a == 1 || b == 1)
                {
                    return 1;
                }
                if (a < b)
                {
                    var temp = a;
                    a = b;
                    b = temp;
                }
                var m = a % b;
                if (m == 0)
                {
                    return b;
                }
                return Gcd(b, m);
            }
            if (n == 1)
            {
                return new string[0];
            }
            var result = new List<string>();
            for (int i = 1; i < n; i++)
            {
                for (int j = i + 1; j <= n; j++)
                {
                    if (Gcd(i, j) == 1)
                    {
                        result.Add(i + "/" + j);
                    }
                }
            }
            return result;
        }
        #endregion

        #region 1984. 学生分数的最小差值
        //https://leetcode-cn.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/
        public int MinimumDifference(int[] nums, int k)
        {
            if (k <= 1)
            {
                return 0;
            }
            Array.Sort(nums);
            var result = int.MaxValue;
            for (int i = 0, j = k - 1; j < nums.Length; i++, j++)
            {
                result = Math.Min(result, nums[j] - nums[i]);
            }
            return result;
        }
        #endregion

        #region 1020. 飞地的数量
        //https://leetcode-cn.com/problems/number-of-enclaves/
        public int NumEnclaves(int[][] grid)
        {
            if (grid == null || grid.Length == 0)
            {
                return 0;
            }
            var steps = new int[][] { new[] { 1, 0 }, new[] { -1, 0 }, new[] { 0, 1 }, new[] { 0, -1 } };
            int total = 0, w = grid[0].Length, h = grid.Length;
            var queue = new Queue<(int, int)>();
            var visited = new bool[h, w];
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    if (grid[i][j] == 0 || visited[i, j]) continue;
                    queue.Enqueue((i, j));
                    var flag = true;
                    var num = 0;
                    while (queue.TryDequeue(out var point))
                    {
                        if (visited[point.Item1, point.Item2] || grid[point.Item1][point.Item2] == 0) continue;
                        num++;
                        foreach (var step in steps)
                        {
                            int x = point.Item1 + step[0], y = point.Item2 + step[1];
                            if (x < 0 || x >= h || y < 0 || y >= w)
                            {
                                flag = false;
                                continue;
                            }
                            queue.Enqueue((x, y));
                        }
                        visited[point.Item1, point.Item2] = true;
                    }
                    if (flag)
                    {
                        total += num;
                    }
                }
            }
            return total;
        }
        #endregion

        #region 1189. “气球” 的最大数量
        //https://leetcode-cn.com/problems/maximum-number-of-balloons/
        public int MaxNumberOfBalloons(string text)
        {
            const string balloon = "balloon";
            if (string.IsNullOrEmpty(text) || text.Length < balloon.Length)
            {
                return 0;
            }
            var set = balloon.ToHashSet();
            var dict = new Dictionary<char, int>();
            foreach (var ch in text)
            {
                if (!set.Contains(ch))
                {
                    continue;
                }
                dict.TryGetValue(ch, out int count);
                dict[ch] = count + 1;
            }
            if (dict.Count < set.Count)
            {
                return 0;
            }
            dict['l'] /= 2;
            dict['o'] /= 2;
            return dict.Min(kv => kv.Value);
        }
        #endregion

        #region 540. 有序数组中的单一元素
        //https://leetcode-cn.com/problems/single-element-in-a-sorted-array/
        public int SingleNonDuplicate(int[] nums)
        {
            int s = 0, e = nums.Length - 1;
            while (s < e)
            {
                var m = (s + e) / 2;
                if (nums[m] != nums[m + 1] && nums[m] != nums[m - 1])
                {
                    //1 1 2 3 3
                    return nums[m];
                }
                if ((m & 1) == 1)
                {
                    // 1 1 2 2 3 3 4
                    // 1 2 2 3 3 4 4
                    if (nums[m] == nums[m + 1])
                    {
                        e = m - 1;
                    }
                    else
                    {
                        s = m + 1;
                    }
                }
                else
                {
                    //1 1 2 2 3
                    //1 2 2 3 3
                    if (nums[m] == nums[m + 1])
                    {
                        s = m + 2;
                    }
                    else
                    {
                        e = m - 2;
                    }
                }
            }
            return nums[s];
        }
        #endregion

        #region 1719. 重构一棵树的方案数
        //https://leetcode-cn.com/problems/number-of-ways-to-reconstruct-a-tree/
        //复制结果，没想明白
        public int CheckWays(int[][] pairs)
        {
            var adj = new Dictionary<int, ISet<int>>();
            foreach (int[] p in pairs)
            {
                if (!adj.TryGetValue(p[0], out var s0))
                {
                    s0 = new HashSet<int>();
                    adj[p[0]] = s0;
                }
                if (!adj.TryGetValue(p[1], out var s1))
                {
                    s1 = new HashSet<int>();
                    adj[p[1]] = s1;
                }
                s0.Add(p[1]);
                s1.Add(p[0]);
            }
            /* 检测是否存在根节点*/
            int root = -1;
            foreach (KeyValuePair<int, ISet<int>> pair in adj)
            {
                int node = pair.Key;
                ISet<int> neighbours = pair.Value;
                if (neighbours.Count == adj.Count - 1)
                {
                    root = node;
                }
            }
            if (root == -1)
            {
                return 0;
            }

            int res = 1;
            foreach (KeyValuePair<int, ISet<int>> pair in adj)
            {
                int node = pair.Key;
                ISet<int> neighbours = pair.Value;
                /* 如果当前节点为根节点 */
                if (node == root)
                {
                    continue;
                }
                int currDegree = neighbours.Count;
                int parent = -1;
                int parentDegree = int.MaxValue;

                /* 根据 degree 的大小找到 node 的父节点 parent */
                foreach (int neighbour in neighbours)
                {
                    if (adj[neighbour].Count < parentDegree && adj[neighbour].Count >= currDegree)
                    {
                        parent = neighbour;
                        parentDegree = adj[neighbour].Count;
                    }
                }
                if (parent == -1)
                {
                    return 0;
                }

                /* 检测父节点的集合是否包含所有的孩子节点 */
                foreach (int neighbour in neighbours)
                {
                    if (neighbour == parent)
                    {
                        continue;
                    }
                    if (!adj[parent].Contains(neighbour))
                    {
                        return 0;
                    }
                }
                if (parentDegree == currDegree)
                {
                    res = 2;
                }
            }
            return res;
        }
        #endregion

        #region 688. 骑士在棋盘上的概率
        //https://leetcode-cn.com/problems/knight-probability-in-chessboard/
        public double KnightProbability(int n, int k, int row, int column)
        {
            var cache = new double?[n, n, k + 1];
            var steps = new[] {
                new int[] { -2, -1 }, new int[] { -2, 1 }, new int[] { 2, -1 }, new int[] { 2, 1 },
                new int[] { -1, -2 }, new int[] { -1, 2 }, new int[] { 1, -2 }, new int[] { 1, 2 } };
            double Dfs(int x, int y, int s)
            {
                if (x < 0 || x >= n || y < 0 || y >= n)
                {
                    return 0;
                }
                if (s <= 0)
                {
                    return 1;
                }
                var result = cache[x, y, s];
                if (result.HasValue)
                {
                    return result.Value;
                }
                result = 0d;
                foreach (var point in steps)
                {
                    result += Dfs(x + point[0], y + point[1], s - 1);
                }
                result /= 8.0d;
                cache[x, y, s] = result;
                return result.Value;
            }
            return Dfs(row, column, k);
        }
        #endregion

        #region 1791. 找出星型图的中心节点
        //https://leetcode-cn.com/problems/find-center-of-star-graph/
        public int FindCenter(int[][] edges)
        {
            return edges[0][0] == edges[1][0] || edges[0][0] == edges[1][1] ? edges[0][0] : edges[0][1];
        }
        #endregion

        #region 969. 煎饼排序
        //https://leetcode-cn.com/problems/pancake-sorting/
        public IList<int> PancakeSort(int[] arr)
        {
            IList<int> ret = new List<int>();
            for (int n = arr.Length; n > 1; n--)
            {
                int index = 0;
                for (int i = 1; i < n; i++)
                {
                    if (arr[i] >= arr[index])
                    {
                        index = i;
                    }
                }
                if (index == n - 1)
                {
                    continue;
                }
                Array.Reverse(arr, 0, index + 1);
                Array.Reverse(arr, 0, n);
                ret.Add(index + 1);
                ret.Add(n);
            }
            return ret;
        }

        #endregion

        #region 717. 1比特与2比特字符
        //https://leetcode-cn.com/problems/1-bit-and-2-bit-characters/
        public bool IsOneBitCharacter(int[] bits)
        {
            int index = 0, last = bits.Length - 1;
            while (index < last)
            {
                if (bits[index] == 0)
                {
                    index++;
                }
                else if (bits[index] == 1)
                {
                    index += 2;
                }
            }
            return index == last;
        }
        #endregion

        #region 838. 推多米诺
        //https://leetcode-cn.com/problems/push-dominoes/
        public string PushDominoes(string dominoes)
        {
            char[] s = dominoes.ToCharArray();
            int n = s.Length, i = 0;
            char left = 'L';
            while (i < n)
            {
                int j = i;
                while (j < n && s[j] == '.')
                { // 找到一段连续的没有被推动的骨牌
                    j++;
                }
                char right = j < n ? s[j] : 'R';
                if (left == right)
                { // 方向相同，那么这些竖立骨牌也会倒向同一方向
                    while (i < j)
                    {
                        s[i++] = right;
                    }
                }
                else if (left == 'R' && right == 'L')
                { // 方向相对，那么就从两侧向中间倒
                    int k = j - 1;
                    while (i < k)
                    {
                        s[i++] = 'R';
                        s[k--] = 'L';
                    }
                }
                left = right;
                i = j + 1;
            }
            return new string(s);
        }
        #endregion

        #region 1994. 好子集的数目
        //https://leetcode-cn.com/problems/the-number-of-good-subsets/
        //TODO  copy代码，后面详细了解
        static int[] PRIMES = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };
        static int NUM_MAX = 30;
        static int MOD = 1000000007;

        public int NumberOfGoodSubsets(int[] nums)
        {
            int[] freq = new int[NUM_MAX + 1];
            foreach (int num in nums)
            {
                ++freq[num];
            }

            int[] f = new int[1 << PRIMES.Length];
            f[0] = 1;
            for (int i = 0; i < freq[1]; ++i)
            {
                f[0] = f[0] * 2 % MOD;
            }

            for (int i = 2; i <= NUM_MAX; ++i)
            {
                if (freq[i] == 0)
                {
                    continue;
                }

                // 检查 i 的每个质因数是否均不超过 1 个
                int subset = 0, x = i;
                bool check = true;
                for (int j = 0; j < PRIMES.Length; ++j)
                {
                    int prime = PRIMES[j];
                    if (x % (prime * prime) == 0)
                    {
                        check = false;
                        break;
                    }
                    if (x % prime == 0)
                    {
                        subset |= (1 << j);
                    }
                }
                if (!check)
                {
                    continue;
                }

                // 动态规划
                for (int mask = (1 << PRIMES.Length) - 1; mask > 0; --mask)
                {
                    if ((mask & subset) == subset)
                    {
                        f[mask] = (int)((f[mask] + ((long)f[mask ^ subset]) * freq[i]) % MOD);
                    }
                }
            }

            int ans = 0;
            for (int mask = 1, maskMax = (1 << PRIMES.Length); mask < maskMax; ++mask)
            {
                ans = (ans + f[mask]) % MOD;
            }

            return ans;
        }

        #endregion

        #region 917. 仅仅反转字母
        //https://leetcode-cn.com/problems/reverse-only-letters/
        public string ReverseOnlyLetters(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return s;
            }
            var chars = s.ToCharArray();
            int begin = 0, end = s.Length - 1;
            while (begin < end)
            {
                while (begin < end && !char.IsLetter(chars[begin]))
                {
                    begin++;
                }
                while (begin < end && !char.IsLetter(chars[end]))
                {
                    end--;
                }
                if (begin < end)
                {
                    var tmp = chars[begin];
                    chars[begin] = chars[end];
                    chars[end] = tmp;
                    begin++;
                    end--;
                }
            }
            return new string(chars);
        }
        #endregion

        #region 1706. 球会落何处
        //https://leetcode-cn.com/problems/where-will-the-ball-fall/
        public int[] FindBall(int[][] grid)
        {
            var ans = new int[grid[0].Length];
            int Dfs(int x, int y)
            {
                if (y >= grid[0].Length || y < 0)
                {
                    return -1;
                }
                if (x >= grid.Length)
                {
                    return y;
                }
                if (grid[x][y] == 1)
                {

                    return y < grid[0].Length - 1 && grid[x][y] == grid[x][y + 1] ? Dfs(x + 1, y + 1) : -1;
                }
                else
                {
                    return y > 0 && grid[x][y] == grid[x][y - 1] ? Dfs(x + 1, y - 1) : -1;
                }
            }
            for (int i = 0; i < ans.Length; i++)
            {
                ans[i] = Dfs(0, i);
            }
            return ans;
        }

        public int[] FindBallByLeetCode(int[][] grid)
        {
            var ans = new int[grid[0].Length];
            for (int i = 0; i < ans.Length; i++)
            {
                var col = i;
                foreach (var arr in grid)
                {
                    var dir = arr[col];
                    col += dir;
                    if (col < 0 || col >= arr.Length || arr[col] != dir)
                    {
                        col = -1;
                        break;
                    }
                }
                ans[i] = col;
            }
            return ans;
        }
        #endregion
    }
}