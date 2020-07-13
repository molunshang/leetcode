using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 6. Z 字形变换

        //https://leetcode-cn.com/problems/zigzag-conversion/
        public string Convert(string s, int numRows)
        {
            if (numRows <= 1)
            {
                return s;
            }

            var lines = new StringBuilder[numRows];
            for (var i = 0; i < lines.Length; i++)
            {
                lines[i] = new StringBuilder();
            }

            int row = 0, index = 0, step = 1;
            while (index < s.Length)
            {
                if (row == 0)
                {
                    step = 1;
                }
                else if (row == numRows - 1)
                {
                    step = -1;
                }

                lines[row].Append(s[index]);
                index++;
                row += step;
            }

            var result = lines[0];
            for (int i = 1; i < lines.Length; i++)
            {
                result.Append(lines[i]);
            }

            return result.ToString();
        }

        #endregion

        #region 面试题 01.05. 一次编辑

        //https://leetcode-cn.com/problems/one-away-lcci/
        bool Check(string s1, int i1, string s2, int i2)
        {
            while (i1 < s1.Length && i2 < s2.Length)
            {
                if (s1[i1] != s2[i2])
                {
                    return false;
                }

                i1++;
                i2++;
            }

            return i1 == s1.Length && i2 == s2.Length;
        }

        public bool OneEditAway(string first, string second)
        {
            var diff = Math.Abs(first.Length - second.Length);
            if (diff > 1)
            {
                return false;
            }

            int f = 0, s = 0;
            while (f < first.Length && s < second.Length)
            {
                if (first[f] == second[s])
                {
                    f++;
                    s++;
                }
                else
                {
                    //1 插入字符
                    //2 删除字符
                    //3 替换字符
                    if (first.Length == second.Length)
                    {
                        //字符串长度相等，只能替换字符
                        return Check(first, f + 1, second, s + 1);
                    }

                    if (first.Length > second.Length)
                    {
                        //first删除字符或者second插入字符
                        return Check(first, f + 1, second, s);
                    }

                    return Check(first, f, second, s + 1);
                }
            }

            return true;
        }

        #endregion

        #region 面试题 02.03. 删除中间节点

        //https://leetcode-cn.com/problems/delete-middle-node-lcci/
        public void DeleteNode(ListNode node)
        {
            ListNode next = node.next;
            node.val = next.val;
            node.next = next.next;
        }

        #endregion

        #region 面试题 04.01. 节点间通路

        //https://leetcode-cn.com/problems/route-between-nodes-lcci/
        public bool FindWhetherExistsPath(int n, int[][] graph, int start, int target)
        {
            var dict = new Dictionary<int, ISet<int>>();
            for (int i = 0; i < graph.Length; i++)
            {
                var line = graph[i];
                int p0 = line[0], p1 = line[1];
                if (!dict.ContainsKey(p0))
                {
                    dict[p0] = new HashSet<int>();
                }

                dict[p0].Add(p1);
            }

            if (!dict.ContainsKey(start) || !dict.ContainsKey(target))
            {
                return false;
            }

            var queue = new Queue<int>();
            var visited = new HashSet<int>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                var p = queue.Dequeue();
                visited.Add(p);
                if (!dict.TryGetValue(p, out var next))
                {
                    continue;
                }

                if (next.Contains(target))
                {
                    return true;
                }

                foreach (var np in next)
                {
                    if (visited.Contains(np))
                    {
                        continue;
                    }

                    queue.Enqueue(np);
                }
            }

            return false;
        }

        #endregion

        #region 面试题 04.03. 特定深度节点链表

        //https://leetcode-cn.com/problems/list-of-depth-lcci/
        public ListNode[] ListOfDepth(TreeNode tree)
        {
            var result = new List<ListNode>();
            var queue = new Queue<TreeNode>();
            queue.Enqueue(tree);
            while (queue.Count > 0)
            {
                var size = queue.Count;
                ListNode node = new ListNode(-1), root = node;
                while (size > 0)
                {
                    size--;
                    tree = queue.Dequeue();
                    node.next = new ListNode(tree.val);
                    node = node.next;
                    if (tree.left != null)
                    {
                        queue.Enqueue(tree.left);
                    }

                    if (tree.right != null)
                    {
                        queue.Enqueue(tree.right);
                    }
                }

                result.Add(root.next);
            }

            return result.ToArray();
        }

        #endregion

        #region 面试题 04.06. 后继者

        //https://leetcode-cn.com/problems/successor-lcci/
        bool InorderSuccessor(TreeNode node, TreeNode p, ref TreeNode next)
        {
            if (node == null)
            {
                return false;
            }

            if (node == p || node.val == p.val)
            {
                next = node.right;
                while (next != null && next.left != null)
                {
                    next = next.left;
                }

                return true;
            }

            bool flag;
            if (node.val > p.val)
            {
                flag = InorderSuccessor(node.left, p, ref next);
            }
            else
            {
                flag = InorderSuccessor(node.right, p, ref next);
            }

            if (flag && next == null && node.val > p.val)
            {
                next = node;
            }

            return flag;
        }

        public TreeNode InorderSuccessor(TreeNode root, TreeNode p)
        {
            if (root == null || p == null)
            {
                return null;
            }

            TreeNode next = null;
            InorderSuccessor(root, p, ref next);
            return next;
        }

        #endregion

        #region 面试题 17.16. 按摩师

        //https://leetcode-cn.com/problems/the-masseuse-lcci/

        public int Massage(int[] nums)
        {
            if (nums.Length <= 1)
            {
                return nums.Length == 1 ? nums[0] : 0;
            }

            int[] dp = new int[nums.Length];
            dp[0] = nums[0];
            dp[1] = Math.Max(nums[0], nums[1]);
            for (int i = 2; i < nums.Length; i++)
            {
                dp[i] = Math.Max(dp[i - 1], dp[i - 2] + nums[i]);
            }

            return dp[dp.Length - 1];
        }

        #endregion

        #region 面试题 08.01. 三步问题

        //https://leetcode-cn.com/problems/three-steps-problem-lcci/
        public int WaysToStep(int n)
        {
            if (n <= 2)
            {
                return n;
            }

            var dp = new int[n + 1];
            dp[0] = 1;
            dp[1] = 1;
            dp[2] = 2;
            for (int i = 3; i < dp.Length; i++)
            {
                dp[i] = ((dp[i - 1] + dp[i - 2]) % 1000000007 + dp[i - 3]) % 1000000007;
            }

            return dp[n];
        }

        #endregion

        #region 392. 判断子序列

        //https://leetcode-cn.com/problems/is-subsequence/
        public bool IsSubsequence(string s, string t)
        {
            if (s.Length > t.Length)
            {
                return false;
            }

            int si = 0, ti = 0;
            while (si < s.Length && ti < t.Length)
            {
                if (s[si] == t[ti])
                {
                    si++;
                }

                ti++;
            }

            return si >= s.Length;
        }

        #endregion

        #region 1025. 除数博弈

        //https://leetcode-cn.com/problems/divisor-game/
        public bool DivisorGame(int n, Dictionary<int, bool> dp)
        {
            if (dp.TryGetValue(n, out var flag))
            {
                return flag;
            }

            flag = false;
            for (int i = 1; i < n; i++)
            {
                if (n % i != 0)
                {
                    continue;
                }

                if (!DivisorGame(n - i, dp))
                {
                    flag = true;
                    break;
                }
            }

            dp[n] = flag;
            return flag;
        }

        public bool DivisorGame(int n)
        {
            var dp = new Dictionary<int, bool>() { { 1, false }, { 2, true } };
            return DivisorGame(n, dp);
        }

        #endregion

        #region 746. 使用最小花费爬楼梯

        //https://leetcode-cn.com/problems/min-cost-climbing-stairs/
        public int MinCostClimbingStairs(int[] cost)
        {
            if (cost.Length <= 2)
            {
                return cost.Length == 2 ? Math.Min(cost[0], cost[1]) : cost[0];
            }

            var dp = new int[cost.Length + 1];
            for (int i = 2; i < dp.Length; i++)
            {
                dp[i] = Math.Min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
            }

            return dp[cost.Length];
        }

        #endregion

        #region 152. 乘积最大子数组

        //https://leetcode-cn.com/problems/maximum-product-subarray/
        public int MaxProduct(int[] nums)
        {
            if (nums.Length <= 0)
            {
                return 0;
            }

            int max = nums[0], min = nums[0], res = nums[0];
            for (int i = 1; i < nums.Length; i++)
            {
                //存在负数，可能导致相乘后max变最小，min变最大，需要维持两个结果
                int tMax = max, tMin = min;
                max = Math.Max(tMax * nums[i], Math.Max(nums[i], nums[i] * tMin));
                min = Math.Min(tMin * nums[i], Math.Min(nums[i], nums[i] * tMax));
                res = Math.Max(max, res);
            }

            return res;
        }

        #endregion

        #region 467. 环绕字符串中唯一的子字符串

        //https://leetcode-cn.com/problems/unique-substrings-in-wraparound-string/
        public int FindSubstringInWraproundString(string p)
        {
            var result = new int[26];
            var size = 0;
            for (int i = 0; i < p.Length; i++)
            {
                if (i > 0 && (p[i - 1] + 1 == p[i] || (p[i - 1] == 'z' && p[i] == 'a')))
                {
                    size++;
                }
                else
                {
                    size = 1;
                }

                result[p[i] - 'a'] = Math.Max(result[p[i] - 'a'], size);
            }

            return result.Sum();
        }

        #endregion

        #region 10. 正则表达式匹配

        //https://leetcode-cn.com/problems/regular-expression-matching/
        bool IsMatch(string s, string p, int si, int pi)
        {
            if (si >= s.Length)
            {
                if (pi >= p.Length)
                {
                    return true;
                }

                if (p[pi] == '*')
                {
                    pi++;
                }

                var count = 0;
                for (var i = pi; i < p.Length; i++)
                {
                    if (p[i] != '*')
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

            if (pi >= p.Length)
            {
                return false;
            }

            var flag = false;
            if (s[si] == p[pi] || p[pi] == '.')
            {
                flag = IsMatch(s, p, si + 1, pi + 1);
                if (!flag && (pi + 1 < p.Length && p[pi + 1] == '*'))
                {
                    //* 当作0个或1个或n个
                    flag = IsMatch(s, p, si, pi + 2) || IsMatch(s, p, si + 1, pi + 2);
                }
            }
            else
            {
                if (p[pi] == '*')
                {
                    //"aaa"
                    //"ab*a*c*a"
                    if (pi > 0 && (p[pi - 1] == '.' || s[si] == p[pi - 1]))
                    {
                        //*作为前一个字符或者断开从下一个字符开始
                        return IsMatch(s, p, si + 1, pi) || IsMatch(s, p, si + 1, pi + 1);
                    }
                }

                if (pi + 1 < p.Length && p[pi + 1] == '*')
                {
                    return IsMatch(s, p, si, pi + 2);
                }
            }

            return flag;
        }

        public bool IsMatch(string s, string p)
        {
            if (string.IsNullOrEmpty(p))
            {
                return string.IsNullOrEmpty(s);
            }

            return IsMatch(s, p, 0, 0);
        }

        #endregion

        #region 面试题 16.18. 模式匹配

        //https://leetcode-cn.com/problems/pattern-matching-lcci/
        bool PatternMatching(string pattern, string value, int i, int j, IDictionary<char, string> dict)
        {
            if (i >= value.Length)
            {
                while (j < pattern.Length)
                {
                    if (dict.TryGetValue(pattern[j], out var sub))
                    {
                        if (sub != string.Empty)
                        {
                            return false;
                        }
                    }
                    else if (dict.Values.Contains(string.Empty))
                    {
                        return false;
                    }

                    dict[pattern[j]] = string.Empty;
                    j++;
                }

                return true;
            }

            if (j >= pattern.Length)
            {
                return false;
            }

            var pchar = pattern[j];
            if (dict.TryGetValue(pchar, out var pStr))
            {
                if (i + pStr.Length > value.Length || pStr != value.Substring(i, pStr.Length))
                {
                    return false;
                }

                return PatternMatching(pattern, value, i + pStr.Length, j + 1, dict);
            }

            for (int k = 0; k <= value.Length - i; k++)
            {
                pStr = value.Substring(i, k);
                if (dict.Values.Contains(pStr))
                {
                    continue;
                }

                dict[pchar] = pStr;
                if (PatternMatching(pattern, value, i + k, j + 1, dict))
                {
                    return true;
                }
            }

            dict.Remove(pchar);
            return false;
        }

        public bool PatternMatching(string pattern, string value)
        {
            if (string.IsNullOrEmpty(pattern))
            {
                return string.IsNullOrEmpty(value);
            }

            return PatternMatching(pattern, value, 0, 0, new Dictionary<char, string>());
        }

        #endregion


        #region 面试题 08.11. 硬币

        //https://leetcode-cn.com/problems/coin-lcci/
        public int WaysToChange(int n)
        {
            if (n == 0)
            {
                return 0;
            }

            //数学分析 
            //对于n 可以由n25个25分，n10个10分，n5个5分，剩下的全是1分 组成
            //对应程序
            // int res = 0;
            // for(int n25 = 0; n25 <= n/25; n25++)  //求出最多能够有几个25分硬币
            // {
            //     int temp1 = n - n25*25;     //选择n25个25分硬币后，最多能够选择几个10分硬币
            //     for(int n10 = 0; n10 <= temp1/10; n10++)   
            //     {
            //         int temp2 = temp1 - n10*10; //选择n25个25分硬币，n10个10分硬币后，最多能够选择几个5分硬币
            //         for(int n5 = 0; n5 <= temp2/5; n5++)
            //         {
            //             res++;
            //         }
            //     }
            // }
            int ans = 0;
            for (int i = 0; i * 25 <= n; ++i)
            {
                int rest = n - i * 25;
                int n10 = rest / 10;
                int n5 = rest % 10 / 5;
                ans = (int)((ans + (long)(n10 + 1) * (n10 + n5 + 1) % 1000000007) % 1000000007);
            }

            return ans;
        }

        #endregion

        #region 30. 串联所有单词的子串

        //https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words/
        public IList<int> FindSubstring(string s, string[] words)
        {
            if (string.IsNullOrEmpty(s) || words.Length == 0)
            {
                return new int[0];
            }

            var dict = new Dictionary<string, int>();
            int len = 0, step = words[0].Length;
            foreach (var word in words)
            {
                if (dict.ContainsKey(word))
                {
                    dict[word]++;
                }
                else
                {
                    dict[word] = 1;
                }
                len += word.Length;
            }
            var flag = false;
            if (flag)
            {
                #region 解法1
                if (dict.Count == 1)
                {
                    dict.Clear();
                    dict[string.Join(string.Empty, words)] = 1;

                }
                var res = new List<int>();
                var counter = new Dictionary<string, int>();
                var visited = new HashSet<int>();
                foreach (var kv in dict)
                {
                    var word = kv.Key;
                    int start, end = s.Length - word.Length, findIndex = 0;
                    while (true)
                    {
                        word = kv.Key;
                        start = s.IndexOf(word, findIndex);
                        if (start < 0)
                        {
                            break;
                        }
                        if (!visited.Add(start))
                        {
                            findIndex = start + word.Length;
                            continue;
                        }
                        findIndex = start + 1;
                        var index = start;
                        do
                        {
                            if (counter.ContainsKey(word))
                            {
                                counter[word]++;
                            }
                            else
                            {
                                counter[word] = 1;
                            }

                            index += word.Length;
                            if (index > end || dict[word] < counter[word])
                            {
                                break;
                            }

                            word = s.Substring(index, word.Length);
                        } while (dict.ContainsKey(word));

                        if (counter.Count == dict.Count && counter.All(ckv => dict[ckv.Key] <= ckv.Value))
                        {
                            res.Add(start);
                        }

                        counter.Clear();
                    }
                    visited.Clear();
                }
                return res;
                #endregion
            }
            else
            {
                #region 解法2
                var res = new List<int>();
                var counter = new Dictionary<string, int>();
                for (int i = 0; i < s.Length - len + 1; i++)
                {
                    var tmp = s.Substring(i, len);
                    for (int j = 0; j < tmp.Length; j += step)
                    {
                        var word = tmp.Substring(j, step);
                        if (!dict.ContainsKey(word))
                        {
                            continue;
                        }
                        if (counter.ContainsKey(word))
                        {
                            counter[word]++;
                        }
                        else
                        {
                            counter[word] = 1;
                        }
                    }

                    if (counter.Count == dict.Count && counter.All(ckv => dict[ckv.Key] <= ckv.Value))
                    {
                        res.Add(i);
                    }
                    counter.Clear();
                }
                return res;
                #endregion
            }
        }

        #endregion
    }
}