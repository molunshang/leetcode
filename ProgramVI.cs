using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 452. 用最少数量的箭引爆气球

        //https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/
        public int FindMinArrowShots(int[][] points)
        {
            if (points.Length <= 0)
            {
                return 0;
            }

            //排序求相交区间
            Array.Sort(points, Comparer<int[]>.Create((a, b) =>
            {
                if (a[1] == b[1])
                {
                    return 0;
                }

                return a[1] > b[1] ? 1 : -1;
            }));
            var end = points[0][1];
            var count = 1;
            foreach (var ints in points)
            {
                if (ints[0] > end)
                {
                    end = ints[1];
                    count++;
                }
            }

            return count;
        }

        #endregion

        #region 164. 最大间距

        //https://leetcode-cn.com/problems/maximum-gap/
        public int MaximumGap(int[] nums)
        {
            if (nums.Length < 2)
            {
                return 0;
            }

            Array.Sort(nums);
            var res = 0;
            for (int i = 1; i < nums.Length; i++)
            {
                res = Math.Max(res, nums[i] - nums[i - 1]);
            }

            return res;
        }

        #endregion

        #region 493. 翻转对

        //https://leetcode-cn.com/problems/reverse-pairs/
        public int ReversePairsII(int[] nums)
        {
            if (nums.Length < 2)
            {
                return 0;
            }

            var sorted = new int[nums.Length];

            int MergeSortPair(int l, int r)
            {
                if (l >= r)
                {
                    return 0;
                }

                var m = (l + r) / 2;
                var lcount = MergeSortPair(l, m);
                var rcount = MergeSortPair(m + 1, r);
                var count = lcount + rcount;

                for (int j = l, k = m + 1; j <= m; j++)
                {
                    while (k <= r && nums[j] > 2L * nums[k])
                    {
                        k++;
                    }

                    count += k - m - 1;
                }

                int i = 0, lp = l, rp = m + 1;
                while (lp <= m && rp <= r)
                {
                    if (nums[lp] <= nums[rp])
                    {
                        sorted[i++] = nums[lp++];
                    }
                    else
                    {
                        sorted[i++] = nums[rp++];
                    }
                }

                while (lp <= m)
                {
                    sorted[i++] = nums[lp++];
                }

                while (rp <= r)
                {
                    sorted[i++] = nums[rp++];
                }

                Array.Copy(sorted, 0, nums, l, r - l + 1);
                return count;
            }

            return MergeSortPair(0, nums.Length - 1);
        }

        #endregion

        #region 面试题 10.03. 搜索旋转数组

        //https://leetcode-cn.com/problems/search-rotate-array-lcci/
        public int SearchI(int[] arr, int target)
        {
            int l = 0, r = arr.Length - 1;
            while (l <= r)
            {
                if (arr[l] == target)
                {
                    return l;
                }

                var m = (l + r) / 2;
                if (arr[m] == target)
                {
                    r = m - 1;
                }
                else if (arr[l] < arr[m])
                {
                    //[l,m]有序
                    if (arr[l] < target && target < arr[m])
                    {
                        r = m - 1;
                    }
                    else
                    {
                        l = m + 1;
                    }
                }
                else if (arr[l] > arr[m])
                {
                    //[m,r]有序
                    if (target > arr[m] && target <= arr[r])
                    {
                        l = m + 1;
                    }
                    else
                    {
                        r = m - 1;
                    }
                }
                else
                {
                    //无法判断哪部分有序
                    l++;
                }
            }

            return l >= arr.Length || arr[l] != target ? -1 : l;
        }

        #endregion

        #region 274. H 指数

        //https://leetcode-cn.com/problems/h-index/
        public int HIndex(int[] citations)
        {
            if (citations.Length <= 0)
            {
                return 0;
            }

            Array.Sort(citations);
            for (int i = 0; i < citations.Length; i++)
            {
                if (citations[i] >= citations.Length - i)
                {
                    return citations.Length - i;
                }
            }

            return citations.Length;
        }

        public int HIndexOn(int[] citations)
        {
            var refs = new int[citations.Length + 1];
            for (int i = 0; i < citations.Length; i++)
            {
                refs[Math.Min(citations[i], citations.Length)]++;
            }

            var h = citations.Length;
            for (int i = refs[h]; i < h; i += refs[h])
            {
                h--;
            }

            return h;
        }

        #endregion

        #region 321. 拼接最大数

        //https://leetcode-cn.com/problems/create-maximum-number/
        //递归+记忆化
        public int[] MaxNumberByReMem(int[] nums1, int[] nums2, int k)
        {
            int[] Max(int[] arr1, int[] arr2)
            {
                if (arr1.Length > arr2.Length)
                {
                    return arr1;
                }

                if (arr1.Length < arr2.Length)
                {
                    return arr2;
                }

                for (int i = 0; i < arr1.Length; i++)
                {
                    if (arr1[i] > arr2[i])
                    {
                        return arr1;
                    }

                    if (arr1[i] < arr2[i])
                    {
                        return arr2;
                    }
                }

                return arr1;
            }

            var cache = new Dictionary<string, int[]>();

            int[] Dfs(int i, int j, int count)
            {
                if (count <= 0)
                {
                    return new int[0];
                }

                var key = i + "," + j + "," + count;
                if (cache.TryGetValue(key, out var result))
                {
                    return result;
                }

                result = new int[0];
                for (int i1 = i; i1 < nums1.Length; i1++)
                {
                    var tmp = Dfs(i1 + 1, j, count - 1);
                    var tmp1 = new int[tmp.Length + 1];
                    tmp1[0] = nums1[i1];
                    Array.Copy(tmp, 0, tmp1, 1, tmp.Length);
                    result = Max(result, tmp1);
                }

                for (int j1 = j; j1 < nums2.Length; j1++)
                {
                    var tmp = Dfs(i, j1 + 1, count - 1);
                    var tmp2 = new int[tmp.Length + 1];
                    tmp2[0] = nums2[j1];
                    Array.Copy(tmp, 0, tmp2, 1, tmp.Length);
                    result = Max(result, tmp2);
                }

                cache[key] = result;
                return result;
            }

            return Dfs(0, 0, k);
        }

        public int[] MaxNumber(int[] nums1, int[] nums2, int k)
        {
            int[] GetMaxSeq(int[] arr, int n)
            {
                var seq = new int[n];
                int index = -1, leave = arr.Length - n;
                foreach (var num in arr)
                {
                    while (index > -1 && seq[index] < num && leave > 0) //leave表示可丢弃个数，必须取够n个数，当leave==0时必须取数，无论大小
                    {
                        index--;
                        leave--;
                    }

                    if (index < seq.Length - 1)
                    {
                        seq[++index] = num;
                    }
                    else
                    {
                        leave--;
                    }
                }

                return seq;
            }

            int Compare(int[] arr1, int[] arr2, int i1, int i2)
            {
                while (i1 < arr1.Length && i2 < arr2.Length)
                {
                    var diff = arr1[i1] - arr2[i2];
                    if (diff != 0)
                    {
                        return diff;
                    }

                    i1++;
                    i2++;
                }

                return (arr1.Length - i1) - (arr2.Length - i2);
            }

            int[] Merge(int[] arr1, int[] arr2)
            {
                if (arr1.Length <= 0)
                {
                    return arr2;
                }

                if (arr2.Length <= 0)
                {
                    return arr1;
                }

                int len = arr1.Length + arr2.Length, i1 = 0, i2 = 0;
                var sorted = new int[len];
                for (int i = 0; i < len; i++)
                {
                    if (Compare(arr1, arr2, i1, i2) >= 0)
                    {
                        sorted[i] = arr1[i1++];
                    }
                    else
                    {
                        sorted[i] = arr2[i2++];
                    }
                }

                return sorted;
            }

            var res = new int[0];
            //k=nums1取数+nums2取数 min:从nums1可取的最小个数(极端情况下nums2中所有数字全被选择)，max:从nums2中可取的最大个数(极端情况下nums1中所有数字全被选择)
            int min = Math.Max(0, k - nums2.Length), max = Math.Min(k, nums1.Length);
            for (int i = min; i <= max; i++)
            {
                var seq1 = GetMaxSeq(nums1, i);
                var seq2 = GetMaxSeq(nums2, k - i);
                var merge = Merge(seq1, seq2);
                if (Compare(merge, res, 0, 0) > 0)
                {
                    res = merge;
                }
            }

            return res;
        }

        #endregion

        #region 861. 翻转矩阵后的得分

        //https://leetcode-cn.com/problems/score-after-flipping-matrix/
        public int MatrixScore(int[][] A)
        {
            if (A.Length <= 0)
            {
                return 0;
            }

            var col = A[0].Length;
            for (int i = 0; i < col; i++)
            {
                if (i == 0)
                {
                    foreach (var ints in A)
                    {
                        if (ints[i] != 0) continue;
                        for (var j = 0; j < ints.Length; j++)
                        {
                            ints[j] = ints[j] ^ 1;
                        }
                    }
                }
                else
                {
                    var zeros = A.Count(ints => ints[i] == 0);
                    if (zeros <= (A.Length - zeros)) continue;
                    foreach (var cols in A)
                    {
                        cols[i] = cols[i] ^ 1;
                    }
                }
            }

            var score = 0;
            foreach (var ints in A)
            {
                var num = 0;
                for (int i = ints.Length - 1, j = 0; i >= 0; i--, j++)
                {
                    num |= ints[j] << i;
                }

                score += num;
            }

            return score;
        }

        #endregion

        #region 842. 将数组拆分成斐波那契序列

        //https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/
        public IList<int> SplitIntoFibonacci(string s)
        {
            var seq = new List<int>();
            if (string.IsNullOrEmpty(s) || s.Length < 3)
            {
                return seq;
            }

            bool Back(int i, int min)
            {
                if (i >= s.Length)
                {
                    return seq.Count > 2;
                }

                for (int len = seq.Count < 2 ? 1 : min, limit = s.Length - i; len <= limit; len++)
                {
                    if (s[i] == '0' && len > 1 || (i + len) > s.Length)
                    {
                        break;
                    }

                    var str = s.Substring(i, len);
                    if (!int.TryParse(str, out var n))
                    {
                        break;
                    }

                    if (seq.Count < 2 || n == seq[seq.Count - 1] + seq[seq.Count - 2])
                    {
                        seq.Add(n);
                        if (Back(i + len, len))
                        {
                            return true;
                        }

                        seq.RemoveAt(seq.Count - 1);
                    }
                    else if (n > seq[seq.Count - 1] + seq[seq.Count - 2])
                    {
                        break;
                    }
                }

                return false;
            }

            Back(0, 1);
            return seq;
        }

        #endregion

        #region 306. 累加数

        //https://leetcode-cn.com/problems/additive-number/
        public bool IsAdditiveNumber(string num)
        {
            if (string.IsNullOrEmpty(num) || num.Length < 3)
            {
                return false;
            }

            bool BackTrace(int i, long prev1, long prev2, int count)
            {
                if (i >= num.Length)
                {
                    return count > 2;
                }

                for (int l = 1, limit = num.Length - i; l <= limit; l++)
                {
                    if (l > 1 && num[i] == '0')
                    {
                        break;
                    }

                    var str = num.Substring(i, l);
                    if (!long.TryParse(str, out var cur))
                    {
                        break;
                    }

                    var sum = prev1 + prev2;
                    if (count < 2 || cur == sum)
                    {
                        if (BackTrace(i + l, prev2, cur, count + 1))
                        {
                            return true;
                        }
                    }
                    else if (cur > sum)
                    {
                        break;
                    }
                }

                return false;
            }

            return BackTrace(0, 0, 0, 0);
        }

        #endregion

        #region 823. 带因子的二叉树

        //https://leetcode-cn.com/problems/binary-trees-with-factors/
        public int NumFactoredBinaryTrees(int[] a)
        {
            var set = a.ToHashSet();
            var cache = new Dictionary<int, long>();

            long TreeCount(int root)
            {
                if (cache.TryGetValue(root, out var num))
                {
                    return num;
                }

                num = 1;
                foreach (var n in a)
                {
                    var m = root / n;
                    if (root % n != 0 || !set.Contains(m))
                    {
                        continue;
                    }

                    long left = TreeCount(n), right = TreeCount(m);
                    num = (num + left * right) % 1000000007;
                }

                cache[root] = num;
                return num;
            }

            var count = a.Sum(TreeCount);
            return (int) (count % 1000000007);
        }

        #endregion

        #region 860. 柠檬水找零

        //https://leetcode-cn.com/problems/lemonade-change/
        public bool LemonadeChange(int[] bills)
        {
            var balance = new int[3];
            foreach (var bill in bills)
            {
                switch (bill)
                {
                    case 5:
                        balance[0]++;
                        break;
                    case 10:
                        balance[1]++;
                        if (balance[0] <= 0)
                        {
                            return false;
                        }

                        balance[0]--;
                        break;
                    case 20:
                        balance[2]++;
                        var num = 3;
                        if (balance[1] > 0)
                        {
                            balance[1]--;
                            num -= 2;
                        }

                        if (balance[0] < num)
                        {
                            return false;
                        }

                        balance[0] -= num;
                        break;
                }
            }

            return true;
        }

        #endregion

        #region 649. Dota2 参议院

        //https://leetcode-cn.com/problems/dota2-senate/
        public string PredictPartyVictory(string senate)
        {
            LinkedList<int> rQueue = new LinkedList<int>(), dQueue = new LinkedList<int>();
            for (var i = 0; i < senate.Length; i++)
            {
                var ch = senate[i];
                if (ch == 'R')
                {
                    rQueue.AddLast(i);
                }
                else
                {
                    dQueue.AddLast(i);
                }
            }

            var index = 0;
            var forbid = new bool[senate.Length];
            while (rQueue.Count > 0 && dQueue.Count > 0)
            {
                index %= senate.Length;
                if (!forbid[index])
                {
                    var ch = senate[index];
                    var queue = ch == 'R' ? dQueue : rQueue;
                    var node = queue.First;
                    while (node != null)
                    {
                        if (index < node.Value)
                        {
                            forbid[node.Value] = true;
                            queue.Remove(node);
                            break;
                        }

                        if (node.Next == null)
                        {
                            node = queue.First;
                            forbid[node.Value] = true;
                            queue.Remove(node);
                            break;
                        }

                        node = node.Next;
                    }
                }

                index++;
            }

            return rQueue.Count > 0 ? "Radiant" : "Dire";
        }

        public string PredictPartyVictoryByLeetcode(string senate)
        {
            Queue<int> rQueue = new Queue<int>(), dQueue = new Queue<int>();
            for (var i = 0; i < senate.Length; i++)
            {
                var ch = senate[i];
                if (ch == 'R')
                {
                    rQueue.Enqueue(i);
                }
                else
                {
                    dQueue.Enqueue(i);
                }
            }

            while (rQueue.Count > 0 && dQueue.Count > 0)
            {
                //d要先于r投票，可以将r投出去，d可以在下一轮继续投票
                int r = rQueue.Dequeue(), d = dQueue.Dequeue();
                if (r > d)
                {
                    dQueue.Enqueue(d + senate.Length);
                }
                else
                {
                    rQueue.Enqueue(r + senate.Length);
                }
            }

            return rQueue.Count > 0 ? "Radiant" : "Dire";
        }

        #endregion

        #region 376. 摆动序列

        //https://leetcode-cn.com/problems/wiggle-subsequence/
        //根据leetcode题解解答
        public int WiggleMaxLength(int[] nums)
        {
            if (nums.Length < 3)
            {
                return nums.Length;
            }

            int up = 1, down = 1;
            for (int i = 1; i < nums.Length; i++)
            {
                if (nums[i] > nums[i - 1])
                {
                    up = Math.Max(up, down + 1);
                }
                else if (nums[i] < nums[i - 1])
                {
                    down = Math.Max(down, up + 1);
                }
            }

            return Math.Max(up, down);
        }

        #endregion

        #region 541. 反转字符串 II

        //https://leetcode-cn.com/problems/reverse-string-ii/
        public string ReverseStr(string s, int k)
        {
            var arr = s.ToCharArray();
            for (int i = 0, step = k * 2; i < arr.Length; i += step)
            {
                int l = i, r = Math.Min(i + k, arr.Length) - 1;
                while (l < r)
                {
                    var tmp = arr[l];
                    arr[l] = arr[r];
                    arr[r] = tmp;
                    l++;
                    r--;
                }
            }

            return new string(arr);
        }

        #endregion

        #region 551. 学生出勤记录 I

        //https://leetcode-cn.com/problems/student-attendance-record-i/
        public bool CheckRecord(string s)
        {
            int a = 0, l = 0;
            for (int i = 0; i < s.Length && a < 2 && l < 3; i++)
            {
                if (s[i] == 'A')
                {
                    a++;
                    l = 0;
                }
                else if (s[i] == 'L')
                {
                    l++;
                }
                else
                {
                    l = 0;
                }
            }

            return a < 2 && l < 3;
        }

        #endregion

        #region 738. 单调递增的数字

        //https://leetcode-cn.com/problems/monotone-increasing-digits/
        public int MonotoneIncreasingDigits(int n)
        {
            var nums = new List<int>();
            while (n != 0)
            {
                nums.Add(n % 10);
                n /= 10;
            }

            var seq = new List<int>();
            for (int i = nums.Count - 1; i > -1 && seq.Count < nums.Count; i--)
            {
                var last = seq.Count - 1;
                if (seq.Count <= 0 || seq[last] <= nums[i])
                {
                    seq.Add(nums[i]);
                }
                else
                {
                    //判断是否可以缩小，缩小则后面所有都填9
                    //333
                    while (seq.Count > 1 && seq[last] <= seq[last - 1])
                    {
                        seq.RemoveAt(last);
                        last--;
                    }

                    seq[last]--;
                    while (seq.Count < nums.Count)
                    {
                        seq.Add(9);
                    }
                }
            }

            return seq.Aggregate(0, (current, bit) => current * 10 + bit);
        }

        #endregion

        #region 609. 在系统中查找重复文件

        //https://leetcode-cn.com/problems/find-duplicate-file-in-system/
        public IList<IList<string>> FindDuplicate(string[] paths)
        {
            var dict = new Dictionary<string, IList<string>>();
            foreach (var path in paths)
            {
                var subs = path.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var dir = subs[0];
                for (var i = 1; i < subs.Length; i++)
                {
                    int s = subs[i].IndexOf('('), e = subs[i].IndexOf(')');
                    var content = subs[i].Substring(s + 1, e - s - 1);
                    if (!dict.TryGetValue(content, out var items))
                    {
                        dict[content] = items = new List<string>();
                    }

                    items.Add(dir + "/" + subs[i].Substring(0, s));
                }
            }

            return dict.Values.Where(it => it.Count > 1).ToArray();
        }

        #endregion

        #region 290. 单词规律

        //https://leetcode-cn.com/problems/word-pattern/
        public bool WordPattern(string pattern, string s)
        {
            if (string.IsNullOrEmpty(pattern))
            {
                return string.IsNullOrEmpty(s);
            }

            var strs = s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (strs.Length != pattern.Length)
            {
                return false;
            }

            var dict = new Dictionary<string, char>();
            var flag = new bool[26];
            for (var i = 0; i < strs.Length; i++)
            {
                var ch = pattern[i];
                var subStr = strs[i];
                if (dict.TryGetValue(subStr, out var val))
                {
                    if (ch != val)
                    {
                        return false;
                    }
                }
                else
                {
                    if (flag[ch - 'a'])
                    {
                        return false;
                    }

                    dict[subStr] = ch;
                    flag[ch - 'a'] = true;
                }
            }

            return true;
        }

        #endregion

        #region 806. 写字符串需要的行数

        //https://leetcode-cn.com/problems/number-of-lines-to-write-string/
        public int[] NumberOfLines(int[] widths, string s)
        {
            var res = new[] {1, 0};
            var leave = 100;
            foreach (var ch in s)
            {
                var w = widths[ch - 'a'];
                if (leave < w)
                {
                    res[0]++;
                    leave = 100;
                }

                leave -= w;
            }

            res[1] = 100 - leave;
            return res;
        }

        #endregion

        #region 714. 买卖股票的最佳时机含手续费

        //https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
        public int MaxProfit(int[] prices, int fee)
        {
            //超时
            var cache = new Dictionary<string, int>();

            int Profit(int i, int balance)
            {
                if (i >= prices.Length)
                {
                    return 0;
                }

                var key = i + "," + balance;
                if (cache.TryGetValue(key, out var p))
                {
                    return p;
                }

                if (balance == 0) //可以买
                {
                    p = Profit(i + 1, prices[i]);
                }
                else if (prices[i] > balance) //可以卖
                {
                    p = Math.Max(p, Profit(i + 1, 0) + (prices[i] - balance - fee));
                }

                p = Math.Max(p, Profit(i + 1, balance));
                cache[key] = p;
                return p;
            }

            return Profit(0, 0);
        }

        public int MaxProfitByLeetCodeDp(int[] prices, int fee)
        {
            int sell = 0, buy = -prices[0];
            for (int i = 1; i < prices.Length; i++)
            {
                sell = Math.Max(sell, buy + prices[i] - fee);
                buy = Math.Max(buy, sell - prices[i]);
            }

            return sell;
        }

        #endregion

        #region 389. 找不同

        //https://leetcode-cn.com/problems/find-the-difference/
        public char FindTheDifference(string s, string t)
        {
            var dict = s.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
            foreach (var ch in t)
            {
                if (!dict.TryGetValue(ch, out var count) || count == 0)
                {
                    return ch;
                }

                dict[ch] = count - 1;
            }

            return t[t.Length - 1];
        }

        #endregion

        #region 1081. 不同字符的最小子序列

        //https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters/
        public string SmallestSubsequence(string s)
        {
            var dict = new Dictionary<char, int>();
            for (int i = 0; i < s.Length; i++)
            {
                dict[s[i]] = i;
            }

            var stack = new Stack<char>();
            var set = new HashSet<char>();
            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                if (set.Contains(ch))
                {
                    continue;
                }

                while (stack.TryPeek(out var top) && top > ch && dict[top] > i)
                {
                    set.Remove(stack.Pop());
                }

                stack.Push(ch);
                set.Add(ch);
            }

            var chars = new char[stack.Count];
            for (int i = chars.Length - 1; i >= 0; i--)
            {
                chars[i] = stack.Pop();
            }

            return new string(chars);
        }

        #endregion

        #region 746. 使用最小花费爬楼梯

        //https://leetcode-cn.com/problems/min-cost-climbing-stairs/
        public int MinCostClimbingStairs(int[] cost)
        {
            int i1 = 0, i2 = 0;
            for (int i = 2; i <= cost.Length; i++)
            {
                var next = Math.Min(i1 + cost[i - 1], i2 + cost[i - 2]);
                i2 = i1;
                i1 = next;
            }

            return i1;
        }

        #endregion

        #region 135. 分发糖果

        //https://leetcode-cn.com/problems/candy/
        //暴力解超时
        public int Candy(int[] ratings)
        {
            if (ratings.Length < 2)
            {
                return ratings.Length == 1 ? 1 : 0;
            }

            var num = 0;
            var cache = new bool?[ratings.Length, ratings.Length + 1];

            bool Loop(int index, int candy)
            {
                if (index >= ratings.Length)
                {
                    return true;
                }

                if (cache[index, candy].HasValue)
                {
                    return cache[index, candy].Value;
                }

                int s, e;

                if (ratings[index] > ratings[index - 1])
                {
                    s = candy + 1;
                    e = candy + (ratings.Length - index);
                }
                else if (ratings[index] < ratings[index - 1])
                {
                    s = 1;
                    e = candy - 1;
                }
                else
                {
                    s = 1;
                    e = candy;
                    ;
                }

                for (int i = s; i <= e; i++)
                {
                    if (Loop(index + 1, i))
                    {
                        num += i;
                        cache[index, candy] = true;
                        return true;
                    }
                }

                cache[index, candy] = false;
                return false;
            }

            for (int i = 1; i <= ratings.Length; i++)
            {
                if (Loop(1, i))
                {
                    num += i;
                    break;
                }
            }

            return num;
        }

        public int CandyByLeetcode(int[] ratings)
        {
            var dp = new int[ratings.Length];
            for (var i = 0; i < ratings.Length; i++)
            {
                if (i > 0 && ratings[i] > ratings[i - 1])
                {
                    dp[i] = dp[i - 1] + 1;
                }
                else
                {
                    dp[i] = 1;
                }
            }

            int num = 0, right = 1;
            for (int i = ratings.Length - 1; i >= 0; i--)
            {
                if (i < ratings.Length - 1 && ratings[i] > ratings[i + 1])
                {
                    right++;
                }
                else
                {
                    right = 1;
                }

                num += Math.Max(dp[i], right);
            }

            return num;
        }

        #endregion

        #region 455. 分发饼干

        //https://leetcode-cn.com/problems/assign-cookies/
        public int FindContentChildren(int[] g, int[] s)
        {
            Array.Sort(g);
            Array.Sort(s);
            int num = 0, i = 0, j = 0;
            while (i < g.Length && j < s.Length)
            {
                if (s[j] >= g[i])
                {
                    num++;
                    i++;
                }

                j++;
            }

            return num;
        }

        #endregion

        #region 188. 买卖股票的最佳时机 IV

        //https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/
        public int MaxProfit(int k, int[] prices)
        {
            if (prices.Length <= 0 || k <= 0)
            {
                return 0;
            }

            //0卖 1买
            //一次买入卖出算1次完整交易，当k*2大于prices的长度时实际就无法约束交易次数
            k = Math.Min(k, prices.Length / 2);
            var dp = new int[prices.Length, k + 1, 2];
            dp[0, 0, 0] = 0;
            dp[0, 0, 1] = -prices[0];
            for (int i = 1; i <= k; i++)
            {
                dp[0, i, 0] = dp[0, i, 1] = int.MinValue / 2;
            }

            var max = 0;
            for (var i = 1; i < prices.Length; i++)
            {
                // buy[i][0] = Math.max(buy[i - 1][0], sell[i - 1][0] - prices[i]);
                dp[i, 0, 1] = Math.Max(dp[i - 1, 0, 1], dp[i - 1, 0, 0] - prices[i]);
                for (int j = 1; j <= k; j++)
                {
                    // sell[i][j] = Math.max(sell[i - 1][j], buy[i - 1][j - 1] + prices[i]);
                    dp[i, j, 0] = Math.Max(dp[i - 1, j, 0], dp[i - 1, j - 1, 1] + prices[i]);
                    // buy[i][j] = Math.max(buy[i - 1][j], sell[i - 1][j] - prices[i]);
                    dp[i, j, 1] = Math.Max(dp[i - 1, j, 1], dp[i - 1, j, 0] - prices[i]);
                    max = Math.Max(max, dp[i, j, 0]);
                }
            }

            return max;
        }

        #endregion

        #region 330. 按要求补齐数组

        //https://leetcode-cn.com/problems/patching-array/
        public int MinPatches(int[] nums, int n)
        {
            //根据leetcode题解解决
            long x = 1;
            int fill = 0, i = 0;
            while (x <= n)
            {
                if (i < nums.Length && nums[i] <= x)
                {
                    x += nums[i];
                    i++;
                }
                else
                {
                    x *= 2;
                    fill++;
                }
            }

            return fill;
        }

        #endregion

        #region 1046. 最后一块石头的重量

        //https://leetcode-cn.com/problems/last-stone-weight/
        public int LastStoneWeight(int[] stones)
        {
            var list = new List<int>(stones);
            list.Sort();
            while (list.Count > 1)
            {
                int x = list[list.Count - 1], y = list[list.Count - 2];
                list.RemoveAt(list.Count - 1);
                list.RemoveAt(list.Count - 1);
                if (x != y)
                {
                    var stone = x - y;
                    var index = list.BinarySearch(stone);
                    list.Insert(index < 0 ? ~index : index, stone);
                }
            }

            return list.Count <= 0 ? 0 : list[0];
        }

        #endregion

        #region 605. 种花问题

        //https://leetcode-cn.com/problems/can-place-flowers/
        public bool CanPlaceFlowers(int[] flowerbed, int n)
        {
            bool Dfs(int index, int count)
            {
                if (index >= flowerbed.Length)
                {
                    return count <= 0;
                }

                if (flowerbed[index] == 1)
                {
                    if (index > 0 && flowerbed[index - 1] == 1)
                    {
                        return false;
                    }

                    return count <= 0 || Dfs(index + 1, count);
                }

                if (count <= 0)
                {
                    return true;
                }

                if (index == 0 || flowerbed[index - 1] == 0)
                {
                    flowerbed[index] = 1;
                    if (Dfs(index + 1, count - 1))
                    {
                        return true;
                    }

                    flowerbed[index] = 0;
                }

                return Dfs(index + 1, count);
            }

            return Dfs(0, n);
        }

        public bool CanPlaceFlowersByLeetcode(int[] flowerbed, int n)
        {
            int count = 0;
            int m = flowerbed.Length;
            int prev = -1;
            for (int i = 0; i < m; i++)
            {
                if (flowerbed[i] == 1)
                {
                    if (prev < 0)
                    {
                        count += i / 2;
                    }
                    else
                    {
                        count += (i - prev - 2) / 2;
                    }

                    if (count >= n)
                    {
                        return true;
                    }

                    prev = i;
                }
            }

            if (prev < 0)
            {
                count += (m + 1) / 2;
            }
            else
            {
                count += (m - prev - 1) / 2;
            }

            return count >= n;
        }

        #endregion

        #region 547. 省份数量

        //https://leetcode-cn.com/problems/number-of-provinces/
        public int FindCircleNum(int[][] isConnected)
        {
            var count = 0;
            var visited = new HashSet<int>();
            var queue = new Queue<int>();
            for (var i = 0; i < isConnected.Length; i++)
            {
                if (visited.Contains(i))
                {
                    continue;
                }

                queue.Enqueue(i);
                while (queue.TryDequeue(out var j))
                {
                    if (!visited.Add(j))
                    {
                        continue;
                    }

                    for (int k = 0; k < isConnected[j].Length; k++)
                    {
                        if (isConnected[j][k] == 1)
                        {
                            queue.Enqueue(k);
                        }
                    }
                }

                count++;
            }

            return count;
        }

        #endregion

        #region 735. 行星碰撞

        //https://leetcode-cn.com/problems/asteroid-collision/
        public int[] AsteroidCollision(int[] asteroids)
        {
            var stack = new Stack<int>();
            for (var i = 0; i < asteroids.Length; i++)
            {
                var n = asteroids[i];
                while (stack.TryPeek(out var top) && n < 0 && top > 0)
                {
                    var abs = Math.Abs(n);
                    if (abs > top)
                    {
                        stack.Pop();
                    }
                    else if (abs == top)
                    {
                        stack.Pop();
                        n = 0;
                    }
                    else
                    {
                        n = 0;
                    }
                }

                if (n != 0)
                {
                    stack.Push(n);
                }
            }

            var res = new int[stack.Count];
            for (int i = res.Length - 1; i >= 0; i--)
            {
                res[i] = stack.Pop();
            }

            return res;
        }

        #endregion

        #region 1297. 子串的最大出现次数

        //https://leetcode-cn.com/problems/maximum-number-of-occurrences-of-a-substring/
        public int MaxFreq(string s, int maxLetters, int minSize, int maxSize)
        {
            var res = 0;
            var dict = new Dictionary<string, int>();
            var charSet = new HashSet<char>();
            //maxSize能够达到题目的最优解，minSize一定也能达到，因此只要考虑minSize即可
            for (int i = 0; i <= s.Length - minSize; i++)
            {
                var str = s.Substring(i, minSize);
                foreach (var ch in str)
                {
                    charSet.Add(ch);
                    if (charSet.Count > maxLetters)
                    {
                        break;
                    }
                }

                if (charSet.Count <= maxLetters)
                {
                    dict.TryGetValue(str, out var count);
                    count++;
                    dict[str] = count;
                    res = Math.Max(res, count);
                }

                charSet.Clear();
            }

            return res;
        }

        #endregion

        #region 958. 二叉树的完全性检验

        //https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/
        public bool IsCompleteTree(TreeNode root)
        {
            if (root == null)
            {
                return true;
            }

            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                root = queue.Dequeue();
                if (root == null)
                {
                    while (queue.TryDequeue(out root))
                    {
                        if (root != null)
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    queue.Enqueue(root.left);
                    queue.Enqueue(root.right);
                }
            }

            return true;
        }

        #endregion

        #region 228. 汇总区间

        //https://leetcode-cn.com/problems/summary-ranges/
        public IList<string> SummaryRanges(int[] nums)
        {
            if (nums.Length <= 0)
            {
                return new string[0];
            }

            var result = new List<string>();
            for (int i = 0, s = 0, k = 0; i <= nums.Length; i++, k++)
            {
                if ((i != nums.Length) && nums[i] == nums[s] + k)
                {
                    continue;
                }

                if (nums[s] == nums[i - 1])
                {
                    result.Add(nums[s].ToString());
                }
                else
                {
                    result.Add(nums[s] + "->" + nums[i - 1]);
                }

                k = 0;
                s = i;
            }

            return result;
        }

        #endregion

        #region 1202. 交换字符串中的元素

        //https://leetcode-cn.com/problems/smallest-string-with-swaps/
        public string SmallestStringWithSwaps(string s, IList<IList<int>> pairs)
        {
            var chars = s.ToCharArray();
            var graph = new Dictionary<int, IList<int>>();
            foreach (var pair in pairs)
            {
                if (!graph.TryGetValue(pair[0], out var next))
                {
                    graph[pair[0]] = next = new List<int>();
                }

                next.Add(pair[1]);
                if (!graph.TryGetValue(pair[1], out next))
                {
                    graph[pair[1]] = next = new List<int>();
                }

                next.Add(pair[0]);
            }

            var queue = new Queue<int>();
            var visited = new HashSet<int>();
            var paths = new List<int>();
            var seqs = new List<char>();
            foreach (var pair in pairs)
            {
                if (visited.Contains(pair[0]))
                {
                    continue;
                }

                queue.Enqueue(pair[0]);
                while (queue.TryDequeue(out var i))
                {
                    if (!visited.Add(i))
                    {
                        continue;
                    }

                    paths.Add(i);
                    seqs.Add(chars[i]);
                    var nexts = graph[i];
                    //acdb
                    foreach (var n in nexts)
                    {
                        queue.Enqueue(n);
                    }
                }

                paths.Sort();
                seqs.Sort();
                for (int i = 0; i < paths.Count; i++)
                {
                    chars[paths[i]] = seqs[i];
                }

                paths.Clear();
                seqs.Clear();
            }

            return new string(chars);
        }

        #endregion

        #region 1203. 项目管理

        //https://leetcode-cn.com/problems/sort-items-by-groups-respecting-dependencies/
        public int[] SortItems(int n, int m, int[] group, IList<IList<int>> beforeItems)
        {
            throw new NotImplementedException();
            var visited = new HashSet<int>();
            var queue = new Queue<int>();
            var seqs = new IList<int>[m + 1];
            for (int i = 0; i < n; i++)
            {
                if (visited.Contains(i))
                {
                    continue;
                }

                queue.Enqueue(i);
                while (queue.TryDequeue(out var p))
                {
                    if (!visited.Add(p))
                    {
                        continue;
                    }

                    var g = group[p] == -1 ? m : group[p];
                    var seq = seqs[g];
                    if (seq == null)
                    {
                        seqs[g] = seq = new List<int>();
                    }

                    seq.Add(p);
                    var before = beforeItems[p];
                    foreach (var b in before)
                    {
                        if (b == i)
                        {
                            return new int[0];
                        }

                        queue.Enqueue(b);
                    }
                }
            }

            var result = new int[n];
            foreach (var seq in seqs)
            {
                for (int i = seq.Count - 1; i >= 0; i--)
                {
                    result[--n] = seq[i];
                }
            }

            return result;
        }

        public int[] SortItemsByLeetCode(int n, int m, int[] group, IList<IList<int>> beforeItems)
        {
            IList<int> TopologicalSort(IList<int>[] adj, int[] inDegree, int count)
            {
                var seq = new List<int>();
                var queue = new Queue<int>();
                for (int i = 0; i < count; i++)
                {
                    if (inDegree[i] == 0)
                    {
                        queue.Enqueue(i);
                    }
                }

                while (queue.TryDequeue(out var front))
                {
                    seq.Add(front);
                    foreach (int successor in adj[front])
                    {
                        inDegree[successor]--;
                        if (inDegree[successor] == 0)
                        {
                            queue.Enqueue(successor);
                        }
                    }
                }

                if (seq.Count == count)
                {
                    return seq;
                }

                return new int[0];
            }

            // 第 1 步：数据预处理，给没有归属于一个组的项目编上组号
            for (int i = 0; i < group.Length; i++)
            {
                if (group[i] == -1)
                {
                    group[i] = m;
                    m++;
                }
            }

            // 第 2 步：实例化组和项目的邻接表
            var groupAdj = new IList<int>[m];
            var itemAdj = new IList<int>[n];
            for (int i = 0; i < m; i++)
            {
                groupAdj[i] = new List<int>();
            }

            for (int i = 0; i < n; i++)
            {
                itemAdj[i] = new List<int>();
            }

            // 第 3 步：建图和统计入度数组
            int[] groupsIndegree = new int[m];
            int[] itemsIndegree = new int[n];

            int len = group.Length;
            for (int i = 0; i < len; i++)
            {
                int currentGroup = group[i];
                foreach (int beforeItem in beforeItems[i])
                {
                    int beforeGroup = group[beforeItem];
                    if (beforeGroup != currentGroup)
                    {
                        groupAdj[beforeGroup].Add(currentGroup);
                        groupsIndegree[currentGroup]++;
                    }
                }
            }

            for (int i = 0; i < n; i++)
            {
                foreach (var item in beforeItems[i])
                {
                    itemAdj[item].Add(i);
                    itemsIndegree[i]++;
                }
            }

            // 第 4 步：得到组和项目的拓扑排序结果
            IList<int> groupsList = TopologicalSort(groupAdj, groupsIndegree, m);
            if (groupsList.Count == 0)
            {
                return new int[0];
            }

            IList<int> itemsList = TopologicalSort(itemAdj, itemsIndegree, n);
            if (itemsList.Count == 0)
            {
                return new int[0];
            }

            // 第 5 步：根据项目的拓扑排序结果，项目到组的多对一关系，建立组到项目的一对多关系
            // key：组，value：在同一组的项目列表
            var groups2Items = new Dictionary<int, IList<int>>();
            foreach (var item in itemsList)
            {
                if (!groups2Items.TryGetValue(group[item], out var items))
                {
                    groups2Items[group[item]] = items = new List<int>();
                }

                items.Add(item);
            }

            // 第 6 步：把组的拓扑排序结果替换成为项目的拓扑排序结果
            var res = new List<int>();
            foreach (var groupId in groupsList)
            {
                if (groups2Items.TryGetValue(groupId, out var items))
                {
                    res.AddRange(items);
                }
            }

            return res.ToArray();
        }

        #endregion

        #region 1018. 可被 5 整除的二进制前缀

        //https://leetcode-cn.com/problems/binary-prefix-divisible-by-5/
        public IList<bool> PrefixesDivBy5(int[] A)
        {
            var result = new bool[A.Length];
            var end = 0;
            for (int i = 0; i < A.Length; i++)
            {
                end = (end * 2 + A[i]) % 5;
                result[i] = end == 0;
            }

            return result;
        }

        #endregion

        #region 803. 打砖块

        //https://leetcode-cn.com/problems/bricks-falling-when-hit/
        public int[] HitBricks(int[][] grid, int[][] hits)
        {
            int m = grid.Length, n = grid[0].Length;

            int Dfs(int x, int y)
            {
                if (x >= m || y >= n || x < 0 || y < 0)
                {
                    return 0;
                }

                if (grid[x][y] == 1)
                {
                    grid[x][y] = 2;
                    return 1 + Dfs(x + 1, y) + Dfs(x - 1, y) + Dfs(x, y + 1) + Dfs(x, y - 1);
                }

                return 0;
            }

            bool Is(int x, int y)
            {
                if (x == 0)
                {
                    return true;
                }

                if (x + 1 < m && grid[x + 1][y] == 2)
                {
                    return true;
                }

                if (x - 1 > -1 && grid[x - 1][y] == 2)
                {
                    return true;
                }

                if (y + 1 < n && grid[x][y + 1] == 2)
                {
                    return true;
                }

                if (y - 1 > -1 && grid[x][y - 1] == 2)
                {
                    return true;
                }

                return false;
            }

            for (int i = 0; i < hits.Length; i++)
            {
                var hit = hits[i];
                int px = hit[0], py = hit[1];
                grid[px][py] -= 1;
            }

            for (int i = 0; i < n; i++)
            {
                Dfs(0, i);
            }

            var result = new int[hits.Length];
            for (int i = hits.Length - 1; i >= 0; i--)
            {
                var hit = hits[i];
                int px = hit[0], py = hit[1];
                grid[px][py] += 1;
                if (grid[px][py] == 0 || !Is(px, py))
                {
                    continue;
                }

                result[i] = Dfs(px, py) - 1;
            }

            return result;
        }

        #endregion

        #region 674. 最长连续递增序列

        //https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/
        public int FindLengthOfLCIS(int[] nums)
        {
            if (nums.Length <= 1)
            {
                return nums.Length;
            }

            int res = 1, len = 1;
            for (int i = 1; i < nums.Length; i++)
            {
                if (nums[i] > nums[i - 1])
                {
                    len++;
                }
                else
                {
                    res = Math.Max(res, len);
                    len = 1;
                }
            }

            return Math.Max(res, len);
        }

        #endregion

        #region 1584. 连接所有点的最小费用

        //https://leetcode-cn.com/problems/min-cost-to-connect-all-points/
        class Edge : IComparable<Edge>
        {
            public int X;
            public int Y;
            public int Length;

            public Edge(int x, int y, int len)
            {
                X = x;
                Y = y;
                Length = len;
            }


            public int CompareTo(Edge other)
            {
                return Length - other.Length;
            }
        }

        class DisJoinUnionSet
        {
            private int[] rank;
            private int[] set;

            public DisJoinUnionSet(int n)
            {
                rank = new int[n];
                Array.Fill(rank, 1);
                set = new int[n];
                for (int i = 0; i < n; i++)
                {
                    set[i] = i;
                }
            }

            public int Find(int n)
            {
                return set[n] == n ? n : (set[n] = Find(set[n]));
            }

            public bool Union(int x, int y)
            {
                int fx = Find(x), fy = Find(y);
                if (fx == fy)
                {
                    return false;
                }

                if (rank[fx] < rank[fy])
                {
                    var tmp = fx;
                    fx = fy;
                    fy = tmp;
                }

                rank[fx] += rank[fy];
                set[fy] = fx;
                return true;
            }
        }

        public int MinCostConnectPoints(int[][] points)
        {
            var edges = new List<Edge>();
            for (int i = 0; i < points.Length; i++)
            {
                for (int j = i + 1; j < points.Length; j++)
                {
                    edges.Add(new Edge(i, j,
                        Math.Abs(points[i][0] - points[j][0]) + Math.Abs(points[i][1] - points[j][1])));
                }
            }

            edges.Sort();
            int result = 0, n = 1;
            var unionSet = new DisJoinUnionSet(points.Length);
            for (int i = 0; i < edges.Count; i++)
            {
                var edge = edges[i];
                if (!unionSet.Union(edge.X, edge.Y))
                {
                    continue;
                }

                result += edge.Length;
                n++;
                if (n == points.Length)
                {
                    break;
                }
            }

            return result;
        }

        #endregion

        #region 628. 三个数的最大乘积

        //https://leetcode-cn.com/problems/maximum-product-of-three-numbers/
        public int MaximumProduct(int[] nums)
        {
            Array.Sort(nums);
            var last = nums.Length - 1;
            if (nums.Length == 3 || nums[last] <= 0 || nums[0] >= 0)
            {
                return nums[last] * nums[last - 1] * nums[last - 2];
            }

            return Math.Max(nums[0] * nums[1] * nums[last], nums[last] * nums[last - 1] * nums[last - 2]);
        }

        #endregion

        #region 面试题 17.07. 婴儿名字

        //https://leetcode-cn.com/problems/baby-names-lcci/
        public string[] TrulyMostPopular(string[] names, string[] synonyms)
        {
            var nameDict = new Dictionary<string, int>();
            foreach (var name in names)
            {
                var index = name.IndexOf('(');
                nameDict.Add(name.Substring(0, index), int.Parse(name.Substring(index + 1, name.Length - index - 2)));
            }

            var indexDict = new Dictionary<string, ISet<string>>();
            foreach (var synonym in synonyms)
            {
                var items = synonym.Substring(1, synonym.Length - 2).Split(',');
                if (!indexDict.TryGetValue(items[0], out var set))
                {
                    set = indexDict[items[0]] = new HashSet<string>();
                }

                set.Add(items[1]);

                if (!indexDict.TryGetValue(items[1], out set))
                {
                    set = indexDict[items[1]] = new HashSet<string>();
                }

                set.Add(items[0]);
            }

            var result = new List<string>();
            var visited = new HashSet<string>();
            var queue = new Queue<string>();
            foreach (var kv in nameDict.Where(kv => !visited.Contains(kv.Key)))
            {
                queue.Enqueue(kv.Key);
                var minName = string.Empty;
                var num = 0;
                while (queue.TryDequeue(out var name))
                {
                    if (!visited.Add(name))
                    {
                        continue;
                    }

                    if (minName == string.Empty || minName.CompareTo(name) > 0)
                    {
                        minName = name;
                    }

                    num += nameDict.GetValueOrDefault(name, 0);
                    if (indexDict.TryGetValue(name, out var others))
                    {
                        foreach (var other in others)
                        {
                            queue.Enqueue(other);
                        }
                    }
                }

                result.Add($"{minName}({num})");
            }


            return result.ToArray();
        }

        #endregion

        #region 1489. 找到最小生成树里的关键边和伪关键边

        //https://leetcode-cn.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
        //力扣解法：最小生成树+每条边遍历校验
        public IList<IList<int>> FindCriticalAndPseudoCriticalEdges(int n, int[][] edges)
        {
            var dict = new Dictionary<int[], int>();
            for (int i = 0; i < edges.Length; i++)
            {
                dict[edges[i]] = i;
            }

            Array.Sort(edges, Comparer<int[]>.Create((a, b) => a[2] - b[2]));
            var mst = new HashSet<int>();
            var uf = new UnionFind(n);
            var val = 0;
            for (var i = 0; i < edges.Length; i++)
            {
                var edge = edges[i];
                if (!uf.Union(edge[0], edge[1]))
                {
                    continue;
                }

                val += edge[2];
                mst.Add(i);
                if (mst.Count == n - 1)
                {
                    break;
                }
            }

            IList<int> highs = new List<int>(), lows = new List<int>();
            for (var i = 0; i < edges.Length; i++)
            {
                uf = new UnionFind(n);
                int curVal = 0, count = 1;
                if (!mst.Contains(i))
                {
                    uf.Union(edges[i][0], edges[i][1]);
                    curVal += edges[i][2];
                    count++;
                }

                for (int j = 0; j < edges.Length && count < n; j++)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    var edge = edges[j];
                    if (!uf.Union(edge[0], edge[1]))
                    {
                        continue;
                    }

                    curVal += edge[2];
                    count++;
                }

                if (mst.Contains(i) && (curVal > val || count < n))
                {
                    highs.Add(dict[edges[i]]);
                }
                else if (curVal == val && count == n)
                {
                    lows.Add(dict[edges[i]]);
                }
            }

            return new[] {highs, lows};
        }

        #endregion

        #region 989. 数组形式的整数加法

        //https://leetcode-cn.com/problems/add-to-array-form-of-integer/
        public IList<int> AddToArrayForm(int[] a, int k)
        {
            if (k == 0)
            {
                return a;
            }

            var stack = new Stack<int>();
            int bit = 0, index = a.Length - 1;
            while (index > -1 || k > 0)
            {
                var num = bit;
                if (index > -1)
                {
                    num += a[index--];
                }

                if (k > 0)
                {
                    num += (k % 10);
                    k /= 10;
                }

                bit = num / 10;
                num %= 10;
                stack.Push(num);
            }

            if (bit > 0)
            {
                stack.Push(bit);
            }

            var result = new int[stack.Count];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = stack.Pop();
            }

            return result;
        }

        #endregion

        #region 1319. 连通网络的操作次数

        //https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/
        public int MakeConnected(int n, int[][] connections)
        {
            var edges = n - 1;
            if (connections.Length < edges)
            {
                return -1;
            }

            var uf = new UnionFind(n);
            foreach (var con in connections)
            {
                int s = con[0], e = con[1];
                if (uf.Union(s, e))
                {
                    edges--;
                }
            }

            return edges;
        }

        #endregion

        #region 959. 由斜杠划分区域

        //https://leetcode-cn.com/problems/regions-cut-by-slashes/
        //力扣题解：每个字符位划分为4个单元格，通过并查集连接，剩余的节点数极为划分出的区域
        public int RegionsBySlashes(string[] grid)
        {
            var n = grid.Length * grid.Length * 4;
            var uf = new UnionFind(n);
            for (int i = 0, c = grid.Length; i < grid.Length; i++)
            {
                var str = grid[i];
                for (int j = 0; j < str.Length; j++)
                {
                    var ch = str[j];
                    var k = (i * c + j) * 4;
                    switch (ch)
                    {
                        case ' ':
                            uf.Union(k, k + 1);
                            uf.Union(k, k + 2);
                            uf.Union(k, k + 3);
                            break;
                        case '\\':
                            uf.Union(k + 1, k + 2);
                            uf.Union(k, k + 3);
                            break;
                        case '/':
                            uf.Union(k, k + 1);
                            uf.Union(k + 3, k + 2);
                            break;
                    }

                    if (i + 1 < grid.Length)
                    {
                        uf.Union(k + 3, ((i + 1) * c + j) * 4 + 1);
                    }

                    if (j + 1 < str.Length)
                    {
                        uf.Union(k + 2, (i * c + j + 1) * 4);
                    }
                }
            }

            return uf.Count;
        }

        #endregion

        #region 1128. 等价多米诺骨牌对的数量

        //https://leetcode-cn.com/problems/number-of-equivalent-domino-pairs/
        public int NumEquivDominoPairs(int[][] dominoes)
        {
            var pair = 0;
            var dict = new Dictionary<int, int>();
            for (int i = dominoes.Length - 1; i >= 0; i--)
            {
                var domino = dominoes[i];
                var key = domino[0] > domino[1] ? domino[0] * 10 + domino[1] : domino[1] * 10 + domino[0];
                if (dict.TryGetValue(key, out var count))
                {
                    pair += count;
                }

                dict[key] = count + 1;
            }

            return pair;
        }

        #endregion

        #region 1579. 保证图可完全遍历

        //https://leetcode-cn.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/
        public int MaxNumEdgesToRemove(int n, int[][] edges)
        {
            UnionFind ua = new UnionFind(n), ub = new UnionFind(n);
            var num = 0;
            foreach (var edge in edges)
            {
                edge[1]--;
                edge[2]--;
            }

            foreach (var edge in edges)
            {
                if (edge[0] != 3)
                {
                    continue;
                }

                if (ua.Union(edge[1], edge[2]))
                {
                    ub.Union(edge[1], edge[2]);
                }
                else
                {
                    num++;
                }
            }

            foreach (var edge in edges)
            {
                if (edge[0] == 1)
                {
                    if (!ua.Union(edge[1], edge[2]))
                    {
                        num++;
                    }
                }
                else if (edge[0] == 2)
                {
                    if (!ub.Union(edge[1], edge[2]))
                    {
                        num++;
                    }
                }
            }

            if (ua.Count != 1 || ub.Count != 1)
            {
                return -1;
            }

            return num;
        }

        #endregion

        #region 1631. 最小体力消耗路径

        //https://leetcode-cn.com/problems/path-with-minimum-effort/

        public int MinimumEffortPath(int[][] heights)
        {
            int rows = heights.Length, cols = heights[0].Length;
            var dp = new int[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    dp[i, j] = int.MaxValue;
                }
            }
            var visited = new bool[rows, cols];
            var steps = new[] {new[] {1, 0}, new[] {-1, 0}, new[] {0, 1}, new[] {0, -1}};
            var queue = new Queue<int[]>();
            queue.Enqueue(new[] {0, 0});
            dp[0, 0] = 0;
            while (queue.TryDequeue(out var p))
            {
                int x = p[0], y = p[1];
                if (visited[x, y])
                {
                    continue;
                }

                visited[x, y] = true;
                foreach (var s in steps)
                {
                    int tx = x + s[0], ty = y + s[1];
                    if (tx < 0 || tx >= rows || ty < 0 || ty >= cols)
                    {
                        continue;
                    }
                    var h = Math.Max(dp[x, y], Math.Abs(heights[x][y] - heights[tx][ty]));
                    if (h >= dp[tx, ty])
                    {
                        continue;
                    }

                    dp[tx, ty] = h;
                    queue.Enqueue(new[] {tx, ty});
                }
            }
            return dp[rows - 1, cols - 1];
        }

        #region 暴力解超时
        public int MinimumEffortPathForce(int[][] heights)
        {
            if (heights.Length <= 0)
            {
                return 0;
            }

            int rows = heights.Length, cols = heights[0].Length;
            var dp = new int[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    dp[i, j] = int.MaxValue;
                }
            }

            var visited = new bool[rows, cols];
            var steps = new[] {new[] {1, 0}, new[] {-1, 0}, new[] {0, 1}, new[] {0, -1}};

            int Dfs(int x, int y)
            {
                visited[x, y] = true;
                var path = int.MaxValue;
                if (x == rows - 1 && y == cols - 1)
                {
                    path = 0;
                }
                else
                {
                    foreach (var i in steps)
                    {
                        int tx = x + i[0], ty = y + i[1];
                        if (tx < 0 || tx >= rows || ty < 0 || ty >= cols || visited[tx, ty])
                        {
                            continue;
                        }
                        var tp = Dfs(tx, ty);
                        if (tp < 0)
                        {
                            continue;
                        }

                        path = Math.Min(path, Math.Max(tp, Math.Abs(heights[x][y] - heights[tx][ty])));
                    }
                }

                dp[x, y] = Math.Min(path, dp[x, y]);
                visited[x, y] = false;
                return dp[x, y];
            }

            Dfs(0, 0);
            return dp[0, 0];
        }

        #endregion
        #endregion
    }
}