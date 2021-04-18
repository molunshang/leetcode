using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace leetcode
{
    static class Arrays
    {
        public static void PrintArray<T>(this IEnumerable<T> array)
        {
            Console.WriteLine(string.Join(",", array));
        }

        public static void PrintArray<T>(this IEnumerable<T> array, Func<T, string> convert)
        {
            Console.WriteLine(string.Join(Environment.NewLine, array.Select(convert)));
        }
    }

    partial class Program
    {
        static Program program = new Program();
        static Solution solution = new Solution();

        static void Main(string[] args)
        {
            program.ContainsNearbyAlmostDuplicate(new[] { -1, 2147483647 }, 1, 2147483647);
            program.IsValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#");
            program.MaxSatisfied(new[] { 7, 8, 8, 6 }, new[] { 0, 1, 0, 1 }, 3);
            program.MaxTurbulenceSize(JsonConvert.DeserializeObject<int[]>("[2,0,2,4,2,5,0,1,2,3]"));
            program.MinimumEffortPath(JsonConvert.DeserializeObject<int[][]>("[[4,3,4,10,5,5,9,2],[10,8,2,10,9,7,5,6],[5,8,10,10,10,7,4,2],[5,1,3,1,1,3,1,9],[6,4,10,6,10,9,4,6]]"));
            program.FindCriticalAndPseudoCriticalEdges(5, JsonConvert.DeserializeObject<int[][]>("[[0, 1, 1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]"));
            program.PredictPartyVictory("DRRDRDRDRDDRDRDR");
            program.NumFactoredBinaryTrees(new[] { 2, 4, 5, 10 });
            program.SplitIntoFibonacci("1320581321313221264343965566089105744171833277577");
            program.MatrixScore(JsonConvert.DeserializeObject<int[][]>("[[1,0,1,1],[1,0,1,0],[1,1,0,0]]"));
            program.SearchI(
                new[] { 12, 20, -21, -21, -19, -14, -11, -8, -8, -8, -6, -6, -4, -4, 0, 1, 5, 5, 6, 11, 11, 12 }, -8);
            program.NthUglyNumber(1000000000, 2, 168079517, 403313907);
            program.GetValidT9Words("8733", new[] { "tree", "used" });
            program.MaxChunksToSorted(new[] { 1, 0, 2, 3, 4 });
            program.FindRotateSteps("godding", "godding");
            program.FurthestBuilding(new[] { 4, 2, 7, 6, 9, 14, 12 }, 5, 1);
            program.CheckSubarraySum(new[] { 0, 0 }, 0);
            program.Deserialize("[123,456,[788,799,833],[[]],10,[]]");
            program.VideoStitching(JsonConvert.DeserializeObject<int[][]>("[[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]]"),
                10);
            program.IsMatchOffer("mississippi", "mis*is*p*.");
            program.FindMinHeightTrees(6, JsonConvert.DeserializeObject<int[][]>("[[0,1],[0,2],[0,3],[3,4],[4,5]]"));
            program.CanMakePaliQueries("rkzavgdmdgt",
                JsonConvert.DeserializeObject<int[][]>(
                    "[[5,8,0],[7,9,1],[3,6,4],[5,5,1],[8,10,0],[3,9,5],[0,10,10],[6,8,3]]"));
            program.OpenLock(JsonConvert.DeserializeObject<string[]>("[\"0201\",\"0101\",\"0102\",\"1212\",\"2002\"]"),
                "0202");
            program.FindLongestSubarray(new[]
                {"A", "1", "B", "C", "D", "2", "3", "4", "E", "5", "F", "G", "6", "7", "H", "I", "J", "K", "L", "M"});
            var test = new TreeNode(0);
            test.left = new TreeNode(0);
            test.left.left = new TreeNode(0);
            test.left.right = new TreeNode(0);
            program.MinCameraCover(test);
            program.MctFromLeafValues(new[] { 6, 2, 4 });
            program.BasicCalculate("5-(4+1)");
            program.FindNumberOfLIS(new[] { 2, 2, 2, 2, 2, 3, 8, 7 });
            program.FindItinerary(
                JsonConvert.DeserializeObject<IList<IList<string>>>(
                    @"[[""JFK"",""KUL""],[""JFK"",""NRT""],[""NRT"",""JFK""]]"));
            program.MinStickers(
                new[] { "slave", "doctor", "kept", "insect", "an", "window", "she", "range", "post", "guide" },
                "supportclose");
            program.RemoveStones(JsonConvert.DeserializeObject<int[][]>("[[0,0],[0,2],[1,1],[2,0],[2,2]]"));
            program.RangeBitwiseAnd(5, 6);
            program.CountEval("0&0&0&1^1|0", 1);
            program.RemoveBoxes(new[] { 1, 3, 2, 2, 2, 3, 4, 3, 1 });
            program.IsPerfectSquare(2147395600);
            program.FindClosestElements(new[] { 1, 10, 15, 25, 35, 45, 50, 59 }, 1, 30);
            program.RestoreIpAddresses("25525511135");
            var root = new TreeNode(1);
            root.left = new TreeNode(3);
            root.left.right = new TreeNode(2);
            program.RecoverTree(root);
            root.left.left = new TreeNode(3);
            root.right = new TreeNode(5);
            root.right.left = new TreeNode(6);
            root.right.right = new TreeNode(7);
            program.Flatten(root);
            program.SmallestRange(
                JsonConvert.DeserializeObject<int[][]>("[[4, 10, 15, 24, 26],[0, 9, 12, 20],[5, 18, 22, 30]]"));
            Console.WriteLine(program.MinimalSteps(new[] { "S#O", "M..", "M.T" }));
            program.SplitArray(new[] { 7, 2, 5, 10, 8, 1 }, 2);
            program.KthGrammar(30, (int)Math.Pow(2, 29) - 1);
            program.WiggleSort(new[] { 4, 5, 5, 6 });
            program.LongestIncreasingPath(JsonConvert.DeserializeObject<int[][]>("[[7,8,9],[9,7,6],[7,2,3]]"));
            program.CalculateMinimumHP(JsonConvert.DeserializeObject<int[][]>("[[-2,-3,3],[-5,-10,1],[10,30,-5]]"));
            program.CalculateMinimumHP(
                JsonConvert.DeserializeObject<int[][]>("[[-5,-10,1],[-2,-3,3],[10,30,-5],[-5,-10,1],[10,30,-5]]"));
            program.CountSmaller(new[] { 5, 2, 6, 1, 1, 1, 2 }).PrintArray();
            program.Respace(
                new[]
                {
                    "vprkj", "sqvuzjz", "ptkrqrkussszzprkqrjrtzzvrkrrrskkrrursqdqpp", "spqzqtrqs", "rkktkruzsjkrzqq",
                    "rk", "k", "zkvdzqrzpkrukdqrqjzkrqrzzkkrr", "pzpstvqzrzprqkkkd", "jvutvjtktqvvdkzujkq", "r",
                    "pspkr", "tdkkktdsrkzpzpuzvszzzzdjj", "zk", "pqkjkzpvdpktzskdkvzjkkj", "sr",
                    "zqjkzksvkvvrsjrjkkjkztrpuzrqrqvvpkutqkrrqpzu"
                }, "rkktkruzsjkrzqqzkvdzqrzpkrukdqrqjzkrqrzzkkrr");
            program.MaxSumDivThree(new[] { 3, 6, 5, 1, 8 });
            program.Calculate("1+2*5/3+6/4*2");
            program.FindOrder(6, new[] { new[] { 1, 2 }, new[] { 3, 0 }, new[] { 4, 2 }, new[] { 3, 5 } });
            program.Solve(JsonConvert.DeserializeObject<char[][]>(
                "[['O','X','O','O','O','X'],['O','O','X','X','X','O'],['X','X','X','X','X','O'],['O','O','O','O','X','X'],['X','X','O','O','X','O'],['O','O','X','X','X','X']]"));
            Console.WriteLine(program.LeastInterval(
                JsonConvert.DeserializeObject<char[]>(
                    "['A','A','A','B','B','B','C','C','C','C','C','C']"), 3));
            program.RemoveInvalidParentheses("(a)())()");
            Console.WriteLine(program.NumSquares(12));
            program.MaximalRectangle(JsonConvert.DeserializeObject<char[][]>(
                "[['1','0','1','1','0'],['1','0','1','1','1'],['1','1','1','1','1'],['1','0','1','1','0']]"));
            program.TotalNQueens(5);
            program.MinDistance("trinitrophenylmethylnitramine", "dinitrophenylhydrazine");
            program.IsMatchI("aaabbbaabaaaaababaabaaabbabbbbbbbbaabababbabbbaaaaba", "a*******b");
            var chars = JsonConvert.DeserializeObject<char[][]>(
                "[['1','1','1','1','0'],['1','1','0','1','0'],['1','1','0','0','0'],['0','0','0','0','0']]");
            program.NumIslands(chars);
            program.FindTargetSumWays(new[] { 1, 1, 1, 1, 1 }, 3);
            program.LengthOfLIS(new[] { 10, 9, 2, 5, 3, 4 });
            program.LongestValidParentheses("(()()");
            program.FindBestValue(new[] { 4, 9, 3 }, 10);
            var node = new ListNode(1);
            node.next = new ListNode(2);
            node.next.next = new ListNode(3);
            node.next.next.next = new ListNode(4);
            node.next.next.next.next = new ListNode(5);
            program.ReverseBetween(node, 2, 4);

            program.LongestCommonSubsequence("hofubmnylkra", "pqhgxgdofcvmr");
            program.Rotate(new[]
            {
                new[] {1, 2, 3, 4, 5}, new[] {6, 7, 8, 9, 10}, new[] {11, 12, 13, 14, 15}, new[] {16, 17, 18, 19, 20},
                new[] {21, 22, 23, 24, 25}
            });
            program.MinFlips(new[] { new[] { 1, 1, 1 }, new[] { 1, 0, 1 }, new[] { 0, 0, 0 } });
            Console.WriteLine(program.WordBreakI("aaaaaaa", new[] { "aaaa", "aaa" }));


            program.GenerateMatrix(3);
            Console.WriteLine(program.Compress(new[]
                {'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'}));
            //[]
            var t = new TreeNode(2);
            t.left = new TreeNode(1);
            t.right = new TreeNode(3);
            program.IsValidBST(t);
            Console.WriteLine(program.EquationsPossible(new[]
                {"a!=i", "g==k", "k==j", "k!=i", "c!=e", "a!=e", "k!=a", "a!=g", "g!=c"}));
            Console.WriteLine(program.EquationsPossible(new[] { "b==b", "b==e", "e==c", "d!=e" }));
            Console.WriteLine(program.EquationsPossible(new[] { "a==b", "b!=c", "c==a" }));
            Console.WriteLine(program.EquationsPossible(new[] { "c==c", "b==d", "x!=z" }));

            program.FindLadders("a", "c", new[] { "a", "b", "c" });
            program.FindLadders("hit", "cog", new[] { "hot", "dot", "dog", "lot", "log", "cog" });
            program.WordBreak("pineapplepenapple", new[] { "apple", "pen", "applepen", "pine", "pineapple" });
            program.Subsets(new[] { 1, 2, 3 });
            program.Permute(new[] { 1, 2, 3 });
            program.LetterCombinations("234");
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(i + "," + program.SearchRotate(new[] { 3, 1 }, i));
            }

            Console.WriteLine(program.FindUnsortedSubarray(new[] { 1, 3, 3, 3, 2, 2, 2, 5, 8 }));
            solution.ConstructArr(new[] { 1, 2, 3, 4, 5, 10 }).PrintArray();
            //4,2,5,1,3

            var codec = new Codec();
            var str = codec.Serialize(root);
            Console.WriteLine(codec.Deserialize(str));

            var cache = new LRUCache(2);
            cache.Put(2, 1);
            cache.Put(3, 2);
            Console.WriteLine(cache.Get(3));
            Console.WriteLine(cache.Get(2));
            cache.Put(4, 3);
            Console.WriteLine(cache.Get(2));
            Console.WriteLine(cache.Get(3));
            Console.WriteLine(cache.Get(4));
            Console.WriteLine(
                new Program().ReversePairs1(new[] { 7, 5, 6, 4, 1, 10 }));
        }

        #region 面试题63. 股票的最大利润

        static int MaxProfit(int[] prices)
        {
            var max = 0;
            for (int i = prices.Length - 1; i >= 1; i--)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    max = Math.Max(prices[i] - prices[j], max);
                }
            }

            return max;
        }

        //动态规划
        public int MaxProfit1(int[] prices)
        {
            if (prices.Length <= 1)
            {
                return 0;
            }

            var minPrice = prices[0];
            var max = 0;
            for (int i = 1; i < prices.Length; i++)
            {
                minPrice = Math.Min(minPrice, prices[i]);
                max = Math.Max(max, prices[i] - minPrice);
            }

            return max;
        }

        #endregion

        #region 80. 删除排序数组中的重复项 II

        //80. 删除排序数组中的重复项 II
        //https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/
        static int RemoveDuplicatesII(int[] nums)
        {
            var res = nums.Length;
            int startIndex = 0, skipIndex = 0;
            for (; startIndex < res - 2; startIndex++)
            {
                skipIndex = 0;
                for (int j = startIndex + 2; j < res && nums[startIndex] == nums[j]; j++)
                {
                    skipIndex++;
                }

                for (int j = startIndex + 2; j < res - skipIndex; j++)
                {
                    nums[j] = nums[j + skipIndex];
                }

                res -= skipIndex;
            }

            return res;
        }


        public int RemoveDuplicates1(int[] nums)
        {
            var index = 1;
            var flag = false;
            for (int i = 1; i < nums.Length; i++)
            {
                if (nums[i] == nums[i - 1])
                {
                    if (flag)
                        continue;
                    nums[index] = nums[i];
                    flag = true;
                }
                else
                {
                    nums[index] = nums[i];
                    flag = false;
                }

                index++;
            }

            return index;
        }

        #endregion

        #region 罗马数字转整数

        //13. 罗马数字转整数
        //https://leetcode-cn.com/problems/roman-to-integer/
        public static int RomanToInt(string s)
        {
            var romanDic = new Dictionary<char, int>
            {
                {'I', 1},
                {'V', 5},
                {'X', 10},
                {'L', 50},
                {'C', 100},
                {'D', 500},
                {'M', 1000}
            };
            var set = new Dictionary<char, char[]>
            {
                {'I', new[] {'V', 'X'}},
                {'X', new[] {'L', 'C'}},
                {'C', new[] {'D', 'M'}}
            };
            var result = 0;
            for (int i = 0; i < s.Length; i++)
            {
                var c = s[i];
                if (set.TryGetValue(c, out var items))
                {
                    var nextC = i + 1;
                    if (nextC >= s.Length || !items.Contains(s[nextC]))
                    {
                        result += romanDic[c];
                    }
                    else
                    {
                        result += (romanDic[s[nextC]] - romanDic[c]);
                        i++;
                    }

                    continue;
                }

                result += romanDic[c];
            }

            return result;
        }

        #endregion

        #region 搜索插入位置

        //35. 搜索插入位置
        //https://leetcode-cn.com/problems/search-insert-position/
        public static int SearchInsert(int[] nums, int target)
        {
            int start = 0, end = nums.Length - 1, middle = (nums.Length - 1) / 2;
            while (start <= end)
            {
                if (nums[middle] == target)
                {
                    return middle;
                }

                if (nums[middle] > target)
                {
                    end = middle - 1;
                }
                else
                {
                    start = middle + 1;
                }

                middle = (start + end) / 2;
            }

            return nums[middle] > target ? middle : middle + 1;
        }

        #endregion

        #region 38. 外观数列

        //38. 外观数列
        //https://leetcode-cn.com/problems/count-and-say/
        public static string CountAndSay(int n)
        {
            if (n <= 0)
            {
                return string.Empty;
            }

            if (n == 1)
            {
                return "1";
            }

            var result = new StringBuilder();
            var str = CountAndSay(n - 1);
            var flag = str[0];
            var size = 1;
            for (int i = 1; i < str.Length; i++)
            {
                if (flag != str[i])
                {
                    result.Append(size).Append(flag);
                    flag = str[i];
                    size = 1;
                    continue;
                }

                size++;
            }

            result.Append(size).Append(flag);
            return result.ToString();
        }

        #endregion

        #region 53. 最大子序和/面试题 16.17. 连续数列

        //https://leetcode-cn.com/problems/maximum-subarray/
        //https://leetcode-cn.com/problems/contiguous-sequence-lcci/
        public static int MaxSubArray(int[] nums)
        {
            int ans = nums[0];
            int sum = 0;
            foreach (int num in nums)
            {
                if (sum > 0)
                {
                    sum += num;
                }
                else
                {
                    sum = num;
                }

                ans = Math.Max(ans, sum);
            }

            return ans;
        }

        #endregion

        #region 面试题59 - I. 滑动窗口的最大值

        //https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/
        public static int[] MaxSlidingWindow(int[] nums, int k)
        {
            if (nums.Length <= 0)
            {
                return nums;
            }

            var list = new int[nums.Length - k + 1];
            for (int i = 0, end = k - 1; end < nums.Length; i++, end++)
            {
                list[i] = nums[i];
                for (int j = i + 1; j <= end; j++)
                {
                    list[i] = Math.Max(list[i], nums[j]);
                }
            }

            return list;
        }

        #endregion


        static void Swap<T>(T[] items, int c1, int c2)
        {
            var tmp = items[c1];
            items[c1] = items[c2];
            items[c2] = tmp;
        }

        public static void HeapSort<T>(T[] items) where T : IComparable<T>
        {
            for (int i = items.Length / 2; i >= 0; i--)
            {
                BuildHeap(items, i, items.Length);
            }

            Swap(items, 0, items.Length - 1);
            for (int i = 1; i < items.Length; i++)
            {
                BuildHeap(items, 0, items.Length - i);
                Swap(items, 0, items.Length - i - 1);
            }
        }

        public static void BuildHeap<T>(T[] items, int start, int end) where T : IComparable<T>
        {
            var parent = items[start];
            int left = start * 2 + 1, right = left + 1;
            while (left < end)
            {
                if (right < end && items[left].CompareTo(items[right]) < 0)
                {
                    left++;
                }

                if (parent.CompareTo(items[left]) > 0)
                {
                    break;
                }

                items[start] = items[left];
                start = left;
                left = left * 2 + 1;
                right = left + 1;
            }

            items[start] = parent;
        }

        public static bool IsAnagram(string s, string t)
        {
            if (s.Length != t.Length)
            {
                return false;
            }

            char[] items1 = s.ToCharArray(), items2 = t.ToCharArray();
            HeapSort(items1);
            HeapSort(items2);
            for (int i = 0; i < items1.Length; i++)
            {
                if (items1[i] != items2[i])
                {
                    return false;
                }
            }

            return true;
        }

        #region 1356. 根据数字二进制下 1 的数目排序

        //https://leetcode-cn.com/problems/sort-integers-by-the-number-of-1-bits/
        public int[] SortByBits(int[] arr)
        {
            int BitCount(int n)
            {
                var count = 0;
                while (n != 0)
                {
                    n = n & (n - 1);
                    count++;
                }

                return count;
            }

            var cache = new Dictionary<int, int>();
            Array.Sort(arr, Comparer<int>.Create((a, b) =>
            {
                if (!cache.TryGetValue(a, out var ac))
                {
                    ac = BitCount(a);
                    cache[a] = ac;
                }

                if (!cache.TryGetValue(b, out var bc))
                {
                    bc = BitCount(b);
                    cache[b] = bc;
                }

                return ac == bc ? a - b : ac - bc;
            }));
            return arr;
        }

        #endregion

        #region 1122. 数组的相对排序

        //https://leetcode-cn.com/problems/relative-sort-array/
        public int[] RelativeSortArray(int[] arr1, int[] arr2)
        {
            var buckets = new int[1001];
            foreach (var num in arr1)
            {
                buckets[num]++;
            }

            var i = 0;
            for (int j = 0; j < arr2.Length; j++)
            {
                var n = arr2[j];
                for (int k = 0; k < buckets[n]; k++)
                {
                    arr1[i++] = n;
                }

                buckets[n] = 0;
            }

            for (int j = 0; j < buckets.Length; j++)
            {
                if (buckets[j] == 0)
                {
                    continue;
                }

                for (int k = 0; k < buckets[j]; k++)
                {
                    arr1[i++] = j;
                }
            }

            return arr1;
        }

        #endregion

        #region 922. 按奇偶排序数组 II

        //https://leetcode-cn.com/problems/sort-array-by-parity-ii/
        //
        public static int[] SortArrayByParityII(int[] num)
        {
            var newArray = new int[num.Length];
            for (int i = 0, i0 = 0, i1 = 1; i < num.Length; i++)
            {
                if (num[i] % 2 == 0)
                {
                    newArray[i0] = num[i];
                    i0 += 2;
                }
                else
                {
                    newArray[i1] = num[i];
                    i1 += 2;
                }
            }

            return newArray;
        }

        #endregion

        #region 349. 两个数组的交集

        //https://leetcode-cn.com/problems/intersection-of-two-arrays/
        //
        public static int[] Intersection(int[] nums1, int[] nums2)
        {
            HashSet<int> set1 = new HashSet<int>(nums1), set2 = new HashSet<int>(nums2);
            set1.IntersectWith(set2);
            return set1.ToArray();
        }

        #endregion

        #region 1403. 非递增顺序的最小子序列

        //https://leetcode-cn.com/problems/minimum-subsequence-in-non-increasing-order/
        //1403. 非递增顺序的最小子序列
        public static IList<int> MinSubsequence(int[] nums)
        {
            Array.Sort(nums);
            int total = nums.Sum(), current = 0;
            var result = new List<int>();
            for (int i = nums.Length - 1; i >= 0; i--)
            {
                current += nums[i];
                result.Add(nums[i]);
                if (current > total - current)
                {
                    break;
                }
            }

            return result;
        }

        #endregion

        //|r1 - r2| + |c1 - c2|
        public int[][] AllCellsDistOrder(int R, int C, int r0, int c0)
        {
            var result = new int[R * C][];
            var i = 0;
            for (int r = 0; r < R; r++)
            {
                for (int c = 0; c < C; c++)
                {
                    result[i++] = new[] { r, c };
                }
            }

            return result.OrderBy(it => Math.Abs(it[0] - r0) + Math.Abs(it[1] - c0)).ToArray();
        }

        #region 两个数组的交集 II

        //https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/
        //350. 两个数组的交集 II
        public static int[] Intersect(int[] nums1, int[] nums2)
        {
            var result = new List<int>();
            var dic1 = new Dictionary<int, int>();
            foreach (var num in nums1)
            {
                if (dic1.ContainsKey(num))
                {
                    dic1[num]++;
                }
                else
                {
                    dic1[num] = 1;
                }
            }

            foreach (var num in nums2)
            {
                if (!dic1.TryGetValue(num, out var size) || size <= 0)
                {
                    continue;
                }

                result.Add(num);
                dic1[num]--;
            }

            return result.ToArray();
        }

        #endregion

        #region 976. 三角形的最大周长

        //https://leetcode-cn.com/problems/largest-perimeter-triangle/
        //976. 三角形的最大周长

        static bool IsCan(IList<int> items)
        {
            if (items.Count < 3)
            {
                return false;
            }

            return items[0] + items[1] > items[2] && items[0] + items[2] > items[1] && items[2] + items[1] > items[0];
        }

        static void SubSeq(IList<int> nums, IList<int> result, IList<IList<int>> list, int start)
        {
            if (result.Count == 2)
            {
                list.Add(result.ToArray());
                return;
            }

            for (var i = start; i < nums.Count; i++)
            {
                result.Add(nums[i]);
                SubSeq(nums, result, list, i + 1);
                result.RemoveAt(result.Count - 1);
            }
        }

        public static int LargestPerimeter(int[] nums)
        {
            if (nums.Length < 3)
            {
                return 0;
            }

            for (int i = 0, end = nums.Length - 1; i < end; i++)
            {
                for (var j = 0; j < end - i; j++)
                {
                    if (nums[j] > nums[j + 1])
                    {
                        Swap(nums, j, j + 1);
                    }
                }

                if (i >= 2)
                {
                    if (nums[end - i + 2] > nums[end - i + 1] + nums[end - i])
                    {
                        return nums[end - i + 2] + nums[end - i + 1] + nums[end - i];
                    }
                }
            }

            return 0;
        }

        #endregion

        #region 1370. 上升下降字符串

        //https://leetcode-cn.com/problems/increasing-decreasing-string/
        //1370. 上升下降字符串
        public static string SortString(string s)
        {
            var chars = new int[26];
            foreach (var c in s)
            {
                chars[c - 'a']++;
            }

            var result = new StringBuilder();
            var flag = true;
            while (result.Length < s.Length)
            {
                if (flag)
                {
                    for (var i = 0; i < chars.Length; i++)
                    {
                        if (chars[i] == 0)
                        {
                            continue;
                        }

                        result.Append((char)(i + 'a'));
                        chars[i]--;
                    }

                    flag = false;
                }
                else
                {
                    for (var i = chars.Length - 1; i >= 0; i--)
                    {
                        if (chars[i] == 0)
                        {
                            continue;
                        }

                        result.Append((char)(i + 'a'));
                        chars[i]--;
                    }

                    flag = true;
                }
            }

            return result.ToString();
        }

        #endregion

        #region 509. 斐波那契数

        //509. 斐波那契数
        //https://leetcode-cn.com/problems/fibonacci-number/
        public static int Fib(int N)
        {
            if (N == 0)
            {
                return 0;
            }

            if (N == 1)
            {
                return 1;
            }

            var items = new int[N];
            items[0] = 0;
            items[1] = 1;
            for (var i = 2; i < items.Length; i++)
            {
                items[i] = items[i - 1] + items[i - 2];
            }

            return items[N - 1] + items[N - 2];
        }

        #endregion

        #region 面试题10- I. 斐波那契数列

        //面试题10- I. 斐波那契数列
        //https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/

        public static int Fib1(int n)
        {
            if (n == 0 || n == 1)
            {
                return n;
            }

            var items = new int[n + 1];
            items[0] = 0;
            items[1] = 1;
            for (var i = 2; i < items.Length; i++)
            {
                items[i] = (items[i - 1] + items[i - 2]) % 1000000007;
            }

            return items[n];
        }

        #endregion

        #region 面试题03. 数组中重复的数字

        //面试题03. 数组中重复的数字
        //https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/
        public static int FindRepeatNumber(int[] nums)
        {
            var bucket = new bool[nums.Length];
            for (var i = 0; i < nums.Length; i++)
            {
                if (bucket[nums[i]])
                {
                    return nums[i];
                }

                bucket[nums[i]] = true;
            }

            return nums[0];
        }

        public static int FindRepeatNumber1(int[] nums)
        {
            var set = new HashSet<int>();
            for (var i = 0; i < nums.Length; i++)
            {
                if (set.Add(nums[i]))
                {
                    return nums[i];
                }
            }

            return -1;
        }

        #endregion

        #region 面试题04. 二维数组中的查找

        //面试题04. 二维数组中的查找
        //https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/

        public static bool FindNumberIn2DArray(int[][] matrix, int target)
        {
            for (int x = 0; x < matrix.Length; x++)
            {
                var num = matrix[x];
                for (int y = 0; y < num.Length; y++)
                {
                    if (num[y] == target)
                    {
                        return true;
                    }

                    if (num[y] > target)
                    {
                        break;
                    }
                }
            }

            return false;
        }

        //https://leetcode-cn.com/problems/search-a-2d-matrix-ii/
        public static bool FindNumberIn2DArray1(int[][] matrix, int target)
        {
            int x = 0, y = matrix.GetLength(1) - 0;
            while (x < matrix.Length && y >= 0)
            {
                if (matrix[x][y] == target)
                {
                    return true;
                }

                if (matrix[x][y] > target)
                {
                    y--;
                }
                else
                {
                    x++;
                }
            }

            return false;
        }

        #endregion

        #region 面试题10- II. 青蛙跳台阶问题

        //面试题10- II. 青蛙跳台阶问题
        //https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/

        public static int NumWays(int n)
        {
            if (n <= 0)
            {
                return 1;
            }

            var nums = new int[n + 1];
            nums[0] = 1;
            nums[1] = 1;
            for (int i = 2; i < nums.Length; i++)
            {
                nums[i] = (nums[i - 1] + nums[i - 2]) % 1000000007;
            }

            return nums[n];
        }

        #endregion

        #region 面试题11. 旋转数组的最小数字

        //面试题11. 旋转数组的最小数字
        //https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/
        // 输入：[3,4,5,1,2]
        //      [4,5,1,2,3]
        //      [1,2,3]
        // 输出：1
        //1.暴力解 直接遍历数组直至找到第一个后者小于前者的数字
        //2.二分法思想 取数组中位数 和开始节点和结束节点比较 如果小于开始节点 说明最小的节点在前半段，如果大于结束节点，说明最小节点在后半段,如果两种情况都不存在，说明该段有序，循环处理，直到找到最小节点
        public int MinArray(int[] numbers)
        {
            int start = 0, end = numbers.Length - 1;
            while (start < end)
            {
                var mid = (start + end) / 2;
                if (numbers[mid] < numbers[start])
                {
                    //此时存在左区间，同时mid可能是最小值，需要保留
                    end = mid;
                }
                else if (numbers[mid] > numbers[end])
                {
                    //此时存在右区间，同时mid不可能是最小值，排除
                    start = mid + 1;
                }
                else if (numbers[mid] == numbers[start] && numbers[start] == numbers[end])
                {
                    //此时无法判断最小节点位于哪个区间，调整区间
                    end--;
                }
                else
                {
                    //此时可以判断start，end整体有序，直接返回最开始位置数
                    return numbers[start];
                }
            }

            return numbers[start];
        }

        #endregion

        #region 面试题06. 从尾到头打印链表

        //面试题06. 从尾到头打印链表
        //https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/

        public int[] ReversePrint(ListNode head)
        {
            var stack = new Stack<int>();
            while (head != null)
            {
                stack.Push(head.val);
                head = head.next;
            }

            var result = new int[stack.Count];
            for (int i = 0; stack.Count > 0; i++)
            {
                result[i] = stack.Pop();
            }

            return result;
        }

        #endregion

        #region 面试题05. 替换空格

        //面试题05. 替换空格
        //https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/
        public static string ReplaceSpace(string s)
        {
            var res = new StringBuilder();
            for (var i = 0; i < s.Length; i++)
            {
                if (s[i] == ' ')
                {
                    res.Append("%20");
                }
                else
                {
                    res.Append(s[i]);
                }
            }

            return res.ToString();
        }

        #endregion

        #region 面试题25. 合并两个排序的链表

        //面试题25. 合并两个排序的链表
        //https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/

        public static ListNode MergeTwoLists(ListNode l1, ListNode l2)
        {
            if (l1 == null)
            {
                return l2;
            }

            if (l2 == null)
            {
                return l1;
            }

            ListNode root, head;
            if (l1.val > l2.val)
            {
                root = new ListNode(l2.val);
                l2 = l2.next;
            }
            else
            {
                root = new ListNode(l1.val);
                l1 = l1.next;
            }

            head = root;
            while (l1 != null && l2 != null)
            {
                if (l1.val <= l2.val)
                {
                    root.next = new ListNode(l1.val);
                    root = root.next;
                    l1 = l1.next;
                }
                else
                {
                    root.next = new ListNode(l2.val);
                    root = root.next;
                    l2 = l2.next;
                }
            }

            while (l1 != null)
            {
                root.next = new ListNode(l1.val);
                root = root.next;
                l1 = l1.next;
            }

            while (l2 != null)
            {
                root.next = new ListNode(l2.val);
                root = root.next;
                l2 = l2.next;
            }

            return head;
        }

        #endregion

        #region 面试题27. 二叉树的镜像

        //面试题27. 二叉树的镜像
        //https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/
        public static TreeNode MirrorTree(TreeNode root)
        {
            if (root == null)
            {
                return null;
            }

            return new TreeNode(root.val) { left = MirrorTree(root.right), right = MirrorTree(root.left) };
        }

        #endregion

        #region 面试题28. 对称的二叉树

        //面试题28. 对称的二叉树
        //https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/

        static bool IsSymmetric(TreeNode left, TreeNode right)
        {
            if (left == null && right == null)
            {
                return true;
            }

            if (left == null || right == null)
            {
                return false;
            }

            if (left.val != right.val)
            {
                return false;
            }

            return IsSymmetric(left.left, right.right) && IsSymmetric(left.right, right.left);
        }

        public static bool IsSymmetric(TreeNode root)
        {
            return IsSymmetric(root.left, root.right);
        }

        #endregion

        #region 面试题21. 调整数组顺序使奇数位于偶数前面

        //面试题21. 调整数组顺序使奇数位于偶数前面
        //https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/

        public int[] Exchange(int[] nums)
        {
            var result = new int[nums.Length];
            var index = 0;
            foreach (var n in nums)
            {
                if (n % 2 == 1)
                {
                    result[index++] = n;
                }
            }

            foreach (var n in nums)
            {
                if (n % 2 == 0)
                {
                    result[index++] = n;
                }
            }

            return result;
        }

        //首尾指针，快排思想
        public int[] Exchange1(int[] nums)
        {
            int start = 0, end = nums.Length - 1;
            while (start < end)
            {
                //从头直至一个偶数
                while (start < end && (nums[start] & 1) == 1)
                {
                    start++;
                }

                //从尾直至一个基数
                while (start < end && (nums[end] & 1) == 0)
                {
                    end--;
                }

                if (start >= end)
                {
                    break;
                }

                var tmp = nums[start];
                nums[start] = nums[end];
                nums[end] = tmp;
                start++;
                end--;
            }

            return nums;
        }

        //快慢指针
        public int[] Exchange2(int[] nums)
        {
            int slow = 0, fast = 0;
            while (fast < nums.Length)
            {
                if ((nums[fast] & 1) == 1)
                {
                    var cmp = nums[slow];
                    nums[slow] = nums[fast];
                    nums[fast] = cmp;
                    slow++;
                }

                fast++;
            }

            return nums;
        }

        #endregion

        #region 面试题15. 二进制中1的个数/191. 位1的个数

        //面试题15. 二进制中1的个数
        //https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/
        //https://leetcode-cn.com/problems/number-of-1-bits/

        public int HammingWeight(uint n)
        {
            var size = 0;
            while (n != 0)
            {
                if ((n & 1) == 1)
                {
                    size++;
                }

                n = n >> 1;
            }

            return size;
        }

        public int HammingWeight1(uint n)
        {
            var res = 0;
            while (n != 0)
            {
                n = n & (n - 1);
                res++;
            }

            return res;
        }

        #endregion

        #region 面试题29. 顺时针打印矩阵

        //面试题29. 顺时针打印矩阵
        //https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/
        public int[] SpiralOrder(int[][] matrix)
        {
            int x = matrix.Length, y = matrix[0].Length, x1 = 0, y1 = 0;
            var res = new int[x * y];
            var index = 0;
            var type = 0;
            while (index < res.Length)
            {
                switch (type)
                {
                    case 0: //向右
                        for (int i = y1; i < y; i++)
                        {
                            res[index++] = matrix[x1][i];
                        }

                        x1++;
                        type = 1;
                        break;
                    case 1: //向下
                        for (int i = x1; i < x; i++)
                        {
                            res[index++] = matrix[i][y - 1];
                        }

                        y--;
                        type = 2;
                        break;
                    case 2: //向左
                        for (int i = y - 1; i >= y1; i--)
                        {
                            res[index++] = matrix[x - 1][i];
                        }

                        x--;
                        type = 3;
                        break;
                    case 3: //向上
                        for (int i = x - 1; i >= x1; i--)
                        {
                            res[index++] = matrix[i][y1];
                        }

                        y1++;
                        type = 0;
                        break;
                }
            }

            return res;
        }

        #endregion

        #region 面试题22. 链表中倒数第k个节点

        //面试题22. 链表中倒数第k个节点
        //https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/
        public ListNode GetKthFromEnd(ListNode head, int k)
        {
            ListNode fast = head, slow = head;
            while (fast != null)
            {
                k--;
                if (k <= 0)
                {
                    break;
                }

                fast = fast.next;
            }

            if (fast == null)
            {
                return null;
            }

            while (fast != null)
            {
                fast = fast.next;
                slow = slow.next;
            }

            return slow;
        }

        #endregion

        #region 面试题17. 打印从1到最大的n位数

        //面试题17. 打印从1到最大的n位数
        //https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/
        public int[] PrintNumbers(int n)
        {
            var num = 0;
            while (n > 0)
            {
                num = num * 10 + 9;
                n--;
            }

            var result = new List<int>();
            for (int i = 1; i <= num; i++)
            {
                result.Add(i);
            }

            return result.ToArray();
        }

        #endregion

        #region 面试题24. 反转链表

        //面试题24. 反转链表
        //https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/
        public ListNode ReverseList(ListNode head)
        {
            if (head == null || head.next == null)
            {
                return head;
            }

            ListNode prev = null, current = head;
            while (current != null)
            {
                var next = current.next;
                current.next = prev;
                prev = current;
                current = next;
            }

            return prev;
        }

        //递归反转链表
        public ListNode ReverseListByDfs(ListNode head)
        {
            if (head == null || head.next == null)
            {
                return head;
            }

            var next = head.next;
            head.next = null;
            var newHead = ReverseListByDfs(next);
            next.next = head;
            return newHead;
        }

        #endregion

        #region 面试题18. 删除链表的节点

        //面试题18. 删除链表的节点
        //https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/
        public ListNode DeleteNode(ListNode head, int val)
        {
            if (head == null)
            {
                return null;
            }

            ListNode prev = null, current = head;
            while (current != null)
            {
                if (current.val == val)
                {
                    if (prev == null)
                    {
                        //头节点
                        return current.next;
                    }

                    prev.next = current.next;
                    current.next = null;
                    break;
                }

                prev = current;
                current = current.next;
            }

            return head;
        }

        #endregion

        #region 面试题07. 重建二叉树

        //面试题07. 重建二叉树
        //https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/
        //原理：二叉树前序遍历 根节点->左节点->右节点 中序遍历 左节点->根节点->右节点 
        // 根据前序遍历原理找出当前根节点，中序遍历中找出根节点左右子树，递归恢复

        public static TreeNode BuildTree(int[] preorder, int pStart, int pEnd, int[] inorder, int iStart, int iEnd)
        {
            if (pStart > pEnd)
            {
                return null;
            }

            var root = new TreeNode(preorder[pStart]);
            var index = 0;
            for (int i = iStart; i <= iEnd; i++)
            {
                if (inorder[i] == root.val)
                {
                    break;
                }

                index++;
            }

            root.left = BuildTree(preorder, pStart + 1, pStart + index, inorder, iStart, iStart + index - 1);
            root.right = BuildTree(preorder, pStart + index + 1, pEnd, inorder, iStart + index + 1, iEnd);
            return root;
        }

        public TreeNode BuildTree(int[] preorder, int[] inorder)
        {
            return BuildTree(preorder, 0, preorder.Length - 1, inorder, 0, inorder.Length - 1);
        }

        #endregion

        #region 面试题30. 包含min函数的栈

        //面试题30. 包含min函数的栈
        //https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/
        //定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

        public class MinStack
        {
            private Stack<int> stack = new Stack<int>();
            private Stack<int> min = new Stack<int>();

            public void Push(int x)
            {
                stack.Push(x);
                if (min.Count <= 0 || min.Peek() >= x)
                {
                    min.Push(x);
                }
            }

            public void Pop()
            {
                var x = stack.Pop();
                if (x <= min.Peek())
                {
                    min.Pop();
                }
            }

            public int Top()
            {
                return stack.Peek();
            }

            public int Min()
            {
                return min.Peek();
            }
        }

        #endregion

        #region 面试题32 - II. 从上到下打印二叉树 II

        //面试题32 - II. 从上到下打印二叉树 II
        //https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/
        //从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
        public IList<IList<int>> LevelOrder(TreeNode root)
        {
            if (root == null)
            {
                return new IList<int>[0];
            }

            var queue = new Queue<TreeNode>();
            var result = new List<IList<int>>();
            queue.Enqueue(root);
            var size = queue.Count;
            while (queue.Count > 0)
            {
                var items = new List<int>();
                while (size > 0)
                {
                    root = queue.Dequeue();
                    items.Add(root.val);
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

                result.Add(items);
                size = queue.Count;
            }

            return result;
        }

        #endregion

        #region 面试题39. 数组中出现次数超过一半的数字

        //面试题39. 数组中出现次数超过一半的数字
        //https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/
        //https://leetcode-cn.com/problems/majority-element/
        //数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
        //解法1：最直观先排序，取中间数
        public int MajorityElement(int[] nums)
        {
            //如果超过1半，挨个比较size肯定会大于0
            int num = nums[0], size = 1;
            for (int i = 1; i < nums.Length; i++)
            {
                if (size <= 0)
                {
                    num = nums[i];
                }

                if (num == nums[i])
                {
                    size++;
                }
                else
                {
                    size--;
                }
            }

            return num;
        }

        #endregion

        #region 面试题40. 最小的k个数

        //面试题40.最小的k个数
        //https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/
        //输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4
        //top k，构建大顶堆，替换最大数，重新构建堆，全部完成后堆内即为最小的数（大数都被丢弃）
        //  1
        // 2 3
        //4

        static void BuildHeap(int[] arr, int index, int length)
        {
            while (true)
            {
                int left = index * 2 + 1, right = left + 1;
                if (left > length)
                {
                    break;
                }

                if (right <= length && arr[left] < arr[right])
                {
                    left++;
                }

                if (arr[left] > arr[index])
                {
                    var tmp = arr[left];
                    arr[left] = arr[index];
                    arr[index] = tmp;
                }

                index = left;
            }
        }

        static void MoveDown(int[] arr, int index, int lastIndex)
        {
            while (true)
            {
                int left = index * 2 + 1, right = left + 1;
                if (left > lastIndex)
                {
                    break;
                }

                if (right <= lastIndex && arr[left] < arr[right])
                {
                    left++;
                }

                if (arr[left] > arr[index])
                {
                    var tmp = arr[left];
                    arr[left] = arr[index];
                    arr[index] = tmp;
                }
                else
                {
                    break;
                }

                index = left;
            }
        }

        public static int[] GetLeastNumbers(int[] arr, int k)
        {
            for (int i = k / 2; i >= 0; i--)
            {
                BuildHeap(arr, i, k - 1);
            }

            for (int i = k; i < arr.Length; i++)
            {
                if (arr[i] < arr[0])
                {
                    arr[0] = arr[i];
                    MoveDown(arr, 0, k - 1);
                }
            }

            var res = new int[k];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = arr[i];
            }

            return res;
        }

        #endregion

        #region 面试题42. 连续子数组的最大和

        //面试题42. 连续子数组的最大和
        //https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/
        //输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值

        public int MaxSubArray1(int[] nums)
        {
            int num = nums[0], sum = num;
            for (int i = 1; i < nums.Length; i++)
            {
                if (sum > 0)
                {
                    sum += nums[i];
                }
                else
                {
                    sum = nums[i];
                }

                num = Math.Max(sum, num);
            }

            return num;
        }

        #endregion

        #region 面试题50. 第一个只出现一次的字符

        //面试题50. 第一个只出现一次的字符
        //https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/
        //https://leetcode-cn.com/problems/first-unique-character-in-a-string/
        public char FirstUniqChar(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return ' ';
            }

            var dic = new Dictionary<char, bool>();
            foreach (var c in s)
            {
                dic[c] = dic.ContainsKey(c);
            }

            foreach (var c in s)
            {
                if (!dic[c])
                {
                    return c;
                }
            }

            return ' ';
        }

        #endregion

        #region 面试题55 - I. 二叉树的深度

        //面试题55 - I. 二叉树的深度
        //https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/
        public int MaxDepth(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            return Math.Max(MaxDepth(root.left), MaxDepth(root.right)) + 1;
        }

        #endregion

        #region 面试题57. 和为s的两个数字/167. 两数之和 II - 输入有序数组

        //面试题57. 和为s的两个数字
        //https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/
        //https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/
        public static int Find(int[] nums, int target)
        {
            int start = 0, end = nums.Length;
            while (start < end)
            {
                var mid = (start + end) / 2;
                if (nums[mid] > target)
                {
                    start = mid + 1;
                }
                else if (nums[mid] < target)
                {
                    end = mid - 1;
                }
                else
                {
                    return mid;
                }
            }

            return -1;
        }

        //二分查找数字
        public int[] TwoSum(int[] nums, int target)
        {
            for (var i = 0; i < nums.Length; i++)
            {
                var num = target - nums[i];
                var index = Find(nums, num);
                if (index != -1)
                {
                    return new[] { nums[i], nums[index] };
                }
            }

            return new int[0];
        }

        //基于set，记录出现过的数字，遍历找出之前是否出现过
        public int[] TwoSum1(int[] nums, int target)
        {
            var set = new HashSet<int>();
            for (var i = 0; i < nums.Length; i++)
            {
                set.Add(nums[i]);
                var num = target - nums[i];
                if (set.Contains(num))
                {
                    return new[] { nums[i], num };
                }
            }

            return new int[0];
        }

        //双指针，分别头尾扫描，两两相加，和大于target说明尾部数大前移，小则头部前移
        public int[] TwoSum2(int[] nums, int target)
        {
            int start = 0, end = nums.Length - 1;
            while (start < end)
            {
                var num = nums[start] + nums[end];
                if (num == target)
                {
                    return new[] { nums[start], nums[end] };
                }

                if (num > target)
                {
                    end--;
                }
                else
                {
                    start++;
                }
            }

            return new int[0];
        }

        #endregion

        #region 面试题54. 二叉搜索树的第k大节点

        //面试题54. 二叉搜索树的第k大节点
        //https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/
        static void Loop(TreeNode root, IList<int> items)
        {
            while (true)
            {
                if (root == null)
                {
                    return;
                }

                Loop(root.left, items);
                items.Add(root.val);
                root = root.right;
            }
        }

        public int KthLargest(TreeNode root, int k)
        {
            var items = new List<int>();
            Loop(root, items);
            return items[items.Count - k];
        }

        #endregion

        #region 面试题53 - I. 在排序数组中查找数字 I

        //面试题53 - I. 在排序数组中查找数字 I
        //https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/
        public int Search(int[] nums, int target)
        {
            if (nums == null || nums.Length == 0)
            {
                return 0;
            }

            int start = 0, end = nums.Length - 1, mid = (start + end) / 2;
            while (start <= end)
            {
                if (nums[mid] == target)
                {
                    var size = 1;
                    for (int i = mid + 1; i <= end; i++)
                    {
                        if (nums[i] != target)
                        {
                            break;
                        }

                        size++;
                    }

                    for (int i = mid - 1; i >= start; i--)
                    {
                        if (nums[i] != target)
                        {
                            break;
                        }

                        size++;
                    }

                    return size;
                }

                if (nums[mid] > target)
                {
                    end = mid - 1;
                }
                else
                {
                    start = mid + 1;
                }

                mid = (start + end) / 2;
            }

            return 0;
        }

        //二分查找边界
        public int SearchFindRange(int[] nums, int target)
        {
            if (nums == null || nums.Length <= 0)
            {
                return 0;
            }

            int start = 0, end = nums.Length - 1, left, right;
            while (start <= end)
            {
                var mid = (start + end) / 2;
                if (nums[mid] <= target)
                {
                    start = mid + 1;
                }
                else
                {
                    end = mid - 1;
                }
            }

            if (end >= 0 && nums[end] != target)
            {
                return 0;
            }

            right = end;
            start = 0;
            while (start <= end)
            {
                var mid = (start + end) / 2;
                if (nums[mid] >= target)
                {
                    end = mid - 1;
                }
                else
                {
                    start = mid + 1;
                }
            }

            left = start;
            return right - left + 1;
        }

        #endregion

        #region 面试题58 - I. 翻转单词顺序

        //面试题58 - I. 翻转单词顺序
        //https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/        
        public string ReverseWords(string s)
        {
            var res = new StringBuilder();
            var length = 0;
            for (int i = s.Length - 1; i >= 0; i--)
            {
                var ch = s[i];
                if (ch == ' ')
                {
                    if (length > 0)
                    {
                        if (res.Length > 0)
                        {
                            res.Append(' ');
                        }

                        res.Append(s, i + 1, length);
                        length = 0;
                    }

                    continue;
                }

                length++;
            }

            if (length > 0)
            {
                if (res.Length > 0)
                {
                    res.Append(' ');
                }

                res.Append(s, 0, length);
            }

            return res.ToString();
        }

        #endregion


        #region 557. 反转字符串中的单词 III

        //https://leetcode-cn.com/problems/reverse-words-in-a-string-iii/
        public string ReverseWordsIII(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return s;
            }

            var result = new StringBuilder();
            for (int i = 0, j = 0; i < s.Length; i++)
            {
                if (s[i] != ' ' && i != s.Length - 1)
                {
                    continue;
                }

                var str = s.Substring(j, i == s.Length - 1 ? i - j + 1 : i - j);
                result.Append(new string(str.Reverse().ToArray())).Append(' ');
                j = i + 1;
            }

            return result.ToString(0, result.Length - 1);
        }

        #endregion


        #region 面试题58 - II. 左旋转字符串

        //面试题58 - II. 左旋转字符串/
        //https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/
        public string ReverseLeftWords(string s, int n)
        {
            n = n % s.Length;
            var result = new StringBuilder();
            result.Append(s.Substring(n));
            result.Append(s.Substring(s.Length - n - 1, n));
            return result.ToString();
        }

        #endregion

        #region 面试题66. 构建乘积数组

        //面试题66. 构建乘积数组
        //https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/
        public int[] ConstructArr(int[] a)
        {
            int[] prev = new int[a.Length], next = new int[a.Length], res = new int[a.Length];
            for (int i = 0, j = a.Length - 1; i < a.Length; i++, j--)
            {
                if (i == 0)
                {
                    prev[i] = a[i];
                    next[j] = a[j];
                }
                else
                {
                    prev[i] = prev[i - 1] * a[i];
                    next[j] = next[j + 1] * a[j];
                }
            }

            for (int i = 0, j = a.Length - 1; i < a.Length; i++)
            {
                res[i] = (i == 0 ? 1 : prev[i - 1]) * (i == j ? 1 : next[i + 1]);
            }

            return res;
        }

        #endregion

        #region 面试题55 - II. 平衡二叉树/面试题 04.04. 检查平衡性

        //https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/
        //https://leetcode-cn.com/problems/check-balance-lcci/
        public bool IsBalanced(TreeNode root)
        {
            if (root == null)
            {
                return true;
            }

            if (Math.Abs(MaxDepth(root.left) - MaxDepth(root.right)) > 1)
            {
                return false;
            }

            return IsBalanced(root.left) && IsBalanced(root.right);
        }

        #endregion

        #region 面试题60. n个骰子的点数

        //面试题60. n个骰子的点数
        //https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/

        void Make(int[] nums, int current, int sum)
        {
            if (current == 1)
            {
                for (int i = 1; i <= 6; i++)
                {
                    nums[i + sum]++;
                }
            }
            else
            {
                for (int i = 1; i <= 6; i++)
                {
                    Make(nums, current - 1, sum + i);
                }
            }
        }

        public double[] TwoSum(int n)
        {
            if (n < 1)
            {
                return new double[0];
            }

            var nums = new int[n * 6 + 1];
            Make(nums, n, 0);
            var total = Math.Pow(6, n);
            var result = new List<double>();
            foreach (var num in nums)
            {
                if (num == 0)
                {
                    continue;
                }

                result.Add(num / total);
            }

            return result.ToArray();
        }

        #endregion

        #region 面试题61. 扑克牌中的顺子

        //面试题61. 扑克牌中的顺子
        //https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/
        public bool IsStraight(int[] nums)
        {
            Array.Sort(nums);
            int start = 0, end = nums.Length - 1;
            while (start < end)
            {
                if (nums[end] == nums[end - 1] && nums[end] != 0)
                {
                    return false;
                }

                var diff = nums[end] - nums[end - 1] - 1;
                if (diff != 0)
                {
                    while (diff != 0 && start < end)
                    {
                        if (nums[start++] != 0)
                        {
                            return false;
                        }

                        if (start == end)
                        {
                            return true;
                        }

                        diff--;
                    }

                    if (diff != 0)
                    {
                        return false;
                    }
                }

                end--;
            }

            return true;
        }

        #endregion

        #region 面试题62. 圆圈中最后剩下的数字

        //面试题62. 圆圈中最后剩下的数字
        //https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/
        public int LastRemaining(int n, int m)
        {
            var result = new List<int>(n);
            for (int i = 0; i < n; i++)
            {
                result.Add(i);
            }

            var prev = (m - 1) % result.Count;
            while (result.Count > 1)
            {
                Console.WriteLine(result[prev]);
                result.RemoveAt(prev);
                prev = (prev + m - 1) % result.Count;
            }

            return result[0];
        }

        #endregion

        #region 面试题68 - I. 二叉搜索树的最近公共祖先

        //面试题68 - I. 二叉搜索树的最近公共祖先
        //https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/
        public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q)
        {
            if (root == null || (root.left == null && root.right == null))
            {
                return null;
            }

            if (p.val > q.val) //1.排序结点，保证p<q
            {
                var tmp = q;
                p = q;
                q = tmp;
            }

            if (q.val < root.val) //2.与根节点比较，如果q<root，说明两个节点都在左子树
            {
                return LowestCommonAncestor(root.left, p, q);
            }

            if (p.val > root.val) //3.p>root，说明两个节点在右节点
            {
                return LowestCommonAncestor(root.right, p, q);
            }

            //此时节点分布在左右子树或1个在根节点，1个在左子树或右子树，此时root是根节点
            return root;
        }

        #endregion

        #region 面试题68 - II. 二叉树的最近公共祖先/面试题 04.08. 首个共同祖先/236. 二叉树的最近公共祖先

        //https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/
        //https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/

        public bool FindChild(TreeNode root, TreeNode child)
        {
            while (true)
            {
                if (root == null)
                {
                    return false;
                }

                if (child == null || root.val == child.val)
                {
                    return true;
                }

                if (FindChild(root.left, child))
                {
                    return true;
                }

                root = root.right;
            }
        }

        public TreeNode LowestCommonAncestorII(TreeNode root, TreeNode p, TreeNode q)
        {
            if (root == null || (root.left == null && root.right == null))
            {
                return null;
            }

            TreeNode node = LowestCommonAncestorII(root.right, p, q);
            if (node == null)
            {
                node = LowestCommonAncestorII(root.left, p, q);
            }

            if (node != null)
            {
                return node;
            }

            //该二叉树非二叉搜索树，无法判断节点分布在哪个子树，需要分别在左右子树进行搜索
            //如果左右子树分布搜索都没有，说明分布在左右子树，此时由该根节点搜索两个节点            
            return FindChild(root, p) && FindChild(root, q) ? root : null;
        }

        public TreeNode LowestCommonAncestorOn(TreeNode root, TreeNode p, TreeNode q)
        {
            while (true)
            {
                if (root == null)
                {
                    return null;
                }

                if (root == p || root == q)
                {
                    return root;
                }

                TreeNode left = LowestCommonAncestorOn(root.left, p, q),
                    right = LowestCommonAncestorOn(root.right, p, q);
                if (left != null && right != null)
                {
                    return root;
                }

                root = left ?? right;
            }
        }

        #endregion

        #region 面试题52. 两个链表的第一个公共节点/面试题 02.07. 链表相交

        //面试题52. 两个链表的第一个公共节点
        //https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/
        //https://leetcode-cn.com/problems/intersection-of-two-linked-lists-lcci/
        public ListNode GetIntersectionNode(ListNode headA, ListNode headB)
        {
            int lenA = 0, lenB = 0;
            ListNode nodeA = headA, nodeB = headB;
            while (nodeA != null && nodeB != null)
            {
                lenA++;
                lenB++;
                nodeA = nodeA.next;
                nodeB = nodeB.next;
            }

            while (nodeA != null)
            {
                lenA++;
                nodeA = nodeA.next;
            }

            while (nodeB != null)
            {
                lenB++;
                nodeB = nodeB.next;
            }

            while (headA != null && headB != null)
            {
                if (lenA > lenB)
                {
                    lenA--;
                    headA = headA.next;
                }
                else if (lenA < lenB)
                {
                    lenB--;
                    headB = headB.next;
                }
                else
                {
                    if (headA == headB)
                    {
                        return headA;
                    }

                    headA = headA.next;
                    headB = headB.next;
                }
            }

            return null;
        }

        #endregion

        #region 面试题53 - II. 0～n-1中缺失的数字（有序数组）

        //面试题53 - II. 0～n-1中缺失的数字
        //https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/
        public int MissingNumber(int[] nums)
        {
            int start = 0, end = nums.Length - 1;
            while (start <= end)
            {
                //[0,1,3,4,5]
                var mid = (start + end) / 2;
                if (nums[mid] == mid)
                {
                    start = mid + 1;
                }
                else if (nums[mid] > mid)
                {
                    end = mid - 1;
                }
            }

            return start;
        }

        #endregion

        #region 面试题57 - II. 和为s的连续正数序列

        //面试题57 - II. 和为s的连续正数序列
        //https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/
        public int[][] FindContinuousSequence(int target)
        {
            var end = target / 2 + 1;
            var result = new List<int[]>();
            var list = new List<int>();
            var sum = 0;
            for (int i = 0; i < end; i++)
            {
                var num = i + 1;
                list.Add(num);
                sum += num;
                while (sum > target && list.Count > 0)
                {
                    sum -= list[0];
                    list.RemoveAt(0);
                }

                if (sum == target)
                {
                    result.Add(list.ToArray());
                }
            }

            return result.ToArray();
        }

        #endregion

        #region 面试题32 - I. 从上到下打印二叉树

        //面试题32 - I. 从上到下打印二叉树
        //https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/
        public int[] LevelOrder1(TreeNode root)
        {
            if (root == null)
            {
                return new int[0];
            }

            var queue = new Queue<TreeNode>();
            var result = new List<int>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                root = queue.Dequeue();
                if (root == null)
                {
                    continue;
                }

                result.Add(root.val);
                queue.Enqueue(root.left);
                queue.Enqueue(root.right);
            }

            return result.ToArray();
        }

        #endregion

        #region 面试题12. 矩阵中的路径/79. 单词搜索

        //面试题12. 矩阵中的路径
        //https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/
        //https://leetcode-cn.com/problems/word-search/

        bool Move(char[][] board, bool[,] flag, int x, int y, int index, string word)
        {
            if (x < 0 || x >= board.Length || y < 0 || y >= board[0].Length || flag[x, y])
            {
                return false;
            }

            if (board[x][y] == word[index])
            {
                flag[x, y] = true;
                if (index == word.Length - 1)
                {
                    return true;
                }

                var res = Move(board, flag, x, y - 1, index + 1, word) ||
                          Move(board, flag, x, y + 1, index + 1, word) ||
                          Move(board, flag, x - 1, y, index + 1, word) ||
                          Move(board, flag, x + 1, y, index + 1, word);
                if (res)
                {
                    return true;
                }

                flag[x, y] = false;
            }

            return false;
        }

        public bool Exist(char[][] board, string word)
        {
            var flag = new bool[board.Length, board[0].Length];
            for (int i = 0; i < board.Length; i++)
            {
                for (int j = 0; j < board[i].Length; j++)
                {
                    if (Move(board, flag, i, j, 0, word))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        #endregion

        #region 面试题26. 树的子结构

        //面试题26. 树的子结构
        //https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/
        bool IsSame(TreeNode a, TreeNode b)
        {
            if (b == null)
            {
                return true;
            }

            if (a == null)
            {
                return false;
            }

            return a.val == b.val && IsSame(a.left, b.left) && IsSame(a.right, b.right);
        }

        public bool IsSubStructure(TreeNode A, TreeNode B)
        {
            if (B == null)
            {
                return false;
            }

            if (A == null)
            {
                return false;
            }

            if (A.val == B.val)
            {
                var res = IsSame(A.left, B.left) && IsSame(A.right, B.right);
                if (res)
                {
                    return true;
                }
            }

            return IsSubStructure(A.left, B) || IsSubStructure(A.right, B);
        }

        #endregion

        #region 202. 快乐数

        //202. 快乐数
        //https://leetcode-cn.com/problems/happy-number/
        public bool IsHappy(int n)
        {
            var num = 0;
            var set = new HashSet<int>();
            while (n != 1 && !set.Contains(n))
            {
                set.Add(n);
                while (n > 0)
                {
                    num += (int)Math.Pow(n % 10, 2);
                    n /= 10;
                }

                n = num;
                num = 0;
            }

            return n == 1;
        }

        #endregion

        #region 面试题14- I. 剪绳子

        //面试题14- I. 剪绳子
        //https://leetcode-cn.com/problems/jian-sheng-zi-lcof/
        //思路：
        //CuttingRope(n)可以看作是Max(CuttingRope(n-1)*1,CuttingRope(n-2)*2,…………,CuttingRope(1)*(n-1))，可以进行递归
        //另外每次将一段绳子剪成两段时，剩下的部分可以继续剪，也可以不剪
        //所以最终公式 F(n)=Max(F(n-i)*i,(n-i)*i)
        int CuttingRope(int n, int[] prevs)
        {
            if (prevs[n] != 0)
            {
                return prevs[n];
            }

            var max = -1;
            for (int i = 2; i <= n; i++)
            {
                max = Math.Max(max, Math.Max(i * CuttingRope(n - i), i * (n - i)));
            }

            prevs[n] = max;
            return max;
        }

        public int CuttingRope(int n)
        {
            if (n <= 2)
            {
                return 1;
            }

            var prevs = new int[n + 1];
            prevs[0] = 0;
            prevs[1] = 1;
            prevs[2] = 1;
            return CuttingRope(n, prevs);
        }

        #endregion

        #region 3. 无重复字符的最长子串

        //3. 无重复字符的最长子串
        //https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
        public int LengthOfLongestSubstring(string s)
        {
            var set = new HashSet<char>();
            var max = -1;
            for (int i = 0, j = 0; i < s.Length; i++)
            {
                while (set.Contains(s[i]) && j <= i)
                {
                    set.Remove(s[j]);
                    j++;
                }

                set.Add(s[i]);
                max = Math.Max(max, set.Count);
            }

            return max;
        }

        #endregion

        #region 面试题32 - III. 从上到下打印二叉树 III

        //面试题32 - III. 从上到下打印二叉树 III
        //https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/
        public IList<IList<int>> LevelOrder3(TreeNode root)
        {
            var result = new List<IList<int>>();
            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            var size = queue.Count;
            while (queue.Count > 0)
            {
                var items = new List<int>();
                while (size > 0)
                {
                    root = queue.Dequeue();
                    size--;
                    if (root == null)
                    {
                        continue;
                    }

                    if (result.Count % 2 == 0)
                    {
                        items.Add(root.val);
                    }
                    else
                    {
                        items.Insert(0, root.val);
                    }

                    queue.Enqueue(root.left);
                    queue.Enqueue(root.right);
                }

                size = queue.Count;
                if (items.Count > 0)
                {
                    result.Add(items);
                }
            }

            return result;
        }

        #endregion

        #region 面试题33. 二叉搜索树的后序遍历序列

        //面试题33. 二叉搜索树的后序遍历序列
        //https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/

        bool CheckTree(int[] postorder, int start, int end)
        {
            if (start >= end)
            {
                return true;
            }

            var root = postorder[end];
            var index = end - 1;
            while (start < index && postorder[index] >= root)
            {
                index--;
            }

            for (int i = start; i < index; i++)
            {
                if (postorder[i] > root)
                {
                    return false;
                }
            }

            return CheckTree(postorder, start, index) && CheckTree(postorder, index + 1, end - 1);
        }

        public bool VerifyPostorder(int[] postorder)
        {
            return CheckTree(postorder, 0, postorder.Length - 1);
        }

        #endregion

        #region 面试题34. 二叉树中和为某一值的路径/113. 路径总和 II

        //https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/
        //https://leetcode-cn.com/problems/path-sum-ii/
        void PathSum(TreeNode node, IList<IList<int>> result, List<int> path, int sum, int target)
        {
            path.Add(node.val);
            sum += node.val;

            if (node.left == null && node.right == null)
            {
                if (sum == target)
                {
                    result.Add(path.ToArray());
                }
            }

            if (sum < target)
            {
                if (node.left != null)
                {
                    PathSum(node.left, result, path, sum, target);
                }

                if (node.right != null)
                {
                    PathSum(node.right, result, path, sum, target);
                }
            }

            path.RemoveAt(path.Count - 1);
        }

        public IList<IList<int>> PathSum(TreeNode root, int sum)
        {
            var result = new List<IList<int>>();
            PathSum(root, result, new List<int>(), 0, sum);
            return result;
        }

        #endregion

        #region 98. 验证二叉搜索树/面试题 04.05. 合法二叉搜索树

        //https://leetcode-cn.com/problems/validate-binary-search-tree/
        //https://leetcode-cn.com/problems/legal-binary-search-tree-lcci/
        public bool IsValidBST(TreeNode node, TreeNode max, TreeNode min)
        {
            if (node == null)
            {
                return true;
            }

            if (max != null && node.val >= max.val)
            {
                return false;
            }

            if (min != null && node.val <= min.val)
            {
                return false;
            }

            return IsValidBST(node.left, node, min) && IsValidBST(node.right, max, node);
        }

        public bool IsValidBST(TreeNode root)
        {
            if (root == null)
            {
                return true;
            }

            return IsValidBST(root, null, null);
        }

        public bool IsValidBSTByStack(TreeNode root)
        {
            if (root == null)
            {
                return true;
            }

            var stack = new Stack<TreeNode>();
            TreeNode prev = null;
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                if (prev == null)
                {
                    prev = root;
                }
                else if (prev.val >= root.val)
                {
                    return false;
                }
                else
                {
                    prev = root;
                }

                root = root.right;
            }

            return true;
        }

        #endregion

        #region 572. 另一个树的子树/面试题 04.10. 检查子树

        //https://leetcode-cn.com/problems/subtree-of-another-tree/
        //https://leetcode-cn.com/problems/check-subtree-lcci/
        bool CheckSubtree(TreeNode s, TreeNode t)
        {
            if (s == null)
            {
                return t == null;
            }

            if (t == null)
            {
                return false;
            }

            if (s.val == t.val)
            {
                return CheckSubtree(s.left, t.left) && CheckSubtree(s.right, t.right);
            }

            return false;
        }

        public bool IsSubtree(TreeNode s, TreeNode t)
        {
            if (s == null)
            {
                return false;
            }

            if (CheckSubtree(s, t))
            {
                return true;
            }

            return IsSubtree(s.left, t) || IsSubtree(s.right, t);
        }

        #endregion

        #region 69. x 的平方根

        //69. x 的平方根
        //https://leetcode-cn.com/problems/sqrtx/

        public int MySqrt(int x)
        {
            if (x <= 1)
            {
                return x;
            }

            double low = 0, high = x, num;
            while (true)
            {
                num = (low + high) / 2;
                var pow = num * num;
                var diff = Math.Abs(pow - x);
                if (diff <= 0.00001)
                {
                    break;
                }

                if (pow > x)
                {
                    high = num;
                }
                else
                {
                    low = num;
                }
            }

            return (int)num;
        }

        #endregion

        #region 50. Pow(x, n)

        //50. Pow(x, n)
        //https://leetcode-cn.com/problems/powx-n/
        double Pow(double x, int n)
        {
            if (n == 0)
            {
                return 1.0;
            }

            if (n == 1)
            {
                return x;
            }

            var y = Pow(x, n / 2);
            return n % 2 == 0 ? y * y : y * y * x;
        }

        public double MyPow(double x, int n)
        {
            if (n >= 0)
            {
                return Pow(x, n);
            }

            return 1.0 / Pow(x, -n);
        }

        #endregion

        #region 面试题67. 把字符串转换成整数

        //面试题67. 把字符串转换成整数
        //https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/
        public int StrToInt(string str)
        {
            bool flag = true, check = true;
            int result = 0, limit = int.MaxValue / 10;
            foreach (var ch in str)
            {
                if (char.IsDigit(ch))
                {
                    if (limit < result)
                    {
                        return flag ? int.MaxValue : int.MinValue;
                    }

                    var newVal = result * 10 + (ch - '0');
                    if (newVal < result)
                    {
                        return flag ? int.MaxValue : int.MinValue;
                    }

                    result = newVal;
                }
                else if (result == 0 && check)
                {
                    if (ch == ' ')
                    {
                        continue;
                    }

                    if (ch == '-')
                    {
                        check = false;
                        flag = false;
                    }
                    else if (ch == '+')
                    {
                        check = false;
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }

            return flag ? result : -result;
        }

        #endregion

        #region 面试题35. 复杂链表的复制

        //面试题35. 复杂链表的复制
        //https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/
        Node CopyNode(Node node, Dictionary<Node, Node> nodeDic)
        {
            if (node == null)
            {
                return null;
            }

            if (nodeDic.TryGetValue(node, out var cpNode))
            {
                return cpNode;
            }

            cpNode = new Node(node.val);
            nodeDic.Add(node, cpNode);
            cpNode.next = CopyNode(node.next, nodeDic);
            cpNode.random = CopyNode(node.random, nodeDic);
            return cpNode;
        }

        public Node CopyRandomList(Node head)
        {
            return CopyNode(head, new Dictionary<Node, Node>());
        }

        #endregion

        #region 面试题36. 二叉搜索树与双向链表

        //面试题36. 二叉搜索树与双向链表
        //https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/
        public TreeNode TreeToDoublyList(TreeNode root)
        {
            if (root == null)
            {
                return root;
            }

            TreeNode head = null, prev = null;
            var stack = new Stack<TreeNode>();
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                if (stack.Count > 0)
                {
                    root = stack.Pop();
                    if (head == null)
                    {
                        head = prev = root;
                    }
                    else
                    {
                        prev.right = root;
                        root.left = prev;
                        prev = root;
                    }

                    root = root.right;
                }
            }

            head.left = prev;
            prev.right = head;

            return root;
        }

        #endregion

        #region 面试题31. 栈的压入、弹出序列

        //面试题31. 栈的压入、弹出序列
        //https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/
        public bool ValidateStackSequences(int[] pushed, int[] popped)
        {
            var stack = new Stack<int>();
            int i = 0, j = 0;
            while (i < pushed.Length && j < popped.Length)
            {
                if (pushed[i] == popped[j])
                {
                    i++;
                    j++;
                    continue;
                }

                if (stack.TryPeek(out var num) && num == popped[j])
                {
                    stack.Pop();
                    j++;
                    continue;
                }

                stack.Push(pushed[i]);
                i++;
            }

            while (stack.Count > 0 && j < popped.Length)
            {
                if (stack.Peek() == popped[j])
                {
                    stack.Pop();
                }

                j++;
            }

            return stack.Count <= 0;
        }

        #endregion

        #region 面试题64. 求1+2+…+n

        //面试题64. 求1+2+…+n
        //https://leetcode-cn.com/problems/qiu-12n-lcof/
        public int SumNums(int n)
        {
            var num = 0;
            var flag = n > 1 && (num = SumNums(n - 1)) > 0;
            return num + n;
        }

        #endregion

        #region 137. 只出现一次的数字 II

        //137. 只出现一次的数字 II
        //https://leetcode-cn.com/problems/single-number-ii/
        public int SingleNumber(int[] nums)
        {
            return -1;
        }

        #endregion

        #region 560. 和为K的子数组

        //560. 和为K的子数组
        //https://leetcode-cn.com/problems/subarray-sum-equals-k/
        public int SubarraySum(int[] nums, int k)
        {
            int sum = 0, count = 0;
            var dic = new Dictionary<int, int> { { 0, 1 } };
            foreach (var n in nums)
            {
                sum += n;
                if (dic.TryGetValue(sum - k, out var num))
                {
                    count += num;
                }

                if (dic.ContainsKey(sum))
                {
                    dic[sum] += 1;
                }
                else
                {
                    dic[sum] = 1;
                }
            }

            return count;
        }

        #endregion

        #region 2. 两数相加/面试题 02.05. 链表求和

        //https://leetcode-cn.com/problems/add-two-numbers/
        //https://leetcode-cn.com/problems/sum-lists-lcci/
        public ListNode AddTwoNumbers(ListNode l1, ListNode l2)
        {
            var head = new ListNode(0);
            var root = head;
            while (l1 != null && l2 != null)
            {
                if (head.next == null)
                {
                    head.next = new ListNode(0);
                }

                head = head.next;
                head.val = head.val + l1.val + l2.val;
                if (head.val > 9)
                {
                    head.val = head.val - 10;
                    head.next = new ListNode(1);
                }

                l1 = l1.next;
                l2 = l2.next;
            }

            while (l1 != null)
            {
                if (head.next == null)
                {
                    head.next = new ListNode(0);
                }

                head = head.next;
                head.val += l1.val;
                if (head.val > 9)
                {
                    head.val = head.val - 10;
                    head.next = new ListNode(1);
                }

                l1 = l1.next;
            }

            while (l2 != null)
            {
                if (head.next == null)
                {
                    head.next = new ListNode(0);
                }

                head = head.next;
                head.val += l2.val;
                if (head.val > 9)
                {
                    head.val = head.val - 10;
                    head.next = new ListNode(1);
                }

                l2 = l2.next;
            }

            return root.next;
        }

        #endregion

        #region 19. 删除链表的倒数第N个节点

        //https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
        public ListNode RemoveNthFromEnd(ListNode head, int n)
        {
            ListNode node1 = head, node2 = head;
            while (node1 != null)
            {
                n--;
                if (n < -1)
                {
                    node2 = node2.next;
                }

                node1 = node1.next;
            }

            if (0 == n)
            {
                return head.next;
            }

            if (node2 != null && node2.next != null)
            {
                node2.next = node2.next.next;
            }

            return head;
        }

        #endregion

        #region 680. 验证回文字符串 Ⅱ

        //680. 验证回文字符串 Ⅱ
        //https://leetcode-cn.com/problems/valid-palindrome-ii/
        public bool ValidPalindrome(string s, int start, int end, bool flag)
        {
            if (start >= end)
            {
                return true;
            }

            while (start < end)
            {
                if (s[start] == s[end])
                {
                    start++;
                    end--;
                    continue;
                }


                if (flag)
                {
                    return false;
                }

                if (s[end - 1] == s[start])
                {
                    end--;
                    if (ValidPalindrome(s, start, end, true))
                    {
                        return true;
                    }
                }

                if (s[start + 1] == s[end])
                {
                    start++;
                    if (ValidPalindrome(s, start, end, true))
                    {
                        return true;
                    }
                }

                return false;
            }

            return true;
        }

        public bool ValidPalindrome(string s)
        {
            return ValidPalindrome(s, 0, s.Length - 1, false);
        }

        #endregion

        #region 1371. 每个元音包含偶数次的最长子字符串

        //1371. 每个元音包含偶数次的最长子字符串
        //https://leetcode-cn.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/

        //暴力解
        public int FindTheLongestSubstring(string s)
        {
            var max = 0;
            for (int i = 0; i < s.Length; i++)
            {
                var set = new Dictionary<char, int> { { 'a', 0 }, { 'e', 0 }, { 'i', 0 }, { 'o', 0 }, { 'u', 0 } };
                for (int j = i; j < s.Length; j++)
                {
                    if (set.TryGetValue(s[j], out var size))
                    {
                        set[s[j]] = size + 1;
                        ;
                    }

                    var flag = true;
                    foreach (var value in set.Values)
                    {
                        if ((value & 1) == 1)
                        {
                            flag = false;
                            break;
                        }
                    }

                    if (flag)
                    {
                        max = Math.Max(max, j - i + 1);
                    }
                }
            }

            return max;
        }

        //数组前缀和解法
        public int FindTheLongestSubstring1(string s)
        {
            int max = 0, mask = 0;
            //前缀和可能出现的情况共32种，1个字符只用奇数和偶数2种情况，共5个字符，共Math.Pow(2,5)种情况 声明数据记录每种情况最先出现的数组索引
            //前缀和只区分奇偶，奇数-奇数和偶数-偶数都是偶数，此种情况下同一种情况前缀和差的数组肯定符号条件
            //求最长数组，记录第一次的情况，每次符合条件求最大
            var states = new int[1 << 5];
            states[0] = -1;
            for (int i = 1; i < states.Length; i++)
            {
                states[i] = int.MaxValue; //
            }

            for (int i = 0; i < s.Length; i++)
            {
                switch (s[i])
                {
                    case 'a':
                        mask = mask ^ (1 << 0);
                        break;
                    case 'e':
                        mask = mask ^ (1 << 1);
                        break;
                    case 'i':
                        mask = mask ^ (1 << 2);
                        break;
                    case 'o':
                        mask = mask ^ (1 << 3);
                        break;
                    case 'u':
                        mask = mask ^ (1 << 4);
                        break;
                }

                if (states[mask] == int.MaxValue)
                {
                    states[mask] = i;
                }
                else
                {
                    max = Math.Max(max, i - states[mask] + 1);
                }
            }

            return max;
        }

        #endregion

        #region 面试题13. 机器人的运动范围

        public int MovingCount(int m, int n, int k)
        {
            int Compute(int num)
            {
                var res = 0;
                while (num != 0)
                {
                    res += (num % 10);
                    num /= 10;
                }

                return res;
            }

            var martix = new int[m, n];
            martix[0, 0] = -1;
            var size = 1;
            //不需要考虑回溯，坐标只需要向下或向右查找
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if ((i == 0 && j == 0) || Compute(i) + Compute(j) > k)
                    {
                        continue;
                    }

                    if (Compute(i) + Compute(j) <= k)
                    {
                        //计算上一步节点是否可达
                        if (i > 0 && martix[i - 1, j] == -1 || j > 0 && martix[i, j - 1] == -1)
                        {
                            martix[i, j] = -1;
                            size++;
                        }
                    }
                }
            }

            return size;
        }

        #endregion

        #region 面试题20. 表示数值的字符串

        public bool IsNumber(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return false;
            }

            s = s.Trim();
            if (s.Length <= 0)
            {
                return s.Length != 0 && char.IsDigit(s[0]);
            }

            var flag = 0;
            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                switch (ch)
                {
                    case '+':
                    case '-': //1
                        if (i == s.Length - 1 || (flag & 13) != 0 && (i <= 0 || s[i - 1] != 'e'))
                        {
                            return false;
                        }

                        flag |= 1;
                        break;
                    case 'e': //2
                        //e出现过或是最后一位或之前没有数字                        
                        if ((flag & 2) != 0 || (flag & 8) == 0 || i == s.Length - 1)
                        {
                            return false;
                        }

                        flag |= 2;
                        break;
                    case '.': //4
                        if ((flag & 6) != 0)
                        {
                            //之前./e已出现
                            return false;
                        }

                        flag |= 4;
                        //e可以出现，清除标记位
                        flag &= 13;
                        break;
                    default:
                        if (ch >= '0' && ch <= '9') //8
                        {
                            flag |= 9;
                        }
                        else
                        {
                            return false;
                        }

                        break;
                }
            }

            return (flag & 8) != 0;
        }

        #endregion

        #region 76. 最小覆盖子串

        //76. 最小覆盖子串
        //https://leetcode-cn.com/problems/minimum-window-substring/
        public string MinWindow(string s, string t)
        {
            if (string.IsNullOrEmpty(s) || s.Length < t.Length)
            {
                return string.Empty;
            }

            var dic = t.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
            int start = 0, end = 0, minStart = 0, minLen = int.MaxValue;
            while (end < s.Length)
            {
                var ch = s[end];
                if (dic.TryGetValue(ch, out var count))
                {
                    dic[ch] = count - 1;
                }

                while (start <= end && dic.All(kv => kv.Value <= 0))
                {
                    if (start == end)
                    {
                        return s.Substring(start, 1);
                    }

                    var len = end - start + 1;
                    if (len < minLen)
                    {
                        minStart = start;
                        minLen = len;
                    }

                    ch = s[start];
                    if (dic.ContainsKey(ch))
                    {
                        dic[ch]++;
                    }

                    start++;
                }

                end++;
            }

            return minLen <= s.Length ? s.Substring(minStart, minLen) : string.Empty;
        }

        #endregion

        #region 4. 寻找两个正序数组的中位数

        //4. 寻找两个正序数组的中位数
        //https://leetcode-cn.com/problems/median-of-two-sorted-arrays/

        int FindTopK(int[] nums1, int[] nums2, int k)
        {
            int length1 = nums1.Length, length2 = nums2.Length;
            int index1 = 0, index2 = 0;
            while (true)
            {
                // 边界情况
                if (index1 == length1)
                {
                    return nums2[index2 + k - 1];
                }

                if (index2 == length2)
                {
                    return nums1[index1 + k - 1];
                }

                if (k == 1)
                {
                    return Math.Min(nums1[index1], nums2[index2]);
                }

                // 正常情况
                int half = k / 2;
                int newIndex1 = Math.Min(index1 + half, length1) - 1;
                int newIndex2 = Math.Min(index2 + half, length2) - 1;
                int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
                if (pivot1 <= pivot2)
                {
                    k -= (newIndex1 - index1 + 1);
                    index1 = newIndex1 + 1;
                }
                else
                {
                    k -= (newIndex2 - index2 + 1);
                    index2 = newIndex2 + 1;
                }
            }
        }

        public double FindMedianSortedArrays(int[] nums1, int[] nums2)
        {
            var length = nums1.Length + nums2.Length;
            if ((length & 1) == 1)
            {
                return FindTopK(nums1, nums2, length / 2 + 1);
            }

            return (FindTopK(nums1, nums2, length / 2 + 1) + FindTopK(nums1, nums2, length / 2)) / 2.0d;
        }

        #endregion


        #region 287. 寻找重复数

        //287. 寻找重复数
        //https://leetcode-cn.com/problems/find-the-duplicate-number/
        public int FindDuplicate(int[] nums)
        {
            //数组取值范围为1-（nums.Length-1），所以中位数可以计算出来，统计nums中大于小于中位数的数量，即可判断数字的范围，逐步缩小范围直至找到结果
            var size = 0;
            int left = 1, right = nums.Length - 1;
            while (left < right)
            {
                var mid = (left + right) / 2; //边界范围中位数
                foreach (var num in nums)
                {
                    if (num <= mid)
                    {
                        size++;
                    }
                }

                if (size > mid) //小于mid的数字个数超出mid，说明该数在数组左区间
                {
                    right = mid;
                }
                else
                {
                    left = mid + 1;
                }

                size = 0;
            }

            return left;
        }

        #endregion

        #region 面试题37. 序列化二叉树

        //面试题37. 序列化二叉树
        //https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/

        public class Codec
        {
            // Encodes a tree to a single string.
            public string Serialize(TreeNode root)
            {
                if (root == null)
                {
                    return string.Empty;
                }

                var res = new StringBuilder();
                var queue = new Queue<TreeNode>();
                queue.Enqueue(root);
                while (queue.Count > 0)
                {
                    root = queue.Dequeue();
                    if (root == null)
                    {
                        res.Append("null");
                    }
                    else
                    {
                        res.Append(root.val.ToString());
                        queue.Enqueue(root.left);
                        queue.Enqueue(root.right);
                    }

                    res.Append(",");
                }

                return res.Remove(res.Length - 1, 1).ToString();
            }

            // Decodes your encoded data to tree.
            public TreeNode Deserialize(string data)
            {
                if (string.IsNullOrEmpty(data))
                {
                    return null;
                }

                var strs = data.Split(",");
                var strIndex = 0;
                var queue = new Queue<TreeNode>();
                var root = new TreeNode(int.Parse(strs[strIndex++]));
                queue.Enqueue(root);
                while (queue.TryDequeue(out var node))
                {
                    if (strs[strIndex] != "null")
                    {
                        node.left = new TreeNode(int.Parse(strs[strIndex]));
                        queue.Enqueue(node.left);
                    }

                    strIndex++;
                    if (strs[strIndex] != "null")
                    {
                        node.right = new TreeNode(int.Parse(strs[strIndex]));
                        queue.Enqueue(node.right);
                    }

                    strIndex++;
                }

                return root;
            }
        }

        #endregion

        #region 面试题43. 1～n整数中1出现的次数

        public int CountDigitOne(int n)
        {
            int high = n / 10, low = 0, cur = n % 10, digit = 1, res = 0;
            while (high != 0 || cur != 0)
            {
                if (cur == 0)
                {
                    res += high * digit;
                }
                else if (cur == 1)
                {
                    res += high * digit + low + 1;
                }
                else
                {
                    res += (high + 1) * digit;
                }

                low += cur * digit;
                digit *= 10;
                cur = high % 10;
                high /= 10;
            }


            return res;
        }

        #endregion


        #region 974. 和可被 K 整除的子数组

        //974. 和可被 K 整除的子数组
        //https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/
        //1.暴力解
        public int SubarraysDivByK(int[] nums, int k)
        {
            var res = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                var sum = 0;
                for (int j = i; j < nums.Length; j++)
                {
                    sum += nums[i];
                    if (sum % k == 0)
                    {
                        res++;
                    }
                }
            }

            return res;
        }

        //2.前缀和+余弦定理
        public int SubarraysDivByK1(int[] nums, int k)
        {
            int res = 0, sum = 0;
            var dic = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                sum += nums[i];
                var m = sum % k;
                dic.TryGetValue(m, out var count);
                res += count;
                dic[m] = count + 1;
            }

            return res;
        }

        #endregion

        #region 面试题46. 把数字翻译成字符串

        void TranslateNum(string num, int index, Dictionary<string, char> dict, ISet<string> result, IList<char> seqs)
        {
            if (index >= num.Length)
            {
                result.Add(new string(seqs.ToArray()));
                return;
            }

            for (int i = 1; i <= 2; i++)
            {
                if (index + i > num.Length)
                {
                    return;
                }

                var key = num.Substring(index, i);
                if (!dict.TryGetValue(key, out var ch))
                {
                    continue;
                }

                seqs.Add(ch);
                TranslateNum(num, index + i, dict, result, seqs);
                seqs.RemoveAt(seqs.Count - 1);
            }
        }

        public int TranslateNum(int num)
        {
            var dict = new Dictionary<string, char>();
            for (int i = 0; i < 26; i++)
            {
                dict[i.ToString()] = (char)('a' + i);
            }

            var strNum = num.ToString();
            var result = new HashSet<string>();
            TranslateNum(strNum, 0, dict, result, new List<char>());
            return result.Count;
        }

        #endregion

        #region 面试题47. 礼物的最大价值

        //面试题47. 礼物的最大价值
        //https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/
        int MaxValue(int[][] grid, int x, int y, int path)
        {
            if (x >= grid.Length)
            {
                y++;
                while (y < grid[0].Length)
                {
                    path += grid[grid.Length - 1][y];
                    y++;
                }

                return path;
            }

            if (y >= grid[0].Length)
            {
                x++;
                while (x < grid.Length)
                {
                    path += grid[x][grid[0].Length - 1];
                    x++;
                }

                return path;
            }

            path += grid[x][y];
            return Math.Max(MaxValue(grid, x + 1, y, path), MaxValue(grid, x, y + 1, path));
        }

        public int MaxValue(int[][] grid)
        {
            return MaxValue(grid, 0, 1, 0);
        }

        //动态规划，m[i,j]的最大值为max(m[i-1,j],m[i,j-1])+m
        public int MaxValue1(int[][] grid)
        {
            var max = new int[grid.Length, grid[0].Length];
            max[0, 0] = grid[0][0];
            for (int i = 0; i < grid.Length; i++)
            {
                for (int j = 0; j < grid[i].Length; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        max[i, j] = grid[i][j];
                    }
                    else if (i == 0)
                    {
                        max[i, j] = max[i, j - 1] + grid[i][j];
                    }
                    else if (j == 0)
                    {
                        max[i, j] = max[i - 1, j] + grid[i][j];
                    }
                    else
                    {
                        max[i, j] = Math.Max(max[i - 1, j], max[i, j - 1]) + grid[i][j];
                    }
                }
            }

            return max[grid.Length - 1, grid[0].Length - 1];
        }

        #endregion

        #region 394. 字符串解码

        //https://leetcode-cn.com/problems/decode-string/
        public string DecodeString(string s)
        {
            var stack = new Stack<string>();
            StringBuilder stb = new StringBuilder(), size = new StringBuilder();
            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i].ToString();
                if (ch == "]")
                {
                    while (stack.TryPop(out ch))
                    {
                        if (ch == "[") //[]内字符串拼接完成，将字符串压入栈中
                        {
                            while (stack.TryPeek(out ch) && ch.Length == 1 && ch[0] >= '0' && ch[0] <= '9')
                            {
                                size.Insert(0, stack.Pop());
                            }

                            var subStr = stb.ToString();
                            for (int j = 1; j < int.Parse(size.ToString()); j++)
                            {
                                stb.Append(subStr);
                            }

                            stack.Push(stb.ToString());
                            stb.Clear();
                            size.Clear();
                            break;
                        }

                        stb.Insert(0, ch);
                    }
                }
                else
                {
                    stack.Push(ch);
                }
            }

            while (stack.TryPop(out var ch))
            {
                stb.Insert(0, ch);
            }

            return stb.ToString();
        }

        public string DecodeString1(string s)
        {
            var strs = new Stack<string>();
            var numbers = new Stack<int>();
            var size = 0;
            StringBuilder stb = new StringBuilder(), res = new StringBuilder();
            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                if (char.IsDigit(ch))
                {
                    size = size * 10 + (ch - '0');
                }
                else if (ch == '[')
                {
                    numbers.Push(size);
                    strs.Push(stb.ToString());
                    stb.Clear();
                    size = 0;
                }
                else if (ch == ']')
                {
                    var num = numbers.Pop();
                    res.Append(strs.Pop());
                    for (int j = 0; j < num; j++)
                    {
                        res.Append(stb);
                    }

                    stb.Clear();
                    stb.Append(res);
                    res.Clear();
                }
                else
                {
                    stb.Append(ch);
                }
            }

            return stb.ToString();
        }

        #endregion

        #region 34. 在排序数组中查找元素的第一个和最后一个位置

        public int[] SearchRange(int[] nums, int target)
        {
            if (nums == null || nums.Length <= 0)
            {
                return new[] { -1, -1 };
            }

            int start = 0, end = nums.Length - 1;
            while (start <= end)
            {
                var mid = (start + end) / 2;
                if (nums[mid] <= target)
                {
                    start = mid + 1;
                }
                else
                {
                    end = mid - 1;
                }
            }

            //如果target存在，start==end时一定是target，此时满足条件start+1,end不变，故只需要判断nums[end]即可知target是否存在
            if (end < 0 || nums[end] != target)
            {
                return new[] { -1, -1 };
            }

            var rIndex = end;
            start = 0;
            while (start <= end)
            {
                var mid = (start + end) / 2;
                if (nums[mid] >= target)
                {
                    end = mid - 1;
                }
                else
                {
                    start = mid + 1;
                }
            }

            return new[] { start, rIndex };
        }

        #endregion

        #region 198. 打家劫舍

        //https://leetcode-cn.com/problems/house-robber/
        public int Rob(int[] nums)
        {
            if (nums == null || nums.Length == 0)
            {
                return 0;
            }

            if (nums.Length == 1)
            {
                return nums[0];
            }

            var dp = new int[nums.Length];
            dp[0] = nums[0];
            dp[1] = Math.Max(dp[0], nums[1]);
            for (int i = 2; i < dp.Length; i++)
            {
                dp[i] = Math.Max(dp[i - 1], dp[i - 2] + nums[i]);
            }

            return dp[dp.Length - 1];
        }

        #endregion

        #region 84. 柱状图中最大的矩形

        //https://leetcode-cn.com/problems/largest-rectangle-in-histogram/
        //暴力解
        public int LargestRectangleArea(int[] heights)
        {
            var max = 0;
            Dictionary<int, int> startDic = new Dictionary<int, int>(), endDic = new Dictionary<int, int>();
            for (int i = 0; i < heights.Length; i++)
            {
                int start = i - 1, end = i + 1;
                while (start >= 0 && heights[start] >= heights[i])
                {
                    if (startDic.TryGetValue(start, out var index))
                    {
                        start = index;
                    }
                    else
                    {
                        start--;
                    }
                }

                startDic[i] = start;
                while (end < heights.Length && heights[end] >= heights[i])
                {
                    if (endDic.TryGetValue(end, out var index))
                    {
                        end = index;
                    }
                    else
                    {
                        end++;
                    }
                }

                endDic[i] = end;
                max = Math.Max(max, (end - start - 1) * heights[i]);
            }

            return max;
        }

        public int LargestRectangleArea1(int[] heights)
        {
            var max = 0;
            int[] startDic = new int[heights.Length], endDic = new int[heights.Length];
            var stack = new Stack<int>();
            for (int i = 0; i < heights.Length; i++)
            {
                while (stack.Count > 0 && heights[stack.Peek()] >= heights[i])
                {
                    stack.Pop();
                }

                startDic[i] = stack.Count > 0 ? stack.Peek() : -1;
                stack.Push(i);
            }

            stack.Clear();
            for (int i = heights.Length - 1; i >= 0; i--)
            {
                while (stack.Count > 0 && heights[stack.Peek()] >= heights[i])
                {
                    stack.Pop();
                }

                endDic[i] = stack.Count > 0 ? stack.Peek() : heights.Length;
                stack.Push(i);
            }

            for (int i = 0; i < heights.Length; i++)
            {
                max = Math.Max((endDic[i] - startDic[i] - 1) * heights[i], max);
            }

            return max;
        }

        #endregion

        #region 面试题45. 把数组排成最小的数

        //https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/
        public string MinNumber(int[] nums)
        {
            var strs = new string[nums.Length];
            for (int i = 0; i < nums.Length; i++)
            {
                strs[i] = nums[i].ToString();
            }

            Array.Sort(strs, (s1, s2) =>
            {
                string c1 = s1 + s2, c2 = s2 + s1;
                return c1.CompareTo(c2);
            });
            var result = new StringBuilder();
            foreach (var str in strs)
            {
                result.Append(str);
            }

            return result.ToString();
        }

        #endregion

        #region 面试题44. 数字序列中某一位的数字

        //https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/
        //range 0-9     9
        //range 10-99   189
        //range 100-999 191+2700=2891
        public int FindNthDigit(int n)
        {
            if (n < 10)
            {
                return n;
            }

            long start = 10, end = 189;
            int number = 10, len = 2;
            while (true)
            {
                if (n >= start && n <= end)
                {
                    int count = (int)(n - start), index = count % len;
                    var num = (number + (count / len)).ToString();
                    return num[index] - '0';
                }

                len++;
                start = end + 1;
                end = 9 * (long)Math.Pow(10, len - 1) * len + end;
                number *= 10;
            }
        }

        #endregion

        #region 1431. 拥有最多糖果的孩子

        //https://leetcode-cn.com/problems/kids-with-the-greatest-number-of-candies/
        public IList<bool> KidsWithCandies(int[] candies, int extraCandies)
        {
            var result = new bool[candies.Length];
            var max = candies.Max();
            for (int i = 0; i < candies.Length; i++)
            {
                result[i] = candies[i] + extraCandies >= max;
            }

            return result;
        }

        #endregion

        #region 面试题49. 丑数

        //https://leetcode-cn.com/problems/chou-shu-lcof/
        public int NthUglyNumber1(int n)
        {
            if (n <= 5)
            {
                return n;
            }

            bool IsUgly(int num)
            {
                while (num % 2 == 0)
                {
                    num /= 2;
                }

                while (num % 3 == 0)
                {
                    num /= 3;
                }

                while (num % 5 == 0)
                {
                    num /= 5;
                }

                return num == 1;
            }

            var res = 0;
            while (n > 0)
            {
                res++;
                if (IsUgly(res))
                {
                    n--;
                }
            }

            return res;
        }

        public int NthUglyNumber(int n)
        {
            if (n <= 5)
            {
                return n;
            }

            var dp = new int[n];
            dp[0] = 1;
            int num2 = 0, num3 = 0, num5 = 0;
            for (int i = 1; i < n; i++)
            {
                dp[i] = Math.Min(Math.Min(dp[num2] * 2, dp[num3] * 3), dp[num5] * 5);
                while (dp[num2] * 2 <= dp[i])
                {
                    num2++;
                }

                while (dp[num3] * 3 <= dp[i])
                {
                    num3++;
                }

                while (dp[num5] * 5 <= dp[i])
                {
                    num5++;
                }
            }

            return dp[n - 1];
        }

        #endregion

        #region 面试题51. 数组中的逆序对

        //https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/
        //暴力解
        public int ReversePairs(int[] nums)
        {
            var res = 0;
            for (int i = 0; i < nums.Length - 1; i++)
            {
                for (int j = i + 1; j < nums.Length; j++)
                {
                    if (nums[i] > nums[j])
                    {
                        res++;
                    }
                }
            }

            return res;
        }

        int Merge(int[] nums, int[] temp, int start, int mid, int end)
        {
            int i1 = start, i2 = mid + 1, i = 0, size = 0, total = 0;
            while (i1 <= mid && i2 <= end)
            {
                if (nums[i1] <= nums[i2])
                {
                    temp[i++] = nums[i1];
                    i1++;
                    total += size;
                }
                else
                {
                    size++;
                    temp[i++] = nums[i2];
                    i2++;
                }
            }

            while (i1 <= mid)
            {
                //此时说明前半段数组剩余数值大于后半段所有数值，应该进行计数
                temp[i++] = nums[i1++];
                total += (end - mid);
            }

            while (i2 <= end)
            {
                temp[i++] = nums[i2++];
            }

            for (int j = 0; j < i; j++)
            {
                nums[start++] = temp[j];
            }

            return total;
        }

        int MergeSort(int[] nums, int[] temp, int start, int end)
        {
            if (start >= end)
            {
                return 0;
            }

            var mid = (start + end) / 2;
            var left = MergeSort(nums, temp, start, mid);
            var right = MergeSort(nums, temp, mid + 1, end);
            if (nums[mid] <= nums[mid + 1])
            {
                return left + right;
            }

            var current = Merge(nums, temp, start, mid, end);
            return left + right + current;
        }

        //归并排序
        public int ReversePairs1(int[] nums)
        {
            var temp = new int[nums.Length];
            return MergeSort(nums, temp, 0, nums.Length - 1);
        }

        #endregion

        #region 240. 搜索二维矩阵 II

        //https://leetcode-cn.com/problems/search-a-2d-matrix-ii/
        public bool SearchMatrix(int[,] matrix, int target)
        {
            int x = 0, y = matrix.GetLength(1) - 1, xlen = matrix.GetLength(0);
            while (x < xlen && y >= 0)
            {
                if (matrix[x, y] == target)
                {
                    return true;
                }

                if (matrix[x, y] < target)
                {
                    x++;
                }
                else
                {
                    y--;
                }
            }

            return false;
        }

        #endregion

        #region 88. 合并两个有序数组/面试题 10.01. 合并排序的数组

        //https://leetcode-cn.com/problems/merge-sorted-array/
        //https://leetcode-cn.com/problems/sorted-merge-lcci/
        //插入排序
        public void MergeByInsertSort(int[] nums1, int m, int[] nums2, int n)
        {
            for (int i = 0; i < n; i++)
            {
                var index = m - 1;
                while (index >= 0)
                {
                    if (nums2[i] < nums1[index])
                    {
                        nums1[index + 1] = nums1[index];
                    }
                    else
                    {
                        break;
                    }

                    index--;
                }

                nums1[index + 1] = nums2[i];
                m++;
            }
        }

        //从后往前
        public void Merge(int[] nums1, int m, int[] nums2, int n)
        {
            int i1 = m - 1, i2 = n - 1, index = nums1.Length - 1;
            while (i1 >= 0 && i2 >= 0)
            {
                if (nums1[i1] >= nums2[i2])
                {
                    nums1[index--] = nums1[i1--];
                }
                else
                {
                    nums1[index--] = nums2[i2--];
                }
            }

            while (i2 >= 0)
            {
                nums1[index--] = nums2[i2--];
            }
        }

        #endregion

        #region 977. 有序数组的平方

        //https://leetcode-cn.com/problems/squares-of-a-sorted-array/
        public int[] SortedSquares(int[] nums)
        {
            var result = new int[nums.Length];
            int start = 0, end = nums.Length - 1, index = result.Length - 1;
            while (start < end)
            {
                if (nums[start] >= 0)
                {
                    break;
                }

                if (Math.Abs(nums[start]) >= Math.Abs(nums[end]))
                {
                    result[index--] = nums[start] * nums[start];
                    start++;
                }
                else
                {
                    result[index--] = nums[end] * nums[end];
                    end--;
                }
            }

            while (start <= end)
            {
                result[index] = nums[end] * nums[end];
                index--;
                end--;
            }

            return result;
        }

        #endregion

        #region 面试题 08.06. 汉诺塔问题

        //https://leetcode-cn.com/problems/hanota-lcci/
        //n = 1 时，直接把盘子从 A 移到 C；
        //n > 1 时，
        //先把上面 n - 1 个盘子从 A 移到 B（子问题，递归）；
        //再将最大的盘子从 A 移到 C；
        //再将 B 上 n - 1 个盘子从 B 移到 C（子问题，递归）。

        public void Move(int n, IList<int> a, IList<int> b, IList<int> c)
        {
            if (n == 1)
            {
                c.Add(a[a.Count - 1]);
                a.RemoveAt(a.Count - 1);
                return;
            }

            Move(n - 1, a, c, b);
            c.Add(a[a.Count - 1]);
            a.RemoveAt(a.Count - 1);
            Move(n - 1, b, a, c);
        }

        public void Hanota(IList<int> a, IList<int> b, IList<int> c)
        {
            if (a.Count <= 0 && b.Count <= 0)
            {
                return;
            }

            Move(a.Count, a, b, c);
        }

        #endregion

        #region 837. 新21点

        //https://leetcode-cn.com/problems/new-21-game/
        //w=10 n=21 k=17
        public double New21GameOmp(int n, int k, int w)
        {
            if (k == 0)
            {
                return 1.0;
            }

            var dp = new double[k + 1 + w];
            for (int i = k; i <= Math.Min(n, k + w - 1); i++)
            {
                dp[i] = 1.0;
            }

            dp[k - 1] = 1.0 * Math.Min(w, n - k + 1) / w;
            for (int i = k - 2; i >= 0; i--)
            {
                dp[i] = dp[i + 1] - (dp[i + 1 + w] - dp[i + 1]) / w;
            }

            return dp[0];
        }

        public double New21Game(int n, int k, int w)
        {
            if (k == 0)
            {
                return 1.0;
            }

            var dp = new double[k + 1 + w];
            for (int i = k; i <= Math.Min(n, k + w - 1); i++)
            {
                dp[i] = 1.0;
            }

            dp[k - 1] = 1.0 * Math.Min(w, n - k + 1) / w;
            for (int i = k - 1; i >= 0; i--)
            {
                for (int j = 1; j <= w; j++)
                {
                    dp[i] += dp[i + j] / w;
                }
            }

            return dp[0];
        }

        #endregion

        #region 面试题65. 不用加减乘除做加法

        //https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/

        public int Add(int a, int b)
        {
            //1 a&b 求出相同位是1的数
            //2 左移1位，相同位1相加结果进1
            //3 a^b 求出不同位数
            //4 此时(a&b<<1)+(a^b)即为所求的结果，跳转步骤1（不考虑溢出及负数问题）
            // return a;

            while (true)
            {
                var n = (a & b) << 1;
                a = a ^ b;
                b = n; //没有进位，说明和只和非进位有关系
                if (b == 0)
                {
                    return a;
                }
            }
        }

        #endregion

        #region 581. 最短无序连续子数组

        //https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/
        public int FindUnsortedSubarray(int[] nums)
        {
            var copy = new int[nums.Length];
            Array.Copy(nums, copy, copy.Length);
            Array.Sort(copy);
            int start = 0, end = nums.Length - 1;
            while (start < end)
            {
                if (nums[start] == copy[start])
                {
                    start++;
                }
                else if (nums[end] == copy[end])
                {
                    end--;
                }
                else
                {
                    return end - start + 1;
                }
            }

            return 0;
        }

        #endregion

        #region 83. 删除排序链表中的重复元素

        //https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/
        public ListNode DeleteDuplicates(ListNode head)
        {
            if (head == null)
            {
                return null;
            }

            ListNode prev = head, current = head.next;
            while (current != null)
            {
                if (prev.val == current.val)
                {
                    prev.next = current.next;
                }
                else
                {
                    prev = current;
                }

                current = current.next;
            }

            return head;
        }

        #endregion

        #region 11. 盛最多水的容器

        //https://leetcode-cn.com/problems/container-with-most-water/
        //暴力解
        public int MaxArea(int[] height)
        {
            var max = 0;
            for (int i = 0; i < height.Length - 1; i++)
            {
                for (int j = i + 1; j < height.Length; j++)
                {
                    var h = Math.Min(height[i], height[j]);
                    max = Math.Max(max, (j - i) * h);
                }
            }

            return max;
        }

        //双指针
        public int MaxArea1(int[] height)
        {
            int max = 0, start = 0, end = height.Length - 1;
            while (start < end)
            {
                var h = Math.Min(height[start], height[end]);
                max = Math.Max(max, (end - start) * h);
                if (height[start] <= height[end])
                {
                    start++;
                }
                else
                {
                    end--;
                }
            }

            return max;
        }

        #endregion

        #region 22. 括号生成/面试题 08.09. 括号

        //https://leetcode-cn.com/problems/generate-parentheses/
        //https://leetcode-cn.com/problems/bracket-lcci/

        void GenerateParenthesis(int left, int right, IList<string> result, StringBuilder str)
        {
            if (left <= 0 && right <= 0)
            {
                var it = str.ToString();
                result.Add(it);
                return;
            }

            if (left > 0)
            {
                str.Append("(");
                GenerateParenthesis(left - 1, right, result, str);
                str.Remove(str.Length - 1, 1);
            }

            if (right > 0 && left < right)
            {
                str.Append(")");
                GenerateParenthesis(left, right - 1, result, str);
                str.Remove(str.Length - 1, 1);
            }
        }

        public IList<string> GenerateParenthesis(int n)
        {
            var result = new List<string>();
            GenerateParenthesis(n, n, result, new StringBuilder());
            return result;
        }

        #endregion

        #region 23. 合并K个排序链表

        //https://leetcode-cn.com/problems/merge-k-sorted-lists/
        ListNode MergeKLists(ListNode[] lists, int start, int end)
        {
            if (start >= end)
            {
                return lists[start];
            }

            var mid = (start + end) / 2;
            var node1 = MergeKLists(lists, start, mid);
            var node2 = MergeKLists(lists, mid + 1, end);
            ListNode head = new ListNode(0), node = head;
            while (node1 != null && node2 != null)
            {
                if (node1.val <= node2.val)
                {
                    node.next = new ListNode(node1.val);
                    node1 = node1.next;
                }
                else
                {
                    node.next = new ListNode(node2.val);
                    node2 = node2.next;
                }

                node = node.next;
            }

            if (node1 != null)
            {
                node.next = node1;
            }

            if (node2 != null)
            {
                node.next = node2;
            }

            return head.next;
        }

        public ListNode MergeKLists(ListNode[] lists)
        {
            if (lists.Length <= 0)
            {
                return null;
            }

            return MergeKLists(lists, 0, lists.Length - 1);
        }

        #endregion

        #region 238. 除自身以外数组的乘积

        //https://leetcode-cn.com/problems/product-of-array-except-self/
        public int[] ProductExceptSelf(int[] nums)
        {
            int[] prev = new int[nums.Length], next = new int[nums.Length];
            for (int i = 0, j = nums.Length - 1; i < nums.Length; i++, j--)
            {
                if (i <= 1)
                {
                    prev[i] = i == 0 ? 1 : nums[i - 1];
                    next[j] = i == 0 ? 1 : nums[j + 1];
                }
                else
                {
                    prev[i] = nums[i - 1] * prev[i - 1];
                    next[j] = nums[j + 1] * next[j + 1];
                }
            }

            var result = new int[nums.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = prev[i] * next[i];
            }

            return result;
        }

        #endregion

        #region 93. 复原IP地址

        //https://leetcode-cn.com/problems/restore-ip-addresses/
        bool IsVaildIP(IList<string> strs)
        {
            if (strs.Count != 4)
            {
                return false;
            }

            foreach (var str in strs)
            {
                if (str.Length > 1 && str[0] == '0')
                {
                    return false;
                }

                var num = int.Parse(str);
                if (num < 0 || num > 255)
                {
                    return false;
                }
            }

            return true;
        }

        void RestoreIpAddresses(string s, int index, IList<string> result, List<string> sub, int limit)
        {
            var len = s.Length - index;
            if (len > limit)
            {
                return;
            }

            limit -= 3;
            if (index >= s.Length || sub.Count >= 4)
            {
                if (index == s.Length && IsVaildIP(sub))
                {
                    result.Add(string.Join(".", sub));
                }

                return;
            }

            for (int i = 1; i <= 3; i++)
            {
                if (index + i > s.Length)
                {
                    break;
                }

                sub.Add(s.Substring(index, i));
                RestoreIpAddresses(s, index + i, result, sub, limit);
                sub.RemoveAt(sub.Count - 1);
            }
        }

        public IList<string> RestoreIpAddresses(string s)
        {
            var result = new List<string>();
            RestoreIpAddresses(s, 0, result, new List<string>(), 12);
            return result;
        }

        #endregion

        #region 189. 旋转数组

        //https://leetcode-cn.com/problems/rotate-array/
        public void Rotate(int[] nums, int k)
        {
            k = k % nums.Length;
            var temp = new int[k];
            Array.Copy(nums, nums.Length - k, temp, 0, k);
            Array.Copy(nums, 0, nums, k, nums.Length - k);
            Array.Copy(temp, 0, nums, 0, k);
        }

        public void RotateByReverse(int[] nums, int k)
        {
            k = k % nums.Length;
            Array.Reverse(nums);
            Array.Reverse(nums, 0, k);
            Array.Reverse(nums, k, nums.Length - k);
        }

        public void RotateByRoll(int[] nums, int k)
        {
            k = k % nums.Length;
            var count = 0;
            //需要移动数组中所有数字，所以当count==nums.length时说明已经完成
            for (int i = 0; count < nums.Length; i++)
            {
                var current = i;
                var mv = nums[current];
                do
                {
                    var next = (current + k) % nums.Length;
                    var tmp = nums[next];
                    nums[next] = mv;
                    mv = tmp;
                    current = next;
                    count++;
                } while (current != i);
            }
        }

        #endregion

        #region 15. 三数之和

        //https://leetcode-cn.com/problems/3sum/
        void ThreeSum(int index, int[] nums, IList<IList<int>> result, IList<int> sub)
        {
            if (sub.Count == 3 && sub.Sum() == 0)
            {
                result.Add(sub.OrderBy(n => n).ToArray());
                return;
            }

            if (index >= nums.Length)
            {
                return;
            }

            for (int i = index; i < nums.Length - 1; i++)
            {
                sub.Add(nums[i]);
                ThreeSum(i + 1, nums, result, sub);
                sub.RemoveAt(sub.Count - 1);
            }
        }

        public IList<IList<int>> ThreeSum(int[] nums)
        {
            var result = new List<IList<int>>();
            ThreeSum(0, nums, result, new List<int>());
            return result;
        }

        public IList<IList<int>> ThreeSum1(int[] nums)
        {
            var result = new List<IList<int>>();
            Array.Sort(nums);
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] > 0 || nums[nums.Length - i - 1] < 0)
                {
                    break;
                }

                if (i > 0 && nums[i] == nums[i - 1])
                {
                    continue;
                }

                int start = i + 1, end = nums.Length - 1;
                while (start < end)
                {
                    var num = nums[i] + nums[start] + nums[end];
                    if (num == 0)
                    {
                        result.Add(new[] { nums[i], nums[start], nums[end] });
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
                    else if (num < 0)
                    {
                        start++;
                    }
                    else
                    {
                        end--;
                    }
                }
            }

            return result;
        }

        #endregion

        #region 36. 有效的数独

        //https://leetcode-cn.com/problems/valid-sudoku/
        public bool IsValidSudoku(char[][] board)
        {
            bool[,] rows = new bool[board.Length, board[0].Length], cols = new bool[board.Length, board[0].Length];
            var martix = new bool[3, 3][];
            for (int i = 0; i < board.Length; i++)
            {
                for (int j = 0; j < board[0].Length; j++)
                {
                    if (board[i][j] == '.')
                    {
                        continue;
                    }

                    var n = board[i][j] - '1';
                    if (rows[i, n] || cols[j, n])
                    {
                        return false;
                    }

                    int rIndex = i / 3, cIndex = j / 3;
                    if (martix[rIndex, cIndex] == null)
                    {
                        martix[rIndex, cIndex] = new bool[9];
                    }
                    else if (martix[rIndex, cIndex][n])
                    {
                        return false;
                    }

                    martix[rIndex, cIndex][n] = true;
                    rows[i, n] = cols[j, n] = true;
                }
            }

            return true;
        }

        #endregion

        #region 41. 缺失的第一个正数

        //https://leetcode-cn.com/problems/first-missing-positive/
        public int FirstMissingPositive(int[] nums)
        {
            int min = int.MaxValue, max = 0;
            var set = new HashSet<int>();
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] <= 0)
                {
                    continue;
                }

                set.Add(nums[i]);
                min = Math.Min(min, nums[i]);
                max = Math.Max(max, nums[i]);
            }

            if (min > 1)
            {
                return 1;
            }

            while (min < max)
            {
                if (!set.Contains(min))
                {
                    return min;
                }

                min++;
            }

            return min + 1;
        }

        #endregion

        #region 46. 全排列

        //https://leetcode-cn.com/problems/permutations/
        void Swap(int[] nums, int i, int j)
        {
            if (i == j)
            {
                return;
            }

            var tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }

        void Permute(int[] nums, int index, IList<IList<int>> result)
        {
            if (index >= nums.Length)
            {
                result.Add(new List<int>(nums));
                return;
            }

            for (int i = index; i < nums.Length; i++)
            {
                Swap(nums, index, i);
                Permute(nums, index + 1, result);
                Swap(nums, index, i);
            }
        }

        public IList<IList<int>> Permute(int[] nums)
        {
            var result = new List<IList<int>>();
            Permute(nums, 0, result);
            return result;
        }

        #endregion

        #region 49. 字母异位词分组

        //https://leetcode-cn.com/problems/group-anagrams/
        public IList<IList<string>> GroupAnagrams(string[] list)
        {
            var dic = new Dictionary<string, IList<string>>();
            var result = new List<IList<string>>();
            foreach (var str in list)
            {
                var chars = str.ToCharArray();
                Array.Sort(chars);
                var key = new string(chars);
                if (!dic.TryGetValue(key, out var items))
                {
                    dic[key] = items = new List<string>();
                    result.Add(items);
                }

                items.Add(str);
            }

            return result;
        }

        #endregion

        #region 55. 跳跃游戏

        //https://leetcode-cn.com/problems/jump-game/
        public bool CanJump(int[] nums)
        {
            if (nums.Length <= 1)
            {
                return true;
            }

            var most = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                if (i <= most)
                {
                    most = Math.Max(most, i + nums[i]);
                    if (most >= nums.Length - 1)
                    {
                        return true;
                    }
                }
            }

            return false;
            //3 0 1 0 1
            // var flags = new bool[nums.Length];
            // for (int i = nums.Length - 2; i >= 0; i--)
            // {
            //     if (nums[i] >= 1)
            //     {
            //         if (flags[i + 1])
            //         {
            //             flags[i] = true;
            //         }
            //         else
            //         {
            //             for (int j = i + 1; j <= Math.Min(i + nums[i], nums.Length - 1); j++)
            //             {
            //                 if (flags[j])
            //                 {
            //                     flags[i] = true;
            //                     break;
            //                 }
            //             }
            //         }
            //     }
            // }
            //
            // return flags[0];
        }

        #endregion

        #region 56. 合并区间

        //https://leetcode-cn.com/problems/merge-intervals/
        public int[][] Merge(int[][] intervals)
        {
            if (intervals.Length <= 0)
            {
                return intervals;
            }

            var result = new List<int[]>();
            Array.Sort(intervals, (n1, n2) => n1[0] - n2[0]);
            result.Add(intervals[0]);
            for (int i = 1; i < intervals.Length; i++)
            {
                int[] n1 = result[result.Count - 1], n2 = intervals[i];
                if (n1[1] >= n2[0]) //两个数组重叠
                {
                    n1[1] = Math.Max(n2[1], n1[1]);
                }
                else //没有重叠
                {
                    result.Add(n2);
                }
            }

            return result.ToArray();
        }

        #endregion

        #region 62. 不同路径

        //https://leetcode-cn.com/problems/unique-paths/
        //dp[i,j]=dp[i+1,j]+dp[i,j+1]
        public int UniquePaths(int m, int n)
        {
            var dp = new int[m, n];
            for (int i = 0; i < n - 1; i++)
            {
                dp[m - 1, i] = 1;
            }

            for (int i = 0; i < m - 1; i++)
            {
                dp[i, n - 1] = 1;
            }

            for (int i = m - 2; i >= 0; i--)
            {
                for (int j = n - 2; j >= 0; j--)
                {
                    dp[i, j] = dp[i + 1, j] + dp[i, j + 1];
                }
            }

            return dp[0, 0];
        }

        #endregion

        #region 66. 加一

        //https://leetcode-cn.com/problems/plus-one/
        public int[] PlusOne(int[] digits)
        {
            for (int i = digits.Length - 1; i >= 0; i--)
            {
                var num = digits[i] + 1;
                if (num > 9)
                {
                    digits[i] = 0;
                    if (i == 0)
                    {
                        var result = new int[digits.Length + 1];
                        result[0] = 1;
                        Array.Copy(digits, 0, result, 1, digits.Length);
                        return result;
                    }
                }
                else
                {
                    digits[i] = num;
                    break;
                }
            }

            return digits;
        }

        #endregion

        #region 70. 爬楼梯

        //https://leetcode-cn.com/problems/climbing-stairs/
        public int ClimbStairs(int n)
        {
            if (n == 1)
            {
                return 1;
            }

            if (n == 2)
            {
                return 2;
            }

            var nums = new int[n];
            nums[0] = 1;
            nums[1] = 2;
            for (int i = 2; i < n; i++)
            {
                nums[i] = nums[i - 1] + nums[i - 2];
            }

            return nums[n - 1];
        }

        public int ClimbStairsN(int n)
        {
            if (n <= 2)
            {
                return n;
            }

            int one = 1, two = 2, three = 0;
            for (int i = 2; i < n; i++)
            {
                three = one + two;
                one = two;
                two = three;
            }

            return three;
        }

        #endregion

        #region 73. 矩阵置零/面试题 01.08. 零矩阵

        //https://leetcode-cn.com/problems/set-matrix-zeroes/
        //https://leetcode-cn.com/problems/zero-matrix-lcci/
        public void SetZeroes(int[][] matrix)
        {
            bool[] rows = new bool[matrix.Length], cols = new bool[matrix[0].Length];
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    if (matrix[i][j] == 0)
                    {
                        rows[i] = true;
                        cols[j] = true;
                    }
                }
            }

            for (int i = 0; i < rows.Length; i++)
            {
                if (rows[i])
                {
                    for (int j = 0; j < matrix[0].Length; j++)
                    {
                        matrix[i][j] = 0;
                    }
                }
            }

            for (int i = 0; i < cols.Length; i++)
            {
                if (cols[i])
                {
                    for (int j = 0; j < matrix.Length; j++)
                    {
                        matrix[j][i] = 0;
                    }
                }
            }
        }

        public void SetZeroesByLeetcode(int[][] matrix)
        {
            bool col0 = false, row0 = false;
            for (int i = 0; i < matrix.Length && !col0; i++)
            {
                col0 = matrix[i][0] == 0;
            }
            for (int i = 0; i < matrix[0].Length && !row0; i++)
            {
                row0 = matrix[0][i] == 0;
            }
            for (int i = 1; i < matrix.Length; i++)
            {
                for (int j = 1; j < matrix[i].Length; j++)
                {
                    if (matrix[i][j] == 0)
                    {
                        matrix[0][j] = matrix[i][0] = 0;
                    }
                }
            }

            for (int i = 1; i < matrix.Length; i++)
            {
                for (int j = 1; j < matrix[i].Length; j++)
                {
                    if (matrix[0][j] == 0 || matrix[i][0] == 0)
                    {
                        matrix[i][j] = 0;
                    }
                }
            }
            if (col0)
            {
                for (int i = 0; i < matrix.Length; i++)
                {
                    matrix[i][0] = 0;
                }
            }
            if (row0)
            {
                for (int i = 0; i < matrix[0].Length; i++)
                {
                    matrix[0][i] = 0;
                }
            }
        }
        #endregion

        #region 75. 颜色分类

        //https://leetcode-cn.com/problems/sort-colors/
        public void SortColors(int[] nums)
        {
            var sort = new int[3];
            for (var i = 0; i < nums.Length; i++)
            {
                sort[nums[i]]++;
            }

            for (int i = 0, j = 0; i < sort.Length; i++)
            {
                for (var n = 0; n < sort[i]; n++)
                {
                    nums[j++] = i;
                }
            }
        }

        #endregion

        #region 131. 分割回文串

        //https://leetcode-cn.com/problems/palindrome-partitioning/
        bool IsLoop(IList<char> str)
        {
            if (str.Count == 1)
            {
                return true;
            }

            int start = 0, end = str.Count - 1;
            while (start < end)
            {
                if (str[start] != str[end])
                {
                    return false;
                }

                start++;
                end--;
            }

            return true;
        }

        void Partition(char[] s, int index, IList<IList<string>> result, List<string> items)
        {
            if (index >= s.Length && items.Count > 0)
            {
                result.Add(items.ToArray());
                return;
            }

            var sub = new List<char>();
            for (int i = index; i < s.Length; i++)
            {
                sub.Add(s[i]);
                if (!IsLoop(sub))
                {
                    continue;
                }

                items.Add(new string(sub.ToArray()));
                Partition(s, i + 1, result, items);
                items.RemoveAt(items.Count - 1);
            }
        }

        public IList<IList<string>> Partition(string s)
        {
            var result = new List<IList<string>>();
            var chars = s.ToCharArray();
            Partition(chars, 0, result, new List<string>());
            return result;
        }

        #endregion

        #region 140. 单词拆分 II

        //https://leetcode-cn.com/problems/word-break-ii/

        //暴力解
        void WordBreak(string s, int index, int min, int max, HashSet<string> wordDict, IList<string> result,
            IList<string> stb)
        {
            if (index >= s.Length)
            {
                result.Add(string.Join(' ', stb));
                return;
            }

            for (int len = min; len <= max; len++)
            {
                if (index + len > s.Length)
                {
                    return;
                }

                var key = s.Substring(index, len);
                if (wordDict.Contains(key))
                {
                    stb.Add(key);
                    WordBreak(s, index + len, min, max, wordDict, result, stb);
                    stb.RemoveAt(stb.Count - 1);
                }
            }
        }

        public IList<string> WordBreak(string s, IList<string> wordDict)
        {
            var dict = new HashSet<string>();
            int min = int.MaxValue, max = int.MinValue;
            foreach (var word in wordDict)
            {
                dict.Add(word);
                min = Math.Min(word.Length, min);
                max = Math.Max(word.Length, max);
            }

            var result = new List<string>();
            WordBreak(s, 0, min, max, dict, result, new List<string>());
            return result;
        }

        //动态规划
        IList<string> WordBreakII(string s, int index, int min, int max, HashSet<string> wordDict,
            Dictionary<int, IList<string>> nexts)
        {
            if (nexts.TryGetValue(index, out var list))
            {
                return list;
            }

            list = new List<string>();
            if (index >= s.Length)
            {
                list.Add("");
            }
            else
            {
                for (int len = min; len <= max; len++)
                {
                    if (index + len > s.Length)
                    {
                        break;
                    }

                    var key = s.Substring(index, len);
                    if (wordDict.Contains(key))
                    {
                        var seqs = WordBreakII(s, index + len, min, max, wordDict, nexts);
                        foreach (var seq in seqs)
                        {
                            list.Add(string.IsNullOrEmpty(seq) ? key : key + " " + seq);
                        }
                    }
                }
            }

            nexts[index] = list;
            return list;
        }

        public IList<string> WordBreakII(string s, IList<string> wordDict)
        {
            var dict = new HashSet<string>();
            int min = int.MaxValue, max = int.MinValue;
            foreach (var word in wordDict)
            {
                dict.Add(word);
                min = Math.Min(word.Length, min);
                max = Math.Max(word.Length, max);
            }

            return WordBreakII(s, 0, min, max, dict, new Dictionary<int, IList<string>>());
        }

        #endregion

        #region 91. 解码方法

        //https://leetcode-cn.com/problems/decode-ways/
        void NumDecodings(string s, int index, ISet<string> result, Dictionary<string, char> dict, List<char> seqs)
        {
            if (index >= s.Length)
            {
                result.Add(new string(seqs.ToArray()));
                return;
            }

            for (int i = 1; i <= 2; i++)
            {
                if (index + i > s.Length)
                {
                    return;
                }

                var key = s.Substring(index, i);
                if (dict.TryGetValue(key, out var seq))
                {
                    seqs.Add(seq);
                    NumDecodings(s, index + i, result, dict, seqs);
                    seqs.RemoveAt(seqs.Count - 1);
                }
            }
        }

        public int NumDecodings(string s)
        {
            if (s.Length <= 0 || s[0] == '0')
            {
                return 0;
            }

            var dict = new Dictionary<string, char>();
            for (int i = 1; i <= 26; i++)
            {
                dict.Add(i.ToString(), (char)('A' + i - 1));
            }

            var strs = new HashSet<string>();
            NumDecodings(s, 0, strs, dict, new List<char>());
            return strs.Count;
            //dp[i]=dp[i-1]||dp[i-1]+dp[i-2]?
            //12 [1,2] [12]
            //128 [1,2,8] [12,8]
            //1281[1,2,8,1][12,8,1]
            //12811[]
            //121 [1,2,1][12,1][1,21]
            //1211 [1,2,1,1][1,2,11][12,1,1][1,21,1][12,11]
        }

        public int NumDecodingsDynamic(string s)
        {
            //dp[i]=dp[i-1]||dp[i-1]+dp[i-2]?
            //      0 
            //12 [1,2] [12]
            //120 [1,20]
            //128 [1,2,8] [12,8]
            //1281[1,2,8,1][12,8,1]
            //12811[]
            //121 [1,2,1][12,1][1,21]
            //1211 [1,2,1,1][1,2,11][12,1,1][1,21,1][12,11]
            if (s.Length <= 0 || s[0] == '0')
            {
                return 0;
            }

            var dp = new int[s.Length];
            dp[0] = 1;
            for (int i = 1; i < s.Length; i++)
            {
                var ch = s[i];
                var pre = int.Parse(s.Substring(i - 1, 2));
                if (ch == '0')
                {
                    //121 [1,2,1] [12,1][1,21]
                    //1210 [1,2,10][12,10]
                    //181 [18,1][1,8,1]
                    //1810[18,10]
                    //1010
                    if (ch == '0')
                    {
                        if (pre == 0 || pre > 20)
                        {
                            return 0;
                        }

                        if (i == 1)
                        {
                            dp[i] = 1;
                        }
                        else
                        {
                            if (s[i - 2] == '0')
                            {
                                dp[i] = dp[i - 1];
                            }
                            else
                            {
                                dp[i] = dp[i - 2];
                            }
                        }
                    }
                    else
                    {
                        if (pre > 10 && pre < 27)
                        {
                            dp[i] = dp[i - 1] + (i == 1 ? 1 : dp[i - 2]);
                        }
                        else
                        {
                            dp[i] = dp[i - 1];
                        }
                    }
                }
                else
                {
                    if (pre > 10 && pre < 27)
                    {
                        dp[i] = dp[i - 1] + (i == 1 ? 1 : dp[i - 2]);
                    }
                    else
                    {
                        dp[i] = dp[i - 1];
                    }
                }
            }

            return dp[s.Length - 1];
        }

        #endregion

        #region 128. 最长连续序列

        //https://leetcode-cn.com/problems/longest-consecutive-sequence/
        public int LongestConsecutive(int[] nums)
        {
            var set = new HashSet<int>();
            foreach (var num in nums)
            {
                set.Add(num);
            }

            int len = 0, mostLen = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                var cur = nums[i];
                if (!set.Contains(cur - 1))
                {
                    while (set.Contains(cur))
                    {
                        len++;
                        cur++;
                    }

                    mostLen = Math.Max(len, mostLen);
                }
            }

            return mostLen;
        }

        #endregion

        #region 47. 全排列 II

        //https://leetcode-cn.com/problems/permutations-ii/
        void PermuteUnique(int[] nums, int index, IList<IList<int>> result)
        {
            if (index >= nums.Length)
            {
                result.Add(new List<int>(nums));
                return;
            }

            var visited = new HashSet<int>();
            for (int i = index; i < nums.Length; i++)
            {
                if (!visited.Add(nums[i]))
                {
                    continue;
                }

                Swap(nums, index, i);
                PermuteUnique(nums, index + 1, result);
                Swap(nums, index, i);
            }
        }

        public IList<IList<int>> PermuteUnique(int[] nums)
        {
            var result = new List<IList<int>>();
            PermuteUnique(nums, 0, result);
            return result;
        }

        #endregion

        #region 118. 杨辉三角

        //https://leetcode-cn.com/problems/pascals-triangle/
        public IList<IList<int>> Generate(int numRows)
        {
            //1
            //1 1
            //1 2 1
            //1 3 3 1
            var result = new List<IList<int>>();
            if (numRows <= 0)
            {
                return result;
            }

            result.Add(new[] { 1 });
            for (int i = 1; i < numRows; i++)
            {
                var prev = result[i - 1];
                var row = new List<int>();
                row.Add(1);
                for (int j = 1; j < prev.Count; j++)
                {
                    row.Add(prev[j - 1] + prev[j]);
                }

                row.Add(1);
                result.Add(row);
            }

            return result;
        }

        #endregion

        #region 119. 杨辉三角 II

        //https://leetcode-cn.com/problems/pascals-triangle-ii/
        public IList<int> GetRow(int rowIndex)
        {
            if (rowIndex <= 0)
            {
                return new int[0];
            }

            List<int> prev = new List<int>(), current = new List<int>();
            current.Add(1);
            for (int i = 1; i < rowIndex; i++)
            {
                var tmp = prev;
                prev = current;
                current = tmp;
                current.Clear();
                current.Add(1);
                for (int j = 1; j < prev.Count; j++)
                {
                    current.Add(prev[j - 1] + prev[j]);
                }

                current.Add(1);
            }

            return current;
        }

        public IList<int> GetRowSingleList(int rowIndex)
        {
            var row = new List<int>();
            row.Add(1);
            for (int i = 1; i <= rowIndex; i++)
            {
                int one, two = row[0];
                for (int j = 1; j < row.Count; j++)
                {
                    one = two;
                    two = row[j];
                    row[j] = one + two;
                }

                row.Add(1);
            }

            return row;
        }

        #endregion

        #region 126. 单词接龙 II

        //https://leetcode-cn.com/problems/word-ladder-ii/
        public IList<IList<string>> FindLadders(string beginWord, string endWord, IList<string> wordList)
        {
            bool Can(string str1, string str2)
            {
                if (str1.Length != str2.Length)
                {
                    return false;
                }

                var diff = 0;
                for (int i = 0; i < str1.Length; i++)
                {
                    if (str1[i] != str2[i])
                    {
                        diff++;
                    }
                }

                return diff == 1;
            }

            var dict = new HashSet<string>();
            var words = new List<string>();
            var wordDict = new Dictionary<string, ISet<string>>();
            words.Add(beginWord);
            foreach (var word in wordList)
            {
                if (dict.Add(word))
                {
                    words.Add(word);
                    wordDict[word] = new HashSet<string>();
                }
            }

            if (!dict.Contains(endWord))
            {
                return new IList<string>[0];
            }

            if (!wordDict.ContainsKey(beginWord))
            {
                wordDict[beginWord] = new HashSet<string>();
            }

            for (int i = 0; i < words.Count - 1; i++)
            {
                var word = words[i];
                var next = wordDict[word];
                for (int j = i + 1; j < words.Count; j++)
                {
                    if (Can(word, words[j]))
                    {
                        next.Add(words[j]);
                        wordDict[words[j]].Add(word);
                    }
                }

                wordDict[word] = next;
            }

            var result = new List<IList<string>>();
            IList<string> path = new List<string>();
            var queue = new Queue<Tuple<string, IList<string>>>();
            var depths = new Dictionary<string, int>();
            queue.Enqueue(new Tuple<string, IList<string>>(beginWord, path));
            while (queue.TryDequeue(out var last))
            {
                beginWord = last.Item1;
                path = last.Item2;
                if (depths.TryGetValue(beginWord, out var depth) && path.Count > depth)
                {
                    continue;
                }

                depths[beginWord] = path.Count;
                path.Add(beginWord);
                if (beginWord == endWord)
                {
                    result.Add(path);
                }
                else
                {
                    var next = wordDict[beginWord];
                    foreach (var word in next)
                    {
                        var newPath = new List<string>(path);
                        queue.Enqueue(new Tuple<string, IList<string>>(word, newPath));
                    }
                }
            }

            return result;
        }

        #endregion

        #region 990. 等式方程的可满足性

        //https://leetcode-cn.com/problems/satisfiability-of-equality-equations/
        public bool EquationsPossible(string[] equations)
        {
            Dictionary<char, ISet<char>> equals = new Dictionary<char, ISet<char>>(),
                not = new Dictionary<char, ISet<char>>();
            for (int i = 0; i < 26; i++)
            {
                equals[(char)('a' + i)] = new HashSet<char>();
                not[(char)('a' + i)] = new HashSet<char>();
            }

            foreach (var equation in equations)
            {
                char start = equation[0], end = equation[3];
                if (equation[1] == '=')
                {
                    equals[start].Add(end);
                    equals[end].Add(start);
                }
                else if (start != end)
                {
                    not[start].Add(end);
                    not[end].Add(start);
                }
                else
                {
                    return false;
                }
            }

            var queue = new Queue<char>();
            foreach (var equation in equations)
            {
                ISet<char> set1 = new HashSet<char>(), set2 = new HashSet<char>();
                var start = equation[0];
                queue.Enqueue(start);
                while (queue.TryDequeue(out var ch))
                {
                    var set = equals[ch];
                    foreach (var c in set)
                    {
                        if (start != c && set1.Add(c))
                        {
                            queue.Enqueue(c);
                        }
                    }

                    set = not[ch];
                    foreach (var c in set)
                    {
                        set2.Add(c);
                    }
                }

                if (set1.Overlaps(set2))
                {
                    return false;
                }
            }

            return true;
        }

        #endregion

        #region 94. 二叉树的中序遍历(非递归)

        //https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
        public IList<int> InorderTraversal(TreeNode root)
        {
            var stack = new Stack<TreeNode>();
            var result = new List<int>();
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                if (stack.Count > 0)
                {
                    root = stack.Pop();
                    result.Add(root.val);
                    root = root.right;
                }
            }

            return result;
        }

        #endregion

        #region 108. 将有序数组转换为二叉搜索树/面试题 04.02. 最小高度树

        //https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/
        //https://leetcode-cn.com/problems/minimum-height-tree-lcci/
        TreeNode SortedArrayToBST(IList<int> nums, int start, int end)
        {
            if (start > end)
            {
                return null;
            }

            var index = (start + end) / 2;
            var root = new TreeNode(nums[index])
            {
                left = SortedArrayToBST(nums, start, index - 1),
                right = SortedArrayToBST(nums, index + 1, end)
            };
            return root;
        }

        public TreeNode SortedArrayToBST(int[] nums)
        {
            if (nums.Length <= 0)
            {
                return null;
            }

            return SortedArrayToBST(nums, 0, nums.Length - 1);
        }

        #endregion

        #region 面试题 02.04. 分割链表/86. 分隔链表

        //https://leetcode-cn.com/problems/partition-list-lcci/
        //https://leetcode-cn.com/problems/partition-list/
        public ListNode Partition(ListNode head, int x)
        {
            if (head == null || head.next == null)
            {
                return head;
            }
            ListNode small = new ListNode(-1), big = new ListNode(-1);
            ListNode smallHead = small, bigHead = big;
            while (head != null)
            {
                if (head.val >= x)
                {
                    big.next = head;
                    big = big.next;
                }
                else
                {
                    small.next = head;
                    small = small.next;
                }
                head = head.next;
            }
            small.next = bigHead.next;
            return smallHead.next;
        }

        #endregion

        #region 219. 存在重复元素 II

        //https://leetcode-cn.com/problems/contains-duplicate-ii/
        public bool ContainsNearbyDuplicate(int[] nums, int k)
        {
            var dict = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                if (dict.TryGetValue(nums[i], out var index))
                {
                    if (i - index <= k)
                    {
                        return true;
                    }

                    dict[nums[i]] = i;
                }
                else
                {
                    dict[nums[i]] = i;
                }
            }

            return false;
        }

        #endregion

        #region 116. 填充每个节点的下一个右侧节点指针

        //https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/
        public Node Connect(Node root)
        {
            if (root == null)
            {
                return root;
            }

            var queue = new Queue<Node>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                var size = queue.Count;
                Node prev = null;
                while (size > 0)
                {
                    var node = queue.Dequeue();
                    if (prev != null)
                    {
                        prev.next = node;
                        prev = prev.next;
                    }
                    else
                    {
                        prev = node;
                    }

                    if (node.left != null)
                    {
                        queue.Enqueue(node.left);
                    }

                    if (node.right != null)
                    {
                        queue.Enqueue(node.right);
                    }

                    size--;
                }
            }

            return root;
        }

        void Next(Node current, Node next)
        {
            if (current == null)
            {
                return;
            }

            current.next = next;
            Next(current.left, current.right);
            if (next != null)
            {
                Next(next.left, next.right);
            }
        }

        public Node Connect1(Node root)
        {
            if (root == null) return null;
            if (root.left == null) return root;

            root.left.next = root.right;
            root.right.next = root.next?.left;

            Connect(root.left);
            Connect(root.right);

            return root;
        }

        #endregion


        #region 415. 字符串相加

        //https://leetcode-cn.com/problems/add-strings/
        public string AddStrings(string num1, string num2)
        {
            var result = new char[Math.Max(num1.Length, num2.Length) + 1];
            int i1 = num1.Length - 1, i2 = num2.Length - 1, index = result.Length - 1;
            bool plus = false;
            while (i1 >= 0 || i2 >= 0)
            {
                var one = plus ? 1 : 0;
                if (i1 >= 0)
                {
                    one += (num1[i1] - '0');
                    i1--;
                }

                if (i2 >= 0)
                {
                    one += (num2[i2] - '0');
                    i2--;
                }

                if (one > 9)
                {
                    plus = true;
                    one -= 10;
                }
                else
                {
                    plus = false;
                }

                result[index--] = (char)(one + '0');
            }

            if (plus)
            {
                result[index--] = '1';
            }

            return new string(result, index + 1, result.Length - index - 1);
        }

        #endregion

        #region 767. 重构字符串

        //https://leetcode-cn.com/problems/reorganize-string/

        bool ReorganizeString(IList<char> chars, StringBuilder result)
        {
            if (chars.Count <= 0)
            {
                return true;
            }

            for (int i = 0; i < chars.Count; i++)
            {
                if (result.Length <= 0)
                {
                    result.Append(chars[0]);
                    chars.RemoveAt(0);
                    if (ReorganizeString(chars, result))
                    {
                        return true;
                    }
                }
                else
                {
                    var last = result[result.Length - 1];
                    if (last != chars[i])
                    {
                        result.Append(chars[i]);
                        chars.RemoveAt(i);
                        return ReorganizeString(chars, result);
                    }
                }
            }

            return false;
        }

        class Item : IComparer<Item>, IComparable
        {
            public char Char;
            public int Count;


            public int Compare(Item x, Item y)
            {
                return y.Count - x.Count;
            }

            public int CompareTo(object obj)
            {
                return ((Item)obj).Count - Count;
            }
        }

        public string ReorganizeString(string s)
        {
            var dict = new Dictionary<char, Item>();
            foreach (var ch in s)
            {
                if (!dict.ContainsKey(ch))
                {
                    dict[ch] = new Item() { Char = ch, Count = 1 };
                }
                else
                {
                    dict[ch].Count++;
                }
            }

            var items = dict.Values.ToArray();
            Array.Sort(items);
            var result = new StringBuilder();
            while (result.Length < s.Length)
            {
                var flag = true;
                foreach (var item in items)
                {
                    if (item.Count <= 0 || (result.Length > 0 && result[result.Length - 1] == item.Char))
                    {
                        continue;
                    }

                    result.Append(item.Char);
                    item.Count--;
                    flag = false;
                    break;
                }

                if (flag)
                {
                    return string.Empty;
                }

                Array.Sort(items);
            }

            return result.ToString();
        }

        #endregion

        #region 面试题 01.06. 字符串压缩

        //https://leetcode-cn.com/problems/compress-string-lcci/
        public string CompressString(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return s;
            }

            var result = new StringBuilder();
            var last = s[0];
            int size = 1;
            for (int i = 1; i < s.Length; i++)
            {
                if (last == s[i])
                {
                    size++;
                }
                else
                {
                    result.Append(size).Append(last);
                    last = s[i];
                    size = 1;
                }
            }

            if (size > 0)
            {
                result.Append(size).Append(last);
            }

            return result.Length < s.Length ? result.ToString() : s;
        }

        #endregion

        #region 443. 压缩字符串

        //https://leetcode-cn.com/problems/string-compression/
        public int Compress(char[] chars)
        {
            if (chars.Length <= 0)
            {
                return 0;
            }

            var last = chars[0];
            int size = 1, len = 1, j = 1;
            for (int i = 1; i < chars.Length; i++)
            {
                if (last == chars[i])
                {
                    size++;
                }
                else
                {
                    len++;
                    if (size > 1)
                    {
                        var str = size.ToString();
                        for (int k = 0; k < str.Length; k++)
                        {
                            chars[j++] = str[k];
                        }

                        size = 1;
                        len += str.Length;
                    }

                    last = chars[i];
                    chars[j++] = last;
                }
            }

            if (size > 0)
            {
                var str = size.ToString();
                for (int k = 0; k < str.Length; k++)
                {
                    chars[j++] = str[k];
                }

                len += str.Length;
            }

            return len;
        }

        #endregion

        #region 1408. 数组中的字符串匹配

        //https://leetcode-cn.com/problems/string-matching-in-an-array/
        public IList<string> StringMatching(string[] words)
        {
            var not = new HashSet<int>();
            var result = new List<string>();
            for (int i = 0; i < words.Length - 1; i++)
            {
                if (not.Contains(i))
                {
                    continue;
                }

                for (int j = i + 1; j < words.Length; j++)
                {
                    if (not.Contains(j))
                    {
                        continue;
                    }

                    if (words[i].Length > words[j].Length && words[i].Contains(words[j]))
                    {
                        not.Add(j);
                        result.Add(words[j]);
                    }
                    else if (words[j].Length > words[i].Length && words[j].Contains(words[i]))
                    {
                        not.Add(i);
                        result.Add(words[i]);
                        break;
                    }
                }
            }

            return result;
        }

        #endregion

        #region 142. 环形链表 II/面试题 02.08. 环路检测

        //https://leetcode-cn.com/problems/linked-list-cycle-ii/
        //https://leetcode-cn.com/problems/linked-list-cycle-lcci/
        public ListNode DetectCycle(ListNode head)
        {
            if (head == null)
            {
                return null;
            }

            ListNode slow = head, fast = head;
            while (fast != null && fast.next != null)
            {
                fast = fast.next.next;
                slow = slow.next;
                if (fast == slow)
                {
                    fast = head;
                    while (fast != slow)
                    {
                        fast = fast.next;
                        slow = slow.next;
                    }

                    return fast;
                }
            }

            return null;
        }

        #endregion

        #region 678. 有效的括号字符串

        //https://leetcode-cn.com/problems/valid-parenthesis-string/
        public bool CheckValidString(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return true;
            }

            int min = 0, max = 0;
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '(')
                {
                    min++;
                    max++;
                }
                else if (s[i] == ')')
                {
                    if (min > 0)
                    {
                        min--;
                    }

                    max--;
                    if (max < 0)
                    {
                        return false;
                    }
                }
                else
                {
                    if (min > 0)
                    {
                        min--;
                    }

                    max++;
                }
            }

            return min == 0;
        }

        #endregion

        #region 125. 验证回文串

        //https://leetcode-cn.com/problems/valid-palindrome/
        public bool IsPalindrome(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return true;
            }

            int start = 0, end = s.Length - 1;
            while (start < end)
            {
                if (!char.IsLetterOrDigit(s[start]))
                {
                    start++;
                }
                else if (!char.IsLetterOrDigit(s[end]))
                {
                    end--;
                }
                else if (char.ToLower(s[start]) == char.ToLower(s[end]))
                {
                    start++;
                    end--;
                }
                else
                {
                    return false;
                }
            }

            return true;
        }

        #endregion

        #region 234. 回文链表/面试题 02.06. 回文链表

        //https://leetcode-cn.com/problems/palindrome-linked-list/
        //https://leetcode-cn.com/problems/palindrome-linked-list-lcci/
        public bool IsPalindrome(ListNode head)
        {
            var stack = new Stack<ListNode>();
            var node = head;
            while (node != null)
            {
                stack.Push(node);
                node = node.next;
            }

            while (stack.Count > 0)
            {
                if (stack.Pop().val != head.val)
                {
                    return false;
                }

                head = head.next;
            }

            return true;
        }

        #endregion

        #region 141. 环形链表

        //https://leetcode-cn.com/problems/linked-list-cycle/
        public bool HasCycle(ListNode head)
        {
            if (head == null)
            {
                return false;
            }

            ListNode slow = head, fast = head;
            while (fast != null && fast.next != null)
            {
                fast = fast.next.next;
                slow = slow.next;
                if (fast == slow)
                {
                    return true;
                }
            }

            return false;
        }

        #endregion

        #region 171. Excel表列序号

        //https://leetcode-cn.com/problems/excel-sheet-column-number/
        public int TitleToNumber(string s)
        {
            var res = 0;
            for (int i = 0; i < s.Length; i++)
            {
                var num = s[i] - 'A' + 1;
                res += (num * (int)Math.Pow(26, s.Length - i - 1));
            }

            return res;
        }

        #endregion

        #region 412. Fizz Buzz

        //https://leetcode-cn.com/problems/fizz-buzz/
        public IList<string> FizzBuzz(int n)
        {
            var result = new string[n];
            for (int i = 1; i <= n; i++)
            {
                if (i % 3 == 0 && i % 5 == 0)
                {
                    result[i] = "FizzBuzz";
                }
                else if (i % 3 == 0)
                {
                    result[i] = "Fizz";
                }
                else if (i % 5 == 0)
                {
                    result[i] = "Buzz";
                }
                else
                {
                    result[i] = i.ToString();
                }
            }

            return result;
        }

        #endregion

        #region 283. 移动零

        //https://leetcode-cn.com/problems/move-zeroes/
        public void MoveZeroes(int[] nums)
        {
            for (int i = 0, j = 0; i < nums.Length; i++)
            {
                if (nums[i] != 0)
                {
                    var tmp = nums[j];
                    nums[j] = nums[i];
                    nums[i] = tmp;
                    j++;
                }
            }
        }

        #endregion

        #region 344. 反转字符串

        //https://leetcode-cn.com/problems/reverse-string/
        public void ReverseString(char[] s)
        {
            int start = 0, end = s.Length - 1;
            while (start < end)
            {
                var tmp = s[start];
                s[start] = s[end];
                s[end] = tmp;
                start++;
                end--;
            }
        }

        #endregion

        #region 326. 3的幂

        //https://leetcode-cn.com/problems/power-of-three/
        public bool IsPowerOfThree(int n)
        {
            if (n <= 0)
            {
                return false;
            }

            while (n != 1)
            {
                if (n % 3 != 0)
                {
                    return false;
                }

                n /= 3;
            }

            return true;
        }

        #endregion


        #region 230. 二叉搜索树中第K小的元素

        //https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/
        public int KthSmallest(TreeNode root, int k)
        {
            var stack = new Stack<TreeNode>();
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                k--;
                if (k == 0)
                {
                    return root.val;
                }

                root = root.right;
            }

            return -1;
        }

        #endregion

        #region 16. 最接近的三数之和

        //https://leetcode-cn.com/problems/3sum-closest/
        public int ThreeSumClosest(int[] nums, int target)
        {
            Array.Sort(nums);
            int result = target, min = int.MaxValue;
            for (int i = 0; i < nums.Length; i++)
            {
                if (i > 0 && nums[i - 1] == nums[i])
                {
                    continue;
                }

                int start = i + 1, end = nums.Length - 1;
                while (start < end)
                {
                    var sum = nums[i] + nums[start] + nums[end];
                    if (sum == target)
                    {
                        return target;
                    }

                    int diff;
                    if (sum < target)
                    {
                        diff = target - sum;
                        start++;
                    }
                    else
                    {
                        diff = sum - target;
                        end--;
                    }

                    if (diff < min)
                    {
                        min = diff;
                        result = sum;
                    }
                }
            }

            return result;
        }

        #endregion

        #region 59. 螺旋矩阵 II

        //https://leetcode-cn.com/problems/spiral-matrix-ii/
        public int[][] GenerateMatrix(int n)
        {
            var result = new int[n][];
            for (int i = 0; i < n; i++)
            {
                result[i] = new int[n];
            }

            int size = 0, total = n * n;
            int x0 = 0, x1 = n - 1, y0 = 0, y1 = n - 1;
            while (total > size)
            {
                for (int i = x0; i <= x1; i++)
                {
                    result[y0][i] = ++size;
                }

                y0++;
                for (int i = y0; i <= y1; i++)
                {
                    result[i][x1] = ++size;
                }

                x1--;
                for (int i = x1; i >= x0; i--)
                {
                    result[y1][i] = ++size;
                }

                y1--;
                for (int i = y1; i >= y0; i--)
                {
                    result[i][x0] = ++size;
                }

                x0++;
            }

            return result;
        }

        #endregion

        #region 61. 旋转链表

        //https://leetcode-cn.com/problems/rotate-list/
        //获取倒数k+1的节点进行链接
        public ListNode RotateRight(ListNode head, int k)
        {
            if (head == null || k == 0)
            {
                return head;
            }

            var stack = new Stack<ListNode>();
            var node = head;
            while (node != null)
            {
                stack.Push(node);
                node = node.next;
            }

            k %= stack.Count;
            if (k == 0)
            {
                return head;
            }

            var last = stack.Peek();
            while (k > 0)
            {
                node = stack.Pop();
                k--;
            }

            if (stack.Count > 0)
            {
                stack.Peek().next = null;
            }

            last.next = head;
            return node;
        }

        #endregion

        #region 89. 格雷编码

        //https://leetcode-cn.com/problems/gray-code/
        public IList<int> GrayCode(int n)
        {
            var result = new List<int>();
            result.Add(0);
            var mask = 1;
            while (n != 0)
            {
                for (int i = result.Count - 1; i >= 0; i--)
                {
                    result.Add(mask + result[i]);
                }

                mask <<= 1;
                n--;
            }

            return result;
        }

        #endregion

        #region 148. 排序链表

        //https://leetcode-cn.com/problems/sort-list/
        ListNode FindHalf(ListNode node)
        {
            if (node == null || node.next == null)
            {
                return node;
            }

            ListNode fast = node.next.next, slow = node.next, prev = node;
            while (fast != null && fast.next != null)
            {
                fast = fast.next.next;
                prev = slow;
                slow = slow.next;
            }

            prev.next = null;
            return slow;
        }

        public ListNode SortList(ListNode head)
        {
            if (head == null || head.next == null)
            {
                return head;
            }

            var half = FindHalf(head);
            var node1 = SortList(head);
            var node2 = SortList(half);
            ListNode newHead;
            if (node1.val <= node2.val)
            {
                newHead = node1;
                node1 = node1.next;
            }
            else
            {
                newHead = node2;
                node2 = node2.next;
            }

            var node = newHead;
            while (node1 != null && node2 != null)
            {
                if (node1.val <= node2.val)
                {
                    node.next = node1;
                    node1 = node1.next;
                }
                else
                {
                    node.next = node2;
                    node2 = node2.next;
                }

                node = node.next;
            }

            if (node1 != null)
            {
                node.next = node1;
            }

            if (node2 != null)
            {
                node.next = node2;
            }

            return newHead;
        }

        #endregion

        #region 面试题 02.02. 返回倒数第 k 个节点

        //https://leetcode-cn.com/problems/kth-node-from-end-of-list-lcci/
        public int KthToLast(ListNode head, int k)
        {
            ListNode fast = head;
            while (fast != null && k > 1)
            {
                fast = fast.next;
                k--;
            }

            while (fast != null)
            {
                fast = fast.next;
                head = head.next;
            }

            return head.val;
        }

        #endregion

        #region 231. 2的幂

        public bool IsPowerOfTwoI(int n)
        {
            if (n <= 0)
            {
                return false;
            }

            while (n != 1)
            {
                if (n % 2 != 0)
                {
                    return false;
                }

                n >>= 1;
            }

            return true;
        }

        public bool IsPowerOfTwo(int n)
        {
            if (n <= 0)
            {
                return false;
            }

            //位运算
            //2的次幂2进制
            //00001 -1 000
            //00010 -1 001
            //00100 -1 011
            //01000 -1 111
            return (n & (n - 1)) == 0;
        }

        #endregion

        #region 122. 买卖股票的最佳时机 II

        //https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
        public int MaxProfitII(int[] prices)
        {
            var max = 0;
            for (int i = 0, j = 1; i < prices.Length - 1; i++, j++)
            {
                if (prices[i] >= prices[j])
                {
                    continue;
                }

                max += prices[j] - prices[i];
            }

            return max;
        }

        #endregion

        #region 124. 二叉树中的最大路径和

        //https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/
        public int MaxPathSum(TreeNode root, int path, ref int maxPath)
        {
            if (root == null)
            {
                return path;
            }

            int left = Math.Max(MaxPathSum(root.left, 0, ref maxPath), 0),
                right = Math.Max(0, MaxPathSum(root.right, 0, ref maxPath));
            var thisPath = left + right + root.val;
            maxPath = Math.Max(thisPath, maxPath);
            return root.val + Math.Max(left, right);
        }

        public int MaxPathSum(TreeNode root)
        {
            var maxPath = int.MinValue;
            MaxPathSum(root, 0, ref maxPath);
            return maxPath;
        }

        #endregion

        #region 43. 字符串相乘

        //https://leetcode-cn.com/problems/multiply-strings/

        public string Multiply(string num1, string num2)
        {
            if (num1 == "0" || num2 == "0")
            {
                return "0";
            }

            var result = new int[num1.Length + num2.Length];
            var index = result.Length;
            for (int i = num1.Length - 1; i >= 0; i--)
            {
                index--;
                var n1 = num1[i] - '0';
                if (n1 == 0)
                {
                    continue;
                }

                for (int j = num2.Length - 1, k = index; j >= 0; j--, k--)
                {
                    var n2 = num2[j] - '0';
                    var num = n1 * n2 + result[k];
                    if (num >= 10)
                    {
                        result[k] = num % 10;
                        result[k - 1] += num / 10;
                    }
                    else
                    {
                        result[k] = num;
                    }
                }
            }

            var sb = new StringBuilder();
            var flag = false;
            foreach (var num in result)
            {
                if (num == 0 && !flag)
                {
                    continue;
                }

                sb.Append(num);
                flag = true;
            }

            return sb.ToString();
        }

        #endregion

        #region 31. 下一个排列

        //https://leetcode-cn.com/problems/next-permutation/
        //1,3,2 1
        public void NextPermutation(int[] nums)
        {
            var index = nums.Length - 1;
            //步骤1：向前查找第一个后面的数大于前面的数
            while (index > 0 && nums[index] <= nums[index - 1])
            {
                index--;
            }

            if (index == 0)
            {
                Array.Reverse(nums);
                return;
            }

            index--;

            //步骤2：向后查找最小的大于前面的数
            var end = index;
            while (end < nums.Length - 1 && nums[end + 1] > nums[index])
            {
                end++;
            }

            //步骤3：交换两个数
            var tmp = nums[index];
            nums[index] = nums[end];
            nums[end] = tmp;
            //步骤4：根据前置条件可知此时序列是一个降序序列，反转成升序序列
            Array.Reverse(nums, index + 1, nums.Length - index - 1);
        }

        #endregion

        #region 39. 组合总和

        //https://leetcode-cn.com/problems/combination-sum/
        void CombinationSum(int index, int[] candidates, int target, IList<int> seqs, IList<IList<int>> result)
        {
            if (target == 0)
            {
                result.Add(seqs.ToArray());
                return;
            }

            if (target < 0)
            {
                return;
            }

            for (int i = index; i < candidates.Length; i++)
            {
                var num = candidates[i];
                seqs.Add(num);
                CombinationSum(i, candidates, target - num, seqs, result);
                seqs.RemoveAt(seqs.Count - 1);
            }
        }

        public IList<IList<int>> CombinationSum(int[] candidates, int target)
        {
            var result = new List<IList<int>>();
            Array.Sort(candidates);
            CombinationSum(0, candidates, target, new List<int>(), result);
            return result;
        }

        //DFS+剪枝
        public IList<IList<int>> CombinationSumAndCut(int[] candidates, int target)
        {
            var result = new List<IList<int>>();
            var items = new List<int>();
            Array.Sort(candidates);

            void Dfs(int sum, int j)
            {
                if (sum == target)
                {
                    result.Add(items.ToArray());
                    return;
                }

                for (int i = j; i < candidates.Length && candidates[i] <= target - sum; i++)
                {
                    items.Add(candidates[i]);
                    Dfs(sum + candidates[i], i);
                    items.RemoveAt(items.Count - 1);
                }
            }

            Dfs(0, 0);
            return result;
        }

        #endregion

        #region 64. 最小路径和

        //https://leetcode-cn.com/problems/minimum-path-sum/
        public int MinPathSum(int[][] grid)
        {
            //dp[i,j]=min(dp[i-1,j],dp[i,j-1])
            var dp = new int[grid.Length, grid[0].Length];
            dp[0, 0] = grid[0][0];
            for (int i = 1; i < grid[0].Length; i++)
            {
                dp[0, i] = dp[0, i - 1] + grid[0][i];
            }

            for (int i = 1; i < grid.Length; i++)
            {
                dp[i, 0] = dp[i - 1, 0] + grid[i][0];
            }

            for (int i = 1; i < grid.Length; i++)
            {
                for (int j = 1; j < grid[0].Length; j++)
                {
                    dp[i, j] = Math.Min(dp[i - 1, j], dp[i, j - 1]) + grid[i][j];
                }
            }

            return dp[grid.Length - 1, grid[0].Length - 1];
        }

        #endregion

        #region 114. 二叉树展开为链表

        //https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/
        public void Flatten(TreeNode root)
        {
            var stack = new Stack<TreeNode>();
            TreeNode prev = null;
            while (root != null || stack.Count > 0)
            {
                while (root != null)
                {
                    if (prev != null)
                    {
                        prev.left = null;
                        prev.right = root;
                    }

                    prev = root;
                    stack.Push(root.right);
                    root = root.left;
                }

                root = stack.Pop();
            }
        }

        #endregion

        #region 139. 单词拆分

        //https://leetcode-cn.com/problems/word-break/
        bool WordBreakI(string s, int index, int min, int max, ISet<string> wordDict, Dictionary<int, bool> flags)
        {
            if (index >= s.Length)
            {
                return true;
            }

            if (flags.TryGetValue(index, out var flag))
            {
                return flag;
            }

            for (int i = min; i <= max; i++)
            {
                if (index + i > s.Length)
                {
                    return false;
                }

                var key = s.Substring(index, i);
                if (wordDict.Contains(key))
                {
                    if (WordBreakI(s, index + i, min, max, wordDict, flags))
                    {
                        flags[index + i] = true;
                        return true;
                    }

                    flags[index + i] = false;
                }
            }

            return false;
        }

        public bool WordBreakI(string s, IList<string> wordDict)
        {
            if (wordDict.Count <= 0)
            {
                return string.IsNullOrEmpty(s);
            }

            var dictSet = new HashSet<string>();
            int min = int.MaxValue, max = int.MinValue;
            foreach (var word in wordDict)
            {
                dictSet.Add(word);
                min = Math.Min(min, word.Length);
                max = Math.Max(max, word.Length);
            }

            var flags = new Dictionary<int, bool>();
            return WordBreakI(s, 0, min, max, dictSet, flags);
        }

        #endregion

        #region 437. 路径总和 III/面试题 04.12. 求和路径

        //https://leetcode-cn.com/problems/path-sum-iii/
        //https://leetcode-cn.com/problems/paths-with-sum-lcci/
        int CountPath(TreeNode node, int sum)
        {
            if (node == null)
            {
                return 0;
            }

            int count = 0, target = sum - node.val;
            if (target == 0)
            {
                count++;
            }

            count += CountPath(node.left, target);
            count += CountPath(node.right, target);
            return count;
        }

        public int PathSumIII(TreeNode root, int sum)
        {
            if (root == null)
            {
                return 0;
            }

            var count = CountPath(root, sum);
            count += PathSumIII(root.left, sum);
            count += PathSumIII(root.right, sum);
            return count;
        }

        #endregion

        #region 448. 找到所有数组中消失的数字

        //https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/
        public IList<int> FindDisappearedNumbers(int[] nums)
        {
            var result = new List<int>();
            var set = new HashSet<int>(nums.Length);
            foreach (var num in nums)
            {
                set.Add(num);
            }

            for (int i = 1; i <= nums.Length; i++)
            {
                if (!set.Contains(i))
                {
                    result.Add(i);
                }
            }

            return result;
        }

        #endregion

        #region 617. 合并二叉树

        //https://leetcode-cn.com/problems/merge-two-binary-trees/
        public TreeNode MergeTrees(TreeNode t1, TreeNode t2)
        {
            if (t1 == null && t2 == null)
            {
                return null;
            }

            if (t1 == null)
            {
                return t2;
            }

            if (t2 == null)
            {
                return t1;
            }

            var newNode = new TreeNode(t1.val + t2.val);
            newNode.left = MergeTrees(t1.left, t2.left);
            newNode.right = MergeTrees(t1.right, t2.right);
            return newNode;
        }

        #endregion

        #region 538. 把二叉搜索树转换为累加树

        //https://leetcode-cn.com/problems/convert-bst-to-greater-tree/
        //https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree/
        public TreeNode ConvertBST(TreeNode root)
        {
            var stack = new Stack<TreeNode>();
            TreeNode node = root, prev = null;
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

        #region 543. 二叉树的直径

        //https://leetcode-cn.com/problems/diameter-of-binary-tree/
        public int DiameterOfBinaryTree(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            var result = int.MinValue;

            int Deepth(TreeNode node)
            {
                if (node == null)
                {
                    return 0;
                }

                int left = Deepth(node.left), right = Deepth(node.right);
                result = Math.Max(result, left + right);
                return Math.Max(left, right) + 1;
            }

            Deepth(root);
            return result;
        }

        #endregion

        #region 739. 每日温度

        //https://leetcode-cn.com/problems/daily-temperatures/
        //暴力解
        public int[] DailyTemperatures(int[] items)
        {
            var result = new int[items.Length];
            for (int i = 0; i < items.Length - 1; i++)
            {
                for (int j = i + 1; j < items.Length; j++)
                {
                    if (items[i] < items[j])
                    {
                        result[i] = j - i;
                        break;
                    }
                }
            }

            return result;
        }

        //单调栈
        public int[] DailyTemperaturesStack(int[] items)
        {
            var result = new int[items.Length];
            var stack = new Stack<int>();
            for (int i = 0; i < items.Length; i++)
            {
                while (stack.Count > 0 && items[stack.Peek()] < items[i])
                {
                    var prev = stack.Pop();
                    result[prev] = i - prev;
                }

                stack.Push(i);
            }

            return result;
        }

        #endregion

        #region 96. 不同的二叉搜索树

        //https://leetcode-cn.com/problems/unique-binary-search-trees/
        public int NumTrees(int n)
        {
            var cache = new Dictionary<string, int>();

            int Dfs(int start, int end)
            {
                if (start > end)
                {
                    return 1;
                }

                var key = start + "," + end;
                if (cache.TryGetValue(key, out var count))
                {
                    return count;
                }

                for (int i = start; i <= end; i++)
                {
                    count += Dfs(start, i - 1) * Dfs(i + 1, end);
                }

                cache[key] = count;
                return count;
            }

            return Dfs(1, n);
        }

        #endregion

        #region 190. 颠倒二进制位

        //https://leetcode-cn.com/problems/reverse-bits/
        public uint ReverseBits(uint n)
        {
            uint result = 0;
            for (int i = 0; i < 32; i++)
            {
                var bit = 1 & n;
                result <<= 1;
                result |= bit;
                n >>= 1;
            }

            return result;
        }

        #endregion

        #region 204. 计数质数

        //https://leetcode-cn.com/problems/count-primes/
        public int CountPrimes(int n)
        {
            var bitArray = new BitArray(n, true);
            for (int i = 2; i < n; i++)
            {
                if (bitArray[i])
                {
                    for (int j = 2 * i; j < n; j += i)
                    {
                        bitArray[j] = false;
                    }
                }
            }

            var res = 0;
            for (int i = 2; i < n; i++)
            {
                if (bitArray[i])
                {
                    res++;
                }
            }

            return res;
        }

        #endregion

        #region 371. 两整数之和

        //https://leetcode-cn.com/problems/sum-of-two-integers/
        public int GetSum(int a, int b)
        {
            while (true)
            {
                int h = (a & b) << 1, l = a ^ b;
                a = h;
                b = l;
                if (h == 0)
                {
                    return b;
                }
            }
        }

        #endregion

        #region 172. 阶乘后的零/面试题 16.05. 阶乘尾数

        //https://leetcode-cn.com/problems/factorial-trailing-zeroes/
        //https://leetcode-cn.com/problems/factorial-zeros-lcci/
        public int TrailingZeroes(int n)
        {
            var res = 0;
            while (n > 4)
            {
                n /= 5;
                res += n;
            }

            return res;
        }

        #endregion

        #region 1365. 有多少小于当前数字的数字

        //https://leetcode-cn.com/problems/how-many-numbers-are-smaller-than-the-current-number/
        public int[] SmallerNumbersThanCurrent(int[] nums)
        {
            var buckets = new int[101];
            foreach (var num in nums)
            {
                buckets[num]++;
            }

            for (int i = 1; i < buckets.Length; i++)
            {
                buckets[i] += buckets[i - 1];
            }

            var result = new int[nums.Length];
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] == 0)
                {
                    continue;
                }

                result[i] = buckets[nums[i] - 1];
            }

            return result;
        }

        #endregion

        #region 162. 寻找峰值

        //https://leetcode-cn.com/problems/find-peak-element/
        int FindPeakElement(int[] nums, int start, int end)
        {
            while (true)
            {
                if (start == end)
                {
                    return start;
                }

                var target = (start + end) / 2;

                if (nums[target] > nums[target + 1])
                {
                    return FindPeakElement(nums, start, target);
                }

                start = target + 1;
            }
        }

        public int FindPeakElement(int[] nums)
        {
            return FindPeakElement(nums, 0, nums.Length - 1);
        }

        #endregion

        #region 1284. 转化为全零矩阵的最少反转次数

        //https://leetcode-cn.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix/

        bool Check(int[][] flags)
        {
            for (int i = 0; i < flags.Length; i++)
            {
                var arr = flags[i];
                for (int j = 0; j < arr.Length; j++)
                {
                    if (arr[j] == 1)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        void ChangeItem(int[][] mat, int x, int y)
        {
            mat[x][y] = mat[x][y] ^= 1;
            if (x > 0)
            {
                mat[x - 1][y] ^= 1;
            }

            if (x < mat.Length - 1)
            {
                mat[x + 1][y] ^= 1;
            }

            if (y > 0)
            {
                mat[x][y - 1] ^= 1;
            }

            if (y < mat[x].Length - 1)
            {
                mat[x][y + 1] ^= 1;
            }
        }

        bool ChangeMat(int[][] mat, bool[,] flags, int x, int y, int step, ref int result)
        {
            if (x < 0 || y < 0 || x >= mat.Length || y >= mat[0].Length || flags[x, y])
            {
                return false;
            }

            var flag = false;
            ChangeItem(mat, x, y);
            step++;
            if (Check(mat))
            {
                result = Math.Min(result, step);
                flag = true;
            }
            else
            {
                flags[x, y] = true;
                for (int i = 0; i < mat.Length; i++)
                {
                    for (int j = 0; j < mat[0].Length; j++)
                    {
                        if (flags[i, j])
                        {
                            continue;
                        }

                        flag = ChangeMat(mat, flags, i, j, step, ref result) || flag;
                    }
                }

                flags[x, y] = false;
            }

            ChangeItem(mat, x, y);
            return flag;
        }

        public int MinFlips(int[][] mat)
        {
            if (Check(mat))
            {
                return 0;
            }

            var result = int.MaxValue;
            var flags = new bool[mat.Length, mat[0].Length];
            var can = false;
            for (int i = 0; i < mat.Length; i++)
            {
                for (int j = 0; j < mat[0].Length; j++)
                {
                    can = ChangeMat(mat, flags, i, j, 0, ref result) || can;
                }
            }

            return can ? result : -1;
        }

        #endregion

        #region 74. 搜索二维矩阵

        public bool SearchMatrix(int[][] matrix, int target)
        {
            int x = matrix[0].Length - 1, y = 0;
            while (x >= 0 && y < matrix.Length)
            {
                if (matrix[x][y] == target)
                {
                    return true;
                }

                if (matrix[x][y] > target)
                {
                    x--;
                }
                else
                {
                    y++;
                }
            }

            return false;
        }

        #endregion

        #region 328. 奇偶链表

        //https://leetcode-cn.com/problems/odd-even-linked-list/
        public ListNode OddEvenList(ListNode head)
        {
            if (head == null || head.next == null)
            {
                return head;
            }

            ListNode odd = head, evenHead = head.next, even = evenHead;
            while (even != null && even.next != null)
            {
                odd.next = even.next;
                odd = odd.next;
                even.next = odd.next;
                even = even.next;
            }

            odd.next = evenHead;
            return head;
        }

        #endregion

        #region 127. 单词接龙

        //https://leetcode-cn.com/problems/word-ladder/solution/dan-ci-jie-long-by-leetcode/
        public int LadderLength(string beginWord, string endWord, IList<string> wordList)
        {
            if (beginWord == endWord)
            {
                return 0;
            }

            var dict = new Dictionary<string, List<string>>();
            var exists = false;
            foreach (var word in wordList)
            {
                exists = exists || word == endWord;
                for (int i = 0; i < word.Length; i++)
                {
                    var key = word.Substring(0, i) + "*" + word.Substring(i + 1);
                    if (!dict.TryGetValue(key, out var words))
                    {
                        words = new List<string>();
                        dict[key] = words;
                    }

                    words.Add(word);
                }
            }

            if (!exists)
            {
                return 0;
            }

            var queue = new Queue<string>();
            queue.Enqueue(beginWord);
            var visited = new HashSet<string>();
            var step = 0;
            while (queue.Count > 0)
            {
                step++;
                for (int s = 0, l = queue.Count; s < l; s++)
                {
                    var word = queue.Dequeue();
                    if (word == endWord)
                    {
                        return step;
                    }

                    if (!visited.Add(word))
                    {
                        continue;
                    }

                    for (int i = 0; i < word.Length; i++)
                    {
                        var key = word.Substring(0, i) + "*" + word.Substring(i + 1);
                        if (dict.TryGetValue(key, out var words))
                        {
                            foreach (var item in words)
                            {
                                queue.Enqueue(item);
                            }
                        }
                    }
                }
            }

            return 0;
        }

        #endregion

        #region 433. 最小基因变化

        //https://leetcode-cn.com/problems/minimum-genetic-mutation/
        public int MinMutation(string start, string end, string[] bank)
        {
            bool CanJoin(string s1, string s2)
            {
                var diff = 0;
                for (int i = 0; i < s1.Length; i++)
                {
                    if (s1[i] != s2[i])
                    {
                        diff++;
                    }
                }

                return diff == 1;
            }

            var dict = new Dictionary<string, ISet<string>>();
            dict[start] = new HashSet<string>();
            foreach (var word in bank)
            {
                dict[word] = new HashSet<string>();
                if (CanJoin(word, start))
                {
                    dict[start].Add(word);
                    dict[word].Add(start);
                }
            }

            for (int i = 0; i < bank.Length - 1; i++)
            {
                var word1 = bank[i];
                for (int j = i + 1; j < bank.Length; j++)
                {
                    var word2 = bank[j];
                    if (CanJoin(word1, word2))
                    {
                        dict[word1].Add(word2);
                        dict[word2].Add(word1);
                    }
                }
            }

            if (!dict.ContainsKey(end) || dict[start].Count <= 0)
            {
                return 0;
            }

            var visited = new HashSet<string>();
            var step = 0;
            var queue = new Queue<string>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                var size = queue.Count;
                while (size > 0)
                {
                    size--;
                    var word = queue.Dequeue();
                    if (word == end)
                    {
                        return step;
                    }

                    visited.Add(word);
                    foreach (var next in dict[word])
                    {
                        if (visited.Contains(next))
                        {
                            continue;
                        }

                        queue.Enqueue(next);
                    }
                }

                step++;
            }

            return -1;
        }

        #endregion

        #region 48. 旋转图像/面试题 01.07. 旋转矩阵

        //https://leetcode-cn.com/problems/rotate-image/
        //https://leetcode-cn.com/problems/rotate-matrix-lcci/

        void Rotate(int[][] matrix, int x1, int x2, int y1, int y2)
        {
            var tempArr = new int[x2 - x1];
            Array.Copy(matrix[x1], y1, tempArr, 0, tempArr.Length);
            for (int i = 0; i < tempArr.Length; i++)
            {
                var tmp = matrix[x1 + i][y2];
                matrix[x1 + i][y2] = tempArr[i];
                tempArr[i] = tmp;
            }

            for (int i = 0; i < tempArr.Length; i++)
            {
                var tmp = matrix[x2][y2 - i];
                matrix[x2][y2 - i] = tempArr[i];
                tempArr[i] = tmp;
            }

            for (int i = 0; i < tempArr.Length; i++)
            {
                var tmp = matrix[x2 - i][y1];
                matrix[x2 - i][y1] = tempArr[i];
                tempArr[i] = tmp;
            }

            for (int i = 0; i < tempArr.Length; i++)
            {
                matrix[x1][y1 + i] = tempArr[i];
            }
        }

        public void Rotate(int[][] matrix)
        {
            if (matrix == null || matrix.Length <= 0)
            {
                return;
            }

            int start = 0, end = matrix.Length - 1;
            while (start < end)
            {
                Rotate(matrix, start, end, start, end);
                start++;
                end--;
            }
        }

        public void RotateSpaceZero(int[][] matrix)
        {
            if (matrix == null || matrix.Length <= 0)
            {
                return;
            }

            int start = 0, end = matrix.Length - 1;
            while (start < end)
            {
                var len = end - start;
                for (int k = 0; k < len; k++)
                {
                    var tmp = matrix[start][start + k];
                    matrix[start][start + k] = matrix[end - k][start];
                    matrix[end - k][start] = matrix[end][end - k];
                    matrix[end][end - k] = matrix[start + k][end];
                    matrix[start + k][end] = tmp;
                }

                start++;
                end--;
            }
        }

        public void RotateBySelf(int[][] matrix)
        {
            void Rotate(int i)
            {
                for (int s = i, e = matrix[0].Length - i - 1, j = 0; s < e; s++, j++)
                {
                    var mv = matrix[i][s];
                    var tmp = matrix[s][e];
                    matrix[s][e] = mv;
                    mv = tmp;
                    tmp = matrix[e][e - j];
                    matrix[e][e - j] = mv;
                    mv = tmp;
                    tmp = matrix[e - j][i];
                    matrix[e - j][i] = mv;
                    matrix[i][s] = tmp;
                }
            }

            for (int i = 0; i < matrix[0].Length / 2; i++)
            {
                Rotate(i);
            }
        }

        #endregion

        #region 516. 最长回文子序列

        //https://leetcode-cn.com/problems/longest-palindromic-subsequence/
        int LongestPalindromeSubseq(string s, int l, int r, Dictionary<string, int> cache)
        {
            var key = l + "," + r;
            if (cache.TryGetValue(key, out var len))
            {
                return len;
            }

            if (l == r)
            {
                len = 1;
            }
            else
            {
                while (l < r)
                {
                    if (s[l] == s[r])
                    {
                        l++;
                        r--;
                        len += 2;
                    }
                    else
                    {
                        len += Math.Max(LongestPalindromeSubseq(s, l + 1, r, cache),
                            LongestPalindromeSubseq(s, l, r - 1, cache));
                        break;
                    }
                }

                len += (l == r ? 1 : 0);
            }

            cache[key] = len;
            return len;
        }

        public int LongestPalindromeSubseq(string s)
        {
            return LongestPalindromeSubseq(s, 0, s.Length - 1, new Dictionary<string, int>());
        }

        #endregion

        #region 1143. 最长公共子序列

        //https://leetcode-cn.com/problems/longest-common-subsequence/
        public int LongestCommonSubsequence(string text1, string text2)
        {
            int m = text1.Length, n = text2.Length;
            var dp = new int[m + 1, n + 1];
            for (int i = 1; i <= m; i++)
            {
                var c1 = text1[i - 1];
                for (int j = 1; j <= n; j++)
                {
                    if (text2[j - 1] == c1)
                    {
                        dp[i, j] = dp[i - 1, j - 1] + 1;
                    }
                    else
                    {
                        dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
                    }
                }
            }
            return dp[m, n];
        }

        #endregion

        #region 25. K 个一组翻转链表

        //https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
        ListNode ReverseListNode(ListNode head)
        {
            if (head == null)
            {
                return null;
            }

            ListNode prev = null;
            while (head != null)
            {
                var next = head.next;
                head.next = prev;
                prev = head;
                head = next;
            }

            return prev;
        }

        public ListNode ReverseKGroup(ListNode head, int k)
        {
            if (head == null)
            {
                return null;
            }

            ListNode root = null;
            ListNode prevEnd = null;
            ListNode end;
            int len = 0;
            while (true)
            {
                end = head;
                len++;
                while (len < k && head != null)
                {
                    head = head.next;
                    len++;
                }

                if (head == null)
                {
                    break;
                }

                len = 0;
                var next = head.next;
                head.next = null;
                head = next;
                //链表头 node 链表尾
                var start = ReverseListNode(end);
                if (root == null)
                {
                    root = start;
                }
                else
                {
                    prevEnd.next = start;
                }

                prevEnd = end;
            }

            if (prevEnd != null)
            {
                prevEnd.next = end;
                return root;
            }

            return end;
        }

        #endregion

        #region 40. 组合总和 II

        //https://leetcode-cn.com/problems/combination-sum-ii/
        void CombinationSum2(int index, int[] candidates, int target, IList<int> seqs, List<IList<int>> result)
        {
            if (target == 0)
            {
                result.Add(seqs.ToArray());
                return;
            }

            if (index >= candidates.Length || target < 0)
            {
                return;
            }

            for (int i = index; i < candidates.Length; i++)
            {
                if (i > index && candidates[i] == candidates[i - 1])
                {
                    continue;
                }

                seqs.Add(candidates[i]);
                CombinationSum2(i + 1, candidates, target - candidates[i], seqs, result);
                seqs.RemoveAt(seqs.Count - 1);
            }
        }

        public IList<IList<int>> CombinationSum2(int[] candidates, int target)
        {
            var result = new List<IList<int>>();
            Array.Sort(candidates);
            CombinationSum2(0, candidates, target, new List<int>(), result);
            return result;
        }

        #endregion

        #region 1300. 转变数组后最接近目标值的数组和

        //https://leetcode-cn.com/problems/sum-of-mutated-array-closest-to-target/
        public int FindBestValue(int[] arr, int target)
        {
            Array.Sort(arr);
            var prefix = new int[arr.Length + 1];
            for (int i = 1; i < prefix.Length; i++)
            {
                prefix[i] = prefix[i - 1] + arr[i - 1];
            }

            int max = arr[arr.Length - 1], diffMin = target, ans = 0;
            for (int i = 1; i <= max; i++)
            {
                var index = Array.BinarySearch(arr, i);
                if (index < 0)
                {
                    index = -index - 1;
                }

                var sum = prefix[index] + (arr.Length - index) * i;
                var diff = Math.Abs(sum - target);
                if (diff < diffMin)
                {
                    diffMin = diff;
                    ans = i;
                }
                else
                {
                    break;
                }
            }

            return ans;
        }

        #endregion

        #region 14. 最长公共前缀

        //https://leetcode-cn.com/problems/longest-common-prefix/
        public string LongestCommonPrefix(string[] strs)
        {
            if (strs.Length <= 0)
            {
                return string.Empty;
            }

            Array.Sort(strs);
            string s1 = strs[0], s2 = strs[strs.Length - 1];
            var len = 0;
            for (int i = 0; i < Math.Min(s1.Length, s2.Length); i++)
            {
                if (s1[i] != s2[i])
                {
                    break;
                }

                len++;
            }

            return s1.Substring(0, len);
        }

        #endregion

        #region 58. 最后一个单词的长度

        //https://leetcode-cn.com/problems/length-of-last-word/
        public int LengthOfLastWord(string s)
        {
            s = s.Trim();
            for (int i = s.Length - 1; i >= 0; i--)
            {
                if (s[i] == ' ')
                {
                    return s.Length - i - 1;
                }
            }

            return s.Length;
        }

        #endregion

        #region 67. 二进制求和

        //https://leetcode-cn.com/problems/add-binary/
        public string AddBinary(string a, string b)
        {
            int s1 = a.Length - 1, s2 = b.Length - 1;
            var chars = new char[Math.Max(a.Length, b.Length) + 1];
            int pre = 0, index = chars.Length - 1;
            while (s1 >= 0 || s2 >= 0)
            {
                int sum;
                if (s1 >= 0 && s2 >= 0)
                {
                    int c1 = a[s1--] - '0', c2 = b[s2--] - '0';
                    sum = c1 + c2 + pre;
                }
                else if (s1 >= 0)
                {
                    sum = (a[s1--] - '0') + pre;
                }
                else
                {
                    sum = (b[s2--] - '0') + pre;
                }

                if (sum < 2)
                {
                    pre = 0;
                }
                else
                {
                    pre = 1;
                    sum -= 2;
                }

                chars[index--] = sum == 0 ? '0' : '1';
            }

            if (pre > 0)
            {
                chars[index--] = '1';
                return new string(chars);
            }

            return new string(chars, 1, chars.Length - 1);
        }

        #endregion

        #region 100. 相同的树

        //https://leetcode-cn.com/problems/same-tree/
        public bool IsSameTree(TreeNode p, TreeNode q)
        {
            if (p == null)
            {
                return q == null;
            }

            if (q == null || p.val != q.val)
            {
                return false;
            }

            return IsSameTree(p.left, q.left) && IsSameTree(p.right, q.right);
        }

        #endregion

        #region 107. 二叉树的层次遍历 II

        //https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/
        public IList<IList<int>> LevelOrderBottom(TreeNode root)
        {
            if (root == null)
            {
                return new IList<int>[0];
            }

            var queue = new Queue<TreeNode>();
            var result = new List<IList<int>>();
            queue.Enqueue(root);
            int size = queue.Count;
            var items = new List<int>();
            while (queue.Count > 0)
            {
                while (size > 0)
                {
                    size--;
                    root = queue.Dequeue();
                    items.Add(root.val);
                    if (root.left != null)
                    {
                        queue.Enqueue(root.left);
                    }

                    if (root.right != null)
                    {
                        queue.Enqueue(root.right);
                    }
                }

                size = queue.Count;
                if (result.Count <= 0)
                {
                    result.Add(items.ToArray());
                }
                else
                {
                    result.Insert(0, items.ToArray());
                }

                items.Clear();
            }

            return result;
        }

        #endregion

        #region 1014. 最佳观光组合

        //https://leetcode-cn.com/problems/best-sightseeing-pair/
        //暴力法
        public int MaxScoreSightseeingPair(int[] A)
        {
            var max = 0;
            for (int i = 0; i < A.Length - 1; i++)
            {
                for (int j = i + 1; j < A.Length; j++)
                {
                    max = Math.Max(A[j] + A[i] + i - j, max);
                }
            }

            return max;
        }

        public int MaxScoreSightseeingPairFast(int[] A)
        {
            //1.求出i+1之前的max(A[i]+i);
            //2.计算max(ans,max(A[i]+i)+A[i+1]-i-1)
            int preMax = 0, ans = 0;
            for (int i = 0; i < A.Length - 1; i++)
            {
                preMax = Math.Max(preMax, A[i] + i);
                ans = Math.Max(ans, preMax + A[i + 1] - i - 1);
            }

            return ans;
        }

        #endregion

        #region 12. 整数转罗马数字

        //https://leetcode-cn.com/problems/integer-to-roman/
        public string IntToRoman(int num)
        {
            var nums = new[] { 1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000 };
            var romans = new[] { "I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M" };
            var intStr = new StringBuilder();
            for (int i = nums.Length - 1; i >= 0 && num != 0; i--)
            {
                while (num >= nums[i])
                {
                    num -= nums[i];
                    intStr.Append(romans[i]);
                }
            }

            return intStr.ToString();
        }

        #endregion

        #region 111. 二叉树的最小深度

        //https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/
        public int MinDepth(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            int left = MinDepth(root.left), right = MinDepth(root.right);
            if (left == 0)
            {
                return right + 1;
            }

            if (right == 0)
            {
                return left + 1;
            }

            return Math.Min(left, right) + 1;
        }

        #endregion

        #region 112. 路径总和

        //https://leetcode-cn.com/problems/path-sum/
        public bool HasPathSum(TreeNode root, int sum)
        {
            if (root == null)
            {
                return false;
            }

            sum -= root.val;
            if (root.left == null && root.right == null)
            {
                return sum == 0;
            }

            if (root.left != null && HasPathSum(root.left, sum))
            {
                return true;
            }

            return root.right != null && HasPathSum(root.right, sum);
        }

        #endregion

        #region 168. Excel表列名称

        //https://leetcode-cn.com/problems/excel-sheet-column-title/
        public string ConvertToTitle(int n)
        {
            var chars = new char[26];
            for (int i = 1; i < chars.Length; i++)
            {
                chars[i] = (char)('A' + i - 1);
            }

            chars[0] = 'Z';
            var res = new StringBuilder(string.Empty);
            while (n != 0)
            {
                res.Insert(0, chars[n % 26]);
                if (n % 26 == 0)
                {
                    n /= 26;
                    n--;
                }
                else
                {
                    n /= 26;
                }
            }

            return res.ToString();
        }

        #endregion

        #region 203. 移除链表元素

        //https://leetcode-cn.com/problems/remove-linked-list-elements/
        public ListNode RemoveElements(ListNode head, int val)
        {
            while (head != null && head.val == val)
            {
                head = head.next;
            }

            if (head == null)
            {
                return null;
            }

            ListNode prev = null, root = head;
            while (head != null)
            {
                if (head.val == val)
                {
                    prev.next = head.next;
                }
                else
                {
                    prev = head;
                }

                head = head.next;
            }

            return root;
        }

        #endregion

        #region 205. 同构字符串

        //https://leetcode-cn.com/problems/isomorphic-strings/
        public bool IsIsomorphic(string s, string t)
        {
            if (s.Length != t.Length)
            {
                return false;
            }

            Dictionary<char, char> sDict = new Dictionary<char, char>(), tDict = new Dictionary<char, char>();
            for (int i = 0; i < s.Length; i++)
            {
                char sc = s[i], tc = t[i], cmp;
                if (sDict.TryGetValue(sc, out cmp) && cmp != tc)
                {
                    return false;
                }

                if (tDict.TryGetValue(tc, out cmp) && cmp != sc)
                {
                    return false;
                }

                sDict[sc] = tc;
                tDict[tc] = sc;
            }

            return true;
        }

        #endregion

        #region 257. 二叉树的所有路径

        //https://leetcode-cn.com/problems/binary-tree-paths/

        void BinaryTreePaths(TreeNode root, IList<string> result, IList<int> paths)
        {
            if (root == null)
            {
                result.Add(string.Join("->", paths));
                return;
            }

            paths.Add(root.val);
            if (root.left == null && root.right == null)
            {
                result.Add(string.Join("->", paths));
            }
            else
            {
                BinaryTreePaths(root.left, result, paths);
                BinaryTreePaths(root.right, result, paths);
            }

            paths.RemoveAt(paths.Count - 1);
        }

        public IList<string> BinaryTreePaths(TreeNode root)
        {
            if (root == null)
            {
                return new string[0];
            }

            var result = new List<string>();
            BinaryTreePaths(root, result, new List<int>());
            return result;
        }

        #endregion

        #region 258. 各位相加

        //https://leetcode-cn.com/problems/add-digits/
        public int AddDigits(int num)
        {
            while (num >= 10)
            {
                var sum = 0;
                while (num != 0)
                {
                    sum += num % 10;
                    num /= 10;
                }

                num = sum;
            }

            return num;
        }

        #endregion

        #region 263. 丑数

        //https://leetcode-cn.com/problems/ugly-number/
        public bool IsUgly(int num)
        {
            if (num <= 0)
            {
                return false;
            }

            while (num % 2 == 0)
            {
                num /= 2;
            }

            while (num % 3 == 0)
            {
                num /= 3;
            }

            while (num % 5 == 0)
            {
                num /= 5;
            }

            return num == 1;
        }

        #endregion

        #region 面试题 01.01. 判定字符是否唯一

        //https://leetcode-cn.com/problems/is-unique-lcci/
        public bool IsUnique(string astr)
        {
            var set = new HashSet<char>();
            foreach (var ch in astr)
            {
                if (!set.Add(ch))
                {
                    return false;
                }
            }

            return true;
        }

        #endregion

        #region 面试题 01.02. 判定是否互为字符重排

        //https://leetcode-cn.com/problems/check-permutation-lcci/
        public bool CheckPermutation(string s1, string s2)
        {
            var dict = new Dictionary<char, int>();
            foreach (var ch in s1)
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

            foreach (var ch in s2)
            {
                if (!dict.TryGetValue(ch, out var count))
                {
                    return false;
                }

                if (count == 1)
                {
                    dict.Remove(ch);
                }
                else
                {
                    dict[ch] = count - 1;
                }
            }

            return dict.Count == 0;
        }

        #endregion

        #region 面试题 01.03. URL化

        //https://leetcode-cn.com/problems/string-to-url-lcci/
        public string ReplaceSpaces(string s, int length)
        {
            var res = new StringBuilder();
            for (int i = 0; i < s.Length && length > 0; i++)
            {
                if (s[i] == ' ')
                {
                    res.Append("%20");
                }
                else
                {
                    res.Append(s[i]);
                }

                length--;
            }

            return res.ToString();
        }

        #endregion

        #region 面试题 01.04. 回文排列

        //https://leetcode-cn.com/problems/palindrome-permutation-lcci/
        public bool CanPermutePalindrome(string s)
        {
            var set = new HashSet<char>();
            foreach (var ch in s)
            {
                if (set.Contains(ch))
                {
                    set.Remove(ch);
                }
                else
                {
                    set.Add(ch);
                }
            }

            return set.Count <= 1;
        }

        #endregion

        #region 面试题 01.09. 字符串轮转

        //https://leetcode-cn.com/problems/string-rotation-lcci/
        public bool IsFlipedString(string s1, string s2)
        {
            return s1.Length == s2.Length ? (s2 + s2).Contains(s1) : false;
        }

        #endregion

        #region 面试题 02.01. 移除重复节点

        //https://leetcode-cn.com/problems/remove-duplicate-node-lcci/
        public ListNode RemoveDuplicateNodes(ListNode head)
        {
            if (head == null)
            {
                return null;
            }

            var exists = new HashSet<int>();
            exists.Add(head.val);
            ListNode prev = head, node = head.next;
            while (node != null)
            {
                if (!exists.Add(node.val))
                {
                    prev.next = node.next;
                }
                else
                {
                    prev = node;
                }

                node = node.next;
            }

            prev.next = null;
            return head;
        }

        #endregion

        #region 1028. 从先序遍历还原二叉树

        //https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/
        public TreeNode RecoverFromPreorder(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return null;
            }

            var index = s.IndexOf('-');
            if (index < 0)
            {
                return new TreeNode(int.Parse(s));
            }

            var root = new TreeNode(int.Parse(s.Substring(0, index)));
            string left = "", right = "";
            int level = 0, rootLevel = 0;
            for (int i = index; i < s.Length; i++)
            {
                if (char.IsDigit(s[i]))
                {
                    if (rootLevel == 0)
                    {
                        rootLevel = level;
                    }
                    else if (rootLevel == level)
                    {
                        int start = index + rootLevel, end = i - rootLevel;
                        left = s.Substring(start, end - start);
                        right = s.Substring(i);
                        break;
                    }

                    level = 0;
                }
                else
                {
                    level++;
                }
            }

            if (string.IsNullOrEmpty(left))
            {
                left = s.Substring(index + rootLevel);
            }

            root.left = RecoverFromPreorder(left);
            root.right = RecoverFromPreorder(right);
            return root;
        }

        #endregion
    }
}