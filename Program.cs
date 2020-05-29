using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace leetcode
{
    class Program
    {
        static void PrintArray(int[] array)
        {
            Console.WriteLine(string.Join(",", array));
        }

        static void Main(string[] args)
        {
            Console.WriteLine(new Solution().ConstructArr(new[] {1, 2, 3, 4, 5}));
            //4,2,5,1,3
            var root = new TreeNode(1);
            root.left = new TreeNode(2);
            root.right = new TreeNode(3);
            root.right.left = new TreeNode(4);
            root.right.right = new TreeNode(5);
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
                new Program().DecodeString("100[leetcode]"));
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
        static int RemoveDuplicates(int[] nums)
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

        #region 53. 最大子序和

        //53. 最大子序和
        //https://leetcode-cn.com/problems/maximum-subarray/
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
        //1356. 根据数字二进制下 1 的数目排序
        static int CountOne(int num)
        {
            var size = 0;
            while (num != 0)
            {
                if ((num & 1) == 1)
                {
                    size++;
                }

                num >>= 1;
            }

            return size;
        }

        public static int[] SortByBits(int[] arr)
        {
            var indexs = new int[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                indexs[i] = CountOne(arr[i]);
            }

            var flag = true;
            for (int i = 0; i < arr.Length - 1; i++)
            {
                for (int j = 0; j < arr.Length - i - 1; j++)
                {
                    if (indexs[j] == indexs[j + 1])
                    {
                        if (arr[j] > arr[j + 1])
                        {
                            Swap(arr, j, j + 1);
                            flag = false;
                        }
                    }
                    else if (indexs[j] > indexs[j + 1])
                    {
                        Swap(indexs, j, j + 1);
                        Swap(arr, j, j + 1);
                        flag = false;
                    }
                }

                if (flag)
                {
                    break;
                }
            }

            return arr;
        }

        #endregion

        //https://leetcode-cn.com/problems/relative-sort-array/
        //1122. 数组的相对排序
        public static int[] RelativeSortArray(int[] arr1, int[] arr2)
        {
            var tmp = new int[1001];
            foreach (var i in arr1)
            {
                tmp[i]++;
            }

            var index = 0;
            foreach (var i in arr2)
            {
                while (tmp[i] > 0)
                {
                    arr1[index++] = i;
                    tmp[i]--;
                }
            }

            for (int i = 0; i < tmp.Length; i++)
            {
                while (tmp[i] > 0)
                {
                    arr1[index++] = i;
                    tmp[i]--;
                }
            }

            return arr1;
        }

        //https://leetcode-cn.com/problems/sort-array-by-parity-ii/
        //922. 按奇偶排序数组 II
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

        //https://leetcode-cn.com/problems/intersection-of-two-arrays/
        //349. 两个数组的交集
        public static int[] Intersection(int[] nums1, int[] nums2)
        {
            HashSet<int> set1 = new HashSet<int>(nums1), set2 = new HashSet<int>(nums2);
            set1.IntersectWith(set2);
            return set1.ToArray();
        }

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

        //|r1 - r2| + |c1 - c2|
        public int[][] AllCellsDistOrder(int R, int C, int r0, int c0)
        {
            var result = new int[R * C][];
            var i = 0;
            for (int r = 0; r < R; r++)
            {
                for (int c = 0; c < C; c++)
                {
                    result[i++] = new[] {r, c};
                }
            }

            return result.OrderBy(it => Math.Abs(it[0] - r0) + Math.Abs(it[1] - c0)).ToArray();
        }

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

                        result.Append((char) (i + 'a'));
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

                        result.Append((char) (i + 'a'));
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

            return new TreeNode(root.val) {left = MirrorTree(root.right), right = MirrorTree(root.left)};
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

        #region 面试题15. 二进制中1的个数

        //面试题15. 二进制中1的个数
        //https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/
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

        #region 面试题57. 和为s的两个数字

        //面试题57. 和为s的两个数字
        //https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/
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
                    return new[] {nums[i], nums[index]};
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
                    return new[] {nums[i], num};
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
                    return new[] {nums[start], nums[end]};
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

        #region 面试题55 - II. 平衡二叉树

        //面试题55 - II. 平衡二叉树
        //https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/
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
        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q)
        {
            if (root == null || (root.left == null && root.right == null))
            {
                return null;
            }

            if (root.val < p.val && root.val < q.val)
            {
                return lowestCommonAncestor(root.right, p, q);
            }

            if (root.val > q.val && root.val > p.val)
            {
                return lowestCommonAncestor(root.left, p, q);
            }

            //p,q节点必存在tree中，此时节点分布符号搜索二叉树，直接返回root
            return root;
        }

        #endregion

        #region 面试题68 - II. 二叉树的最近公共祖先

        public TreeNode FindChild(TreeNode root, TreeNode child)
        {
            while (true)
            {
                if (root == null)
                {
                    return null;
                }

                if (child == null || root.val == child.val)
                {
                    return root;
                }

                TreeNode node = FindChild(root.left, child);
                if (node != null) return node;
                root = root.right;
            }
        }

        //面试题68 - II. 二叉树的最近公共祖先
        //https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/
        public TreeNode LowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q)
        {
            if (root == null || (root.left == null && root.right == null))
            {
                return null;
            }

            TreeNode node = LowestCommonAncestor1(root.right, p, q);
            if (node == null)
            {
                node = LowestCommonAncestor1(root.right, p, q);
            }

            if (node != null)
            {
                return node;
            }

            TreeNode node1 = FindChild(root, p), node2 = FindChild(root, q);
            return node1 != null && node2 != null ? root : null;
        }

        #endregion

        #region 面试题52. 两个链表的第一个公共节点

        //面试题52. 两个链表的第一个公共节点
        //https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/
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

        #region 面试题53 - II. 0～n-1中缺失的数字

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

        #region 面试题12. 矩阵中的路径

        //面试题12. 矩阵中的路径
        //https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/

        bool Move(char[][] board, bool[,] flag, int x, int y, int index, string word)
        {
            if (x < 0 || x >= board.Length || y < 0 || y >= board[0].Length || flag[x, y])
            {
                return false;
            }

            if (board[x][y] == word[index])
            {
                Console.WriteLine(word[index]);
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
            }

            flag[x, y] = false;
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

        #region 面试题38. 字符串的排列

        //面试题38. 字符串的排列
        //https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/

        void Swap(char[] chars, int i, int j)
        {
            var tmp = chars[j];
            chars[j] = chars[i];
            chars[i] = tmp;
        }

        // void dfs(int x) {
        //     res.add(String.valueOf(c));
        //     HashSet<Character> set = new HashSet<>();
        //     for(int i = x; i < c.length; i++) {
        //         swap(i, x); // 交换，将 c[i] 固定在第 x 位 
        //         dfs(x + 1); // 开启固定第 x + 1 位字符
        //         swap(i, x); // 恢复交换
        //     }
        // }
        void Permutation(char[] chars, int index, ISet<string> strs)
        {
            strs.Add(new string(chars));
            for (int i = index; i < chars.Length; i++)
            {
                Swap(chars, index, i);
                Permutation(chars, index + 1, strs);
                Swap(chars, index, i);
            }
        }

        public string[] Permutation(string s)
        {
            var result = new HashSet<string>();
            var chars = s.ToCharArray();
            Permutation(chars, 0, result);
            return result.ToArray();
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
                    num += (int) Math.Pow(n % 10, 2);
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

        #region 面试题34. 二叉树中和为某一值的路径

        //面试题34. 二叉树中和为某一值的路径
        //https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/
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

        #region 98. 验证二叉搜索树

        //98. 验证二叉搜索树
        //https://leetcode-cn.com/problems/validate-binary-search-tree/
        public bool IsValidBST(TreeNode node, int rootVal, bool isLeft)
        {
            if (node == null)
            {
                return true;
            }

            if (isLeft && (node.val >= rootVal))
            {
                return false;
            }

            if (!isLeft && (node.val <= rootVal))
            {
                return false;
            }

            return IsValidBST(node.left, rootVal, isLeft) && IsValidBST(node.right, rootVal, isLeft);
        }

        public bool IsValidBST(TreeNode root)
        {
            if (root == null)
            {
                return true;
            }

            var flag = IsValidBST(root.left, root.val, true) && IsValidBST(root.right, root.val, false);
            if (!flag)
            {
                return false;
            }

            return IsValidBST(root.left) && IsValidBST(root.right);
        }

        #endregion

        #region 572. 另一个树的子树

        //572. 另一个树的子树
        //https://leetcode-cn.com/problems/subtree-of-another-tree/
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
            double low = 0, high = x, num;
            while (true)
            {
                num = (low + high) / 2;
                var pow = num * num;
                var diff = Math.Abs(pow - x);
                if (diff <= 0.0001)
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

            return (int) num;
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

            var list = new List<TreeNode>();
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
                    list.Add(root);
                    root = root.right;
                }
            }

            root = list[0];
            list[0].left = list[list.Count - 1];
            list[list.Count - 1].right = list[0];
            for (int i = 1; i < list.Count; i++)
            {
                list[i - 1].right = list[i];
                list[i].left = list[i - 1];
            }

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
            if (n <= 1)
            {
                return n;
            }

            var num = 1;
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
            var dic = new Dictionary<int, int> {{0, 1}};
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

        #region 2. 两数相加

        //https://leetcode-cn.com/problems/add-two-numbers/
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
                var set = new Dictionary<char, int> {{'a', 0}, {'e', 0}, {'i', 0}, {'o', 0}, {'u', 0}};
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

        #region 5. 最长回文子串

        //5. 最长回文子串
        //https://leetcode-cn.com/problems/longest-palindromic-substring/
        bool Check(string s, int start, int end)
        {
            while (start < end)
            {
                if (s[start] != s[end])
                {
                    return false;
                }

                start++;
                end--;
            }

            return true;
        }


        public string LongestPalindrome(string s)
        {
            if (string.IsNullOrEmpty(s))
            {
                return string.Empty;
            }

            int start = 0, len = 0;
            for (var i = 0; i < s.Length; i++)
            {
                for (var j = i; j < s.Length; j++)
                {
                    if (len >= (j - i) + 1)
                    {
                        continue;
                    }

                    if (Check(s, i, j))
                    {
                        start = i;
                        len = j - i + 1;
                    }
                }
            }

            return s.Substring(start, len);
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

        //todo 继续
        public bool IsNumber(string s)
        {
            //"+100"、"5e2"、"-123"、"3.1416"、"0123"
            if (string.IsNullOrWhiteSpace(s))
            {
                return false;
            }

            s = s.Trim();
            var allowSet = new HashSet<char> {'e', '.', '+', '-'};
            for (int i = 0; i < 10; i++)
            {
                allowSet.Add((char) ('0' + i));
            }

            for (int i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                switch (ch)
                {
                    case '+':
                    case '-':
                        if (!allowSet.Contains(ch))
                        {
                            return false;
                        }

                        allowSet.Remove('+');
                        allowSet.Remove('-');
                        break;
                    case 'e':
                        break;
                    case '.':
                        break;
                    default:
                        if (!allowSet.Contains(ch))
                        {
                            return false;
                        }

                        allowSet.Remove('+');
                        allowSet.Remove('-');
                        break;
                }
            }

            return true;
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

            Dictionary<char, int> dic = new Dictionary<char, int>(), subStr = new Dictionary<char, int>();
            foreach (var ch in t)
            {
                if (dic.ContainsKey(ch))
                {
                    dic[ch]++;
                }
                else
                {
                    dic[ch] = 1;
                }
            }

            int start = 0, end = 0, minStart = 0, minLen = int.MaxValue;
            while (end < s.Length)
            {
                var ch = s[end];
                if (dic.ContainsKey(ch))
                {
                    dic[ch]--;
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
            //return subStr.Count < set.Count ? string.Empty : s.Substring(start, len);
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

        #region 146. LRU缓存机制

        //146. LRU缓存机制
        //https://leetcode-cn.com/problems/lru-cache/
        class LRUCache
        {
            class CacheNode
            {
                public int val;
                public int key;
                public CacheNode prev;
                public CacheNode next;
            }

            private int capacity;
            private Dictionary<int, CacheNode> dic = new Dictionary<int, CacheNode>();
            private CacheNode head;
            private CacheNode tail;

            public LRUCache(int capacity)
            {
                this.capacity = capacity;
            }

            private void MoveToTail(CacheNode node)
            {
                if (node == tail)
                {
                    //尾节点
                    return;
                }

                if (node == head)
                {
                    //头节点
                    head = node.next;
                    head.prev = null;
                }
                else
                {
                    //非头尾节点，连接前后节点
                    CacheNode prev = node.prev, next = node.next;
                    prev.next = next;
                    next.prev = prev;
                }

                //将节点移动到尾节点
                node.prev = tail;
                tail.next = node;
                tail = node;
                tail.next = null;
            }

            public int Get(int key)
            {
                if (!dic.TryGetValue(key, out var node))
                {
                    return -1;
                }

                MoveToTail(node);
                return node.val;
            }

            public void Put(int key, int value)
            {
                if (dic.TryGetValue(key, out var node))
                {
                    node.val = value;
                    MoveToTail(node);
                }
                else
                {
                    dic[key] = node = new CacheNode {key = key, val = value};
                    if (tail == null)
                    {
                        head = tail = node;
                    }
                    else
                    {
                        tail.next = node;
                        node.prev = tail;
                        tail = node;
                    }
                }

                while (dic.Count > capacity)
                {
                    dic.Remove(head.key);
                    head = head.next;
                    if (head == null)
                    {
                        break;
                    }

                    head.prev = null;
                }
            }
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
                var mid = (left + right) / 2; //数组一半大小，同时也是数组中位数
                foreach (var num in nums)
                {
                    if (num <= mid)
                    {
                        size++;
                    }
                }

                if (size > mid) //小于中位数的数字个数超出数组一半大小，说明该数在数组左区间
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

        #region 面试题44. 数字序列中某一位的数字

        //面试题44. 数字序列中某一位的数字
        //https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/
        public int FindNthDigit(int n)
        {
            //range 0-9 0-9
            //range 10-99 10-180
            //range 100-999 181-2700
            throw new NotImplementedException();
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

        void TranslateNum(IList<int> bits, Dictionary<string, char> dic, int start, List<char> chars,
            HashSet<string> strs)
        {
            if (start >= bits.Count)
            {
                strs.Add(new string(chars.ToArray()));
                return;
            }

            var key = string.Empty;
            for (int i = start, end = Math.Min(start + 1, bits.Count - 1); i <= end; i++)
            {
                key += bits[i].ToString();
                if (dic.TryGetValue(key, out var ch))
                {
                    chars.Add(ch);
                    TranslateNum(bits, dic, i + 1, chars, strs);
                    chars.RemoveAt(chars.Count - 1);
                }
            }
        }

        public int TranslateNum(int num)
        {
            var sets = new HashSet<string>();
            var bits = new List<int>();
            var dic = new Dictionary<string, char>();
            for (int i = 0; i < 26; i++)
            {
                dic[i.ToString()] = (char) ('a' + i);
            }

            while (num != 0 || bits.Count == 0)
            {
                bits.Insert(0, num % 10);
                num /= 10;
            }

            TranslateNum(bits, dic, 0, new List<char>(), sets);
            return sets.Count;
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
                return new[] {-1, -1};
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
                return new[] {-1, -1};
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

            return new[] {start, rIndex};
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
    }
}