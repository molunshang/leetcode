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
            System.Console.WriteLine(string.Join(",", array));
        }

        static void Main(string[] args)
        {
            Console.WriteLine(new Program().IsHappy(7));
        }

        static int MaxProfit(int[] prices)
        {
            var n = int.MaxValue;
            for (int i = 0; i < prices.Length; i++)
            {
                for (int j = i + 1; j < prices.Length; j++)
                {
                    var m = prices[i] - prices[j];
                    if (m < n)
                    {
                        n = m;
                    }
                }
            }

            return n > 0 ? 0 : -n;
        }

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

        //https://leetcode-cn.com/problems/sort-integers-by-the-number-of-1-bits/
        //1356. 根据数字二进制下 1 的数目排序
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
                    result[i++] = new int[] {r, c};
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
        public static int MinArray(int[] numbers)
        {
            int start = 0, end = numbers.Length - 1;
            while (numbers[start] >= numbers[end])
            {
                if (end - start == 1)
                {
                    start = end;
                    break;
                }

                var mid = (start + end) / 2;
                if (numbers[mid] == numbers[start] && numbers[start] == numbers[end])
                {
                    var result = numbers[start];
                    for (var i = start + 1; i <= end; i++)
                    {
                        if (result > numbers[i])
                        {
                            result = numbers[i];
                        }
                    }

                    return result;
                }

                if (numbers[mid] >= numbers[start])
                {
                    start = mid;
                }
                else if (numbers[mid] <= numbers[end])
                {
                    end = mid;
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

            public MinStack()
            {
            }

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

        //todo 待完成

        #region 面试题60. n个骰子的点数

        //面试题60. n个骰子的点数
        //https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/

        public static void Sequence(IList<int> nums, IList<int> sequence, IDictionary<int, int> countDic, int index,
            int n)
        {
            if (sequence.Count == n)
            {
                Console.WriteLine(string.Join(',', sequence));
                var sum = sequence.Sum();
                if (countDic.ContainsKey(sum))
                {
                    countDic[sum]++;
                }
                else
                {
                    countDic[sum] = 1;
                }

                return;
            }

            if (index >= nums.Count)
            {
                return;
            }

            for (int i = index; i < nums.Count; i++)
            {
                sequence.Add(nums[i]);
                Sequence(nums, sequence, countDic, i + 6, n);
                sequence.RemoveAt(sequence.Count - 1);
            }
        }

        public double[] TwoSum(int n)
        {
            var items = new int[n, 6];
            return null;
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
            else if (root.val > q.val && root.val > p.val)
            {
                return lowestCommonAncestor(root.left, p, q);
            }
            else
            {
                //p,q节点必存在tree中，此时节点分布符号搜索二叉树，直接返回root
                return root;
            }
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
        public int CuttingRope(int n)
        {
            if (n <= 2)
            {
                return 1;
            }

            var max = 0;
            for (int i = 1; i < n; i++)
            {
                max = Math.Max(Math.Max(i * CuttingRope(n - i), i * (n - i)), max);
            }

            return max;
        }

        #endregion
    }
}