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
            // var array = new[] {4, 2, 5, 7, 1};
            // SubSeq(array, new List<int>(), 0);
            SortString("aaaabbbbcccc");
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

        static TreeNode BuildTree(int[] preorder, int[] inorder)
        {
            var root = new TreeNode(preorder[0]);

            return root;
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
            if (result.Count == 3)
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
    }
}