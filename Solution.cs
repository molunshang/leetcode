using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public class Solution
{
    public TreeNode BuildTree(int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd)
    {
        if (preStart > preEnd)
        {
            return null;
        }

        var root = new TreeNode(preorder[preStart]);
        int index = inStart;
        while (index <= inEnd)
        {
            if (root.val == inorder[index])
            {
                break;
            }

            index++;
        }

        root.left = BuildTree(preorder, preStart + 1, preStart + (index - inStart), inorder, inStart, index - 1);
        root.right = BuildTree(preorder, preStart + (index - inStart) + 1, preEnd, inorder, index + 1, inEnd);
        return root;
    }

    public TreeNode BuildTree(int[] preorder, int[] inorder)
    {
        //根据前序遍历找出根节点，然后根据后续遍历找出左右子树，递归构建
        return BuildTree(preorder, 0, preorder.Length - 1, inorder, 0, inorder.Length - 1);
    }

    public class CQueue
    {
        private Stack<int> inStack = new Stack<int>();
        private Stack<int> outStack = new Stack<int>();

        public CQueue()
        {
        }

        public void AppendTail(int value)
        {
            inStack.Push(value);
        }

        public int DeleteHead()
        {
            if (inStack.Count <= 0 && outStack.Count <= 0)
            {
                return -1;
            }

            if (outStack.Count <= 0)
            {
                while (inStack.Count > 0)
                {
                    outStack.Push(inStack.Pop());
                }
            }

            return outStack.Pop();
        }
    }

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

    public bool Find(char[][] board, bool[,] flags, int x, int y, string word, int index)
    {
        if (x < 0 || x >= board.Length || y < 0 || y >= board[0].Length || flags[x, y])
        {
            return false;
        }

        if (board[x][y] != word[index])
            return false;
        if (index == word.Length - 1)
        {
            return true;
        }

        flags[x, y] = true;
        var flag = Find(board, flags, x + 1, y, word, index + 1) || Find(board, flags, x - 1, y, word, index + 1) ||
                   Find(board, flags, x, y + 1, word, index + 1) || Find(board, flags, x, y - 1, word, index + 1);
        if (!flag)
        {
            flags[x, y] = false;
        }

        return flag;
    }

    public bool Exist(char[][] board, string word)
    {
        var flags = new bool[board.Length, board[0].Length];
        for (var i = 0; i < board.Length; i++)
        {
            var chars = board[i];
            for (var j = 0; j < chars.Length; j++)
            {
                if (Find(board, flags, i, j, word, 0))
                {
                    return true;
                }
            }
        }

        return false;
    }

    //1.位移法
    public int HammingWeight(uint n)
    {
        var res = 0;
        while (n != 0)
        {
            if ((n & 1) == 1)
            {
                res++;
            }

            n >>= 1;
        }

        return res;
    }

    //2.n&(n-1)消除最低位的1
    public int HammingWeight1(uint n)
    {
        var res = 0;
        while (n != 0)
        {
            res++;
            n &= n - 1;
        }

        return res;
    }

    public double MyPow(double x, int n)
    {
        double PowFunc(double y, int k)
        {
            if (k == 0)
            {
                return 1.0;
            }

            var num = PowFunc(y, k / 2);
            return (k & 1) == 1 ? num * num * y : num * num;
        }

        var res = PowFunc(x, n);
        return n > 0 ? res : 1.0 / res;
    }

    public int[] PrintNumbers(int n)
    {
        var num = 0;
        while (n > 0)
        {
            num = num * 10 + 9;
            n--;
        }

        var res = new int[num];
        for (int i = 0; i < res.Length; i++)
        {
            res[i] = i + 1;
        }

        return res;
    }

    public int[] PrintNumbers1(int n)
    {
        var chars = new char[n];
        for (int i = 0; i < chars.Length; i++)
        {
            chars[i] = '0';
        }

        var result = new List<string>();
        int start = chars.Length - 1, count = 1;
        while (true)
        {
            for (int i = chars.Length - 1; i >= 0; i--)
            {
                //i位数字为9，此时+1满10，i==0，说明已经遍历到最高位，直接返回，否则当前位=0，该位的前1位+1
                if (chars[i] == '9')
                {
                    if (i == 0)
                    {
                        return result.Select(int.Parse).ToArray();
                    }

                    chars[i] = '0';
                    start = Math.Min(start, i - 1); //此前位数-1，所以start=Math.Min(start,i-1)
                    count = chars.Length - start;
                }
                else
                {
                    chars[i]++;
                    result.Add(new string(chars, start, count));
                    //+1从最低位开始，所以每次+1操作后跳出，重新从最低位开始
                    break;
                }
            }
        }
    }

    public ListNode DeleteNode(ListNode head, int val)
    {
        if (head == null)
        {
            return null;
        }

        if (head.val == val)
        {
            return head.next;
        }

        ListNode node = head.next, prev = head;
        while (node != null)
        {
            if (node.val == val)
            {
                prev.next = node.next;
                break;
            }

            prev = node;
            node = node.next;
        }

        return head;
    }

    //首尾指针扫描，找到开头的奇数和尾部的偶数，然后进行交换
    public int[] Exchange(int[] nums)
    {
        int start = 0, end = nums.Length - 1;
        while (start < end)
        {
            while (start < end && (nums[start] & 1) == 1)
            {
                start++;
            }

            while (start < end && (nums[end] & 1) == 0)
            {
                end--;
            }

            if (start < end)
            {
                var tmp = nums[start];
                nums[start] = nums[end];
                nums[end] = tmp;
                start++;
                end--;
            }
        }

        return nums;
    }

    //链表中倒数第k个节点
    //快慢指针 快指针先走k部，慢指针再走，当快指针为null，慢指针即导致k节点
    public ListNode GetKthFromEnd(ListNode head, int k)
    {
        if (head == null)
        {
            return null;
        }

        ListNode fast = head, slow = head;
        while (k > 0 && fast != null)
        {
            fast = fast.next;
            k--;
        }

        if (k > 0)
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

    public ListNode ReverseList(ListNode head)
    {
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

    public ListNode MergeTwoLists(ListNode l1, ListNode l2)
    {
        if (l1 == null)
        {
            return l2;
        }

        if (l2 == null)
        {
            return l1;
        }

        ListNode head = null, newNode = null;
        while (l1 != null && l2 != null)
        {
            int val;
            if (l1.val < l2.val)
            {
                val = l1.val;
                l1 = l1.next;
            }
            else
            {
                val = l2.val;
                l2 = l2.next;
            }

            if (head == null)
            {
                head = newNode = new ListNode(val);
            }
            else
            {
                newNode.next = new ListNode(val);
                newNode = newNode.next;
            }
        }

        if (l1 != null)
        {
            newNode.next = l1;
        }

        if (l2 != null)
        {
            newNode.next = l2;
        }

        return head;
    }

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

        if (a.val != b.val)
        {
            return false;
        }

        return IsSame(a.left, b.left) && IsSame(a.right, b.right);
    }

    public bool IsSubStructure(TreeNode a, TreeNode b)
    {
        if (a == null || b == null)
        {
            return false;
        }

        if (a.val == b.val && IsSame(a.left, b.left) && IsSame(a.right, b.right))
        {
            return true;
        }

        return IsSubStructure(a.left, b) || IsSubStructure(a.right, b);
    }

    public TreeNode MirrorTree(TreeNode root)
    {
        return root == null
            ? null
            : new TreeNode(root.val) {left = MirrorTree(root.right), right = MirrorTree(root.left)};
    }

    bool IsSymmetric(TreeNode left, TreeNode right)
    {
        if (left == null && right == null)
        {
            return true;
        }

        if (left == null || right == null || left.val != right.val)
        {
            return false;
        }

        return IsSymmetric(left.left, right.right) && IsSymmetric(left.right, right.left);
    }

    public bool IsSymmetric(TreeNode root)
    {
        return root == null || IsSymmetric(root.left, root.right);
    }

    public IList<int> SpiralOrder(int[][] matrix)
    {
        if (matrix == null || matrix.Length <= 0)
        {
            return new int[0];
        }

        var result = new int[matrix.Length * matrix[0].Length];
        int index = 0, x0 = 0, x1 = matrix.Length - 1, y0 = 0, y1 = matrix[0].Length - 1, type = 0;
        while (index < result.Length)
        {
            switch (type)
            {
                case 0:
                    for (int i = y0; i <= y1; i++)
                    {
                        result[index++] = matrix[x0][i];
                    }

                    x0++;
                    type = 1;
                    break;
                case 1:
                    for (int i = x0; i <= x1; i++)
                    {
                        result[index++] = matrix[i][y1];
                    }

                    y1--;
                    type = 2;
                    break;
                case 2:
                    for (int i = y1; i >= y0; i--)
                    {
                        result[index++] = matrix[x1][i];
                    }

                    x1--;
                    type = 3;
                    break;
                case 3:
                    for (int i = x1; i >= x0; i--)
                    {
                        result[index++] = matrix[i][y0];
                    }

                    y0++;
                    type = 0;
                    break;
            }
        }

        return result;
    }

    public class MinStack
    {
        private Stack<int> data = new Stack<int>();
        private Stack<int> min = new Stack<int>();

        /** initialize your data structure here. */
        public MinStack()
        {
        }

        public void Push(int x)
        {
            min.Push(min.Count <= 0 || x < min.Peek() ? x : min.Peek());
            data.Push(x);
        }

        public void Pop()
        {
            data.Pop();
            min.Pop();
        }

        public int Top()
        {
            return data.Peek();
        }

        public int Min()
        {
            return min.Peek();
        }
    }

    public bool ValidateStackSequences(int[] pushed, int[] popped)
    {
        var check = new Stack<int>();
        int i = 0, j = 0;
        while (i < pushed.Length && j < popped.Length)
        {
            while (j < popped.Length && check.TryPeek(out var num) && num == popped[j])
            {
                j++;
                check.Pop();
            }

            if (pushed[i] == popped[j])
            {
                i++;
                j++;
            }
            else
            {
                check.Push(pushed[i++]);
            }
        }

        while (check.Count > 0 && j < popped.Length && check.Peek() == popped[j])
        {
            j++;
            check.Pop();
        }

        return check.Count <= 0;
    }

    public int[] LevelOrder(TreeNode root)
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
            result.Add(root.val);
            if (root.left != null)
            {
                queue.Enqueue(root.left);
            }

            if (root.right != null)
            {
                queue.Enqueue(root.right);
            }
        }

        return result.ToArray();
    }

    public IList<IList<int>> LevelOrder2(TreeNode root)
    {
        if (root == null)
        {
            return new List<int>[0];
        }

        var result = new List<IList<int>>();
        var queue = new Queue<TreeNode>();
        var items = new List<int>();
        queue.Enqueue(root);
        var size = queue.Count;
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

            result.Add(items.ToArray());
            items.Clear();
            size = queue.Count;
        }

        return result;
    }

    public IList<IList<int>> LevelOrder3(TreeNode root)
    {
        if (root == null)
        {
            return new int[0][];
        }

        var result = new List<IList<int>>();
        var dequeue = new LinkedList<TreeNode>();
        var items = new List<int>();
        dequeue.AddLast(root);
        int size = dequeue.Count, level = 0;
        while (dequeue.Count > 0)
        {
            if ((level & 1) == 1) //奇数
            {
                while (size > 0)
                {
                    size--;
                    root = dequeue.Last.Value;
                    dequeue.RemoveLast();
                    items.Add(root.val);
                    if (root.right != null)
                    {
                        dequeue.AddFirst(root.right);
                    }

                    if (root.left != null)
                    {
                        dequeue.AddFirst(root.left);
                    }
                }
            }
            else //偶数
            {
                while (size > 0)
                {
                    size--;
                    root = dequeue.First.Value;
                    dequeue.RemoveFirst();
                    items.Add(root.val);
                    if (root.left != null)
                    {
                        dequeue.AddLast(root.left);
                    }

                    if (root.right != null)
                    {
                        dequeue.AddLast(root.right);
                    }
                }
            }

            level++;
            result.Add(items.ToArray());
            items.Clear();
            size = dequeue.Count;
        }

        return result;
    }

    bool VerifyPostorder(int[] postorder, int start, int end)
    {
        while (true)
        {
            if (start >= end)
            {
                return true;
            }

            int root = postorder[end], mid = -1;
            for (int i = end - 1; i >= start; i--)
            {
                if (postorder[i] < root)
                {
                    mid = i;
                    break;
                }
            }

            if (mid < 0)
            {
                //不存在左子树
                end = end - 1;
                continue;
            }

            //检测左子树是否符合要求
            for (int i = start; i < mid; i++)
            {
                if (postorder[i] > root)
                {
                    return false;
                }
            }

            return VerifyPostorder(postorder, start, mid) && VerifyPostorder(postorder, mid + 1, end - 1);
        }
    }

    public bool VerifyPostorder(int[] postorder)
    {
        if (postorder == null || postorder.Length == 0)
        {
            return true;
        }

        return VerifyPostorder(postorder, 0, postorder.Length - 1);
    }

    public IList<IList<int>> PathSum(TreeNode root, int sum)
    {
        if (root == null)
        {
            return new List<int>[0];
        }

        var stack = new Stack<TreeNode>();
        var result = new List<IList<int>>();
        var path = new List<int>();
        var total = 0;
        var flag = false;
        while (stack.Count > 0 || root != null)
        {
            while (root != null)
            {
                total += root.val;
                path.Add(root.val);
                if (total == sum && root.left == null && root.right == null)
                {
                    result.Add(path.ToArray());
                }

                stack.Push(root);
                root = root.left;
                flag = false;
            }

            if (flag)
            {
                var rmIndex = path.Count - 1;
                total -= path[rmIndex];
                path.RemoveAt(rmIndex);
            }

            if (stack.TryPop(out root))
            {
                flag = true;
                root = root.right;
            }
        }

        return result;
    }

    void PathSum(TreeNode root, IList<IList<int>> result, List<int> path, int sum)
    {
        if (root == null)
        {
            return;
        }

        sum -= root.val;
        path.Add(root.val);
        if (sum == 0 && root.left == null && root.right == null)
        {
            result.Add(path.ToArray());
        }
        else
        {
            PathSum(root.left, result, path, sum);
            PathSum(root.right, result, path, sum);
        }

        path.RemoveAt(path.Count - 1);
    }

    public IList<IList<int>> PathSum1(TreeNode root, int sum)
    {
        if (root == null)
        {
            return new List<int>[0];
        }

        var result = new List<IList<int>>();
        var path = new List<int>();
        PathSum(root, result, path, sum);
        return result;
    }

    Node CopyNode(Node node, Dictionary<Node, Node> dic)
    {
        if (node == null)
        {
            return null;
        }

        if (dic.TryGetValue(node, out var cpNode))
        {
            return cpNode;
        }

        dic[node] = cpNode = new Node(node.val);
        cpNode.next = CopyNode(node.next, dic);
        cpNode.random = CopyNode(node.random, dic);
        return cpNode;
    }

    public Node CopyRandomList(Node head)
    {
        return CopyNode(head, new Dictionary<Node, Node>());
    }

    public int MajorityElement(int[] nums)
    {
        int size = 1, num = nums[0];
        for (var i = 1; i < nums.Length; i++)
        {
            if (size == 0)
            {
                num = nums[i];
                size++;
                continue;
            }

            if (nums[i] == num)
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

    void BuildHeap(int[] arr, int parentId)
    {
        while (true)
        {
            int left = (parentId << 1) + 1, right = left + 1;
            if (left >= arr.Length)
            {
                break;
            }

            if (right < arr.Length && arr[left] < arr[right])
            {
                left = right;
            }

            if (arr[left] > arr[parentId])
            {
                var tmp = arr[parentId];
                arr[parentId] = arr[left];
                arr[left] = tmp;
                parentId = left;
            }
            else
            {
                break;
            }
        }
    }

    void MoveDown(int[] arr, int index)
    {
        var target = arr[index];
        while (true)
        {
            int left = (index << 1) + 1, right = left + 1;
            if (left >= arr.Length)
            {
                break;
            }

            if (right < arr.Length && arr[left] < arr[right])
            {
                left = right;
            }

            if (arr[left] > target)
            {
                arr[index] = arr[left];
                index = left;
            }
            else
            {
                break;
            }
        }

        arr[index] = target;
    }

    public int[] GetLeastNumbers(int[] arr, int k)
    {
        if (arr == null || arr.Length <= k)
        {
            return arr;
        }

        if (k <= 0)
        {
            return new int[0];
        }

        var heap = new int[k];
        Array.Copy(arr, heap, k);
        //1.构建大顶堆
        for (int i = k / 2; i >= 0; i--)
        {
            BuildHeap(heap, i);
        }

        for (int i = k; i < arr.Length; i++)
        {
            if (arr[i] < heap[0])
            {
                //2.小于堆顶元素，替换，重新调整为大顶堆
                heap[0] = arr[i];
                MoveDown(heap, 0);
            }
        }

        return heap;
    }

    public ListNode GetIntersectionNode(ListNode headA, ListNode headB)
    {
        int lenA = 0, lenB = 0;
        ListNode node1 = headA, node2 = headB;
        while (node1 != null && node2 != null)
        {
            lenA++;
            lenB++;
            node1 = node1.next;
            node2 = node2.next;
        }

        while (node1 != null)
        {
            lenA++;
            node1 = node1.next;
        }

        while (node2 != null)
        {
            lenB++;
            node2 = node2.next;
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

    public int Search(int[] nums, int target)
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

    public int MissingNumber(int[] nums)
    {
        //由数组特点可知，nums[i]==i,如果nums[i]!=i，说明数组缺失
        //如果是后半段丢失，此时 nums[mid]==mid
        //如果是前半段丢失，此时 nums[mid]>mid(前段丢失，则后段向前替换，此时下标不变，数组内数字变大)
        int start = 0, end = nums.Length - 1;
        while (start <= end)
        {
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

    public char FirstUniqChar(string s)
    {
        var flags = new int[26];
        foreach (var ch in s)
        {
            flags[ch - 'a']++;
        }

        foreach (var ch in s)
        {
            if (flags[ch - 'a'] == 1)
            {
                return ch;
            }
        }

        return ' ';
    }

    void KthLargest(TreeNode root, ref int k, ref int result)
    {
        if (root == null)
        {
            return;
        }

        KthLargest(root.right, ref k, ref result);
        if (k == 0)
        {
            return;
        }

        k--;
        if (k == 0)
        {
            result = root.val;
            return;
        }

        KthLargest(root.left, ref k, ref result);
    }

    public int KthLargest(TreeNode root, int k)
    {
        var res = 0;
        KthLargest(root, ref k, ref res);
        return res;
    }

    public int MaxDepth(TreeNode root)
    {
        if (root == null)
        {
            return 0;
        }

        return Math.Max(MaxDepth(root.left), MaxDepth(root.right)) + 1;
    }

    public bool IsBalanced(TreeNode root)
    {
        if (root == null)
        {
            return true;
        }

        if (!IsBalanced(root.left) || !IsBalanced(root.right))
        {
            return false;
        }

        int left = MaxDepth(root.left), right = MaxDepth(root.right);
        return Math.Abs(left - right) <= 1;
    }

    public int[] TwoSum(int[] nums, int target)
    {
        int start = 0, end = nums.Length - 1;
        while (start < end)
        {
            var num = nums[start] + nums[end];
            if (num == target)
            {
                return new[] {nums[start], nums[end]};
            }

            if (num < target)
            {
                start++;
            }
            else
            {
                end--;
            }
        }

        return new int[0];
    }

    public int[][] FindContinuousSequence(int target)
    {
        var result = new List<int[]>();
        var seqs = new Queue<int>();
        for (int i = 1, j = (target + 1) / 2; i <= j; i++)
        {
            seqs.Enqueue(i);
            target -= i;
            while (seqs.Count > 0 && target < 0)
            {
                target += seqs.Dequeue();
            }

            if (target == 0)
            {
                result.Add(seqs.ToArray());
            }
        }

        return result.ToArray();
    }

    public string ReverseWords(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return string.Empty;
        }

        var result = new StringBuilder();
        var len = 0;
        for (int i = s.Length - 1; i >= 0; i--)
        {
            var ch = s[i];
            if (ch != ' ')
            {
                len++;
            }
            else
            {
                if (len <= 0)
                {
                    continue;
                }

                result.Append(s, i + 1, len).Append(' ');
                len = 0;
            }
        }

        if (len > 0)
        {
            return result.Append(s, 0, len).ToString();
        }

        return result.Length <= 0 ? string.Empty : result.ToString(0, result.Length - 1);
    }

    public int[] MaxSlidingWindow(int[] nums, int k)
    {
        if (nums == null || nums.Length <= 1)
        {
            return nums;
        }

        var result = new List<int>();
        var slice = new LinkedList<int>();
        foreach (var num in nums)
        {
            var count = slice.Count;
            while (slice.Count > 0 && slice.Last.Value < num)
            {
                slice.RemoveLast();
            }

            while (slice.Count < count)
            {
                slice.AddLast(num);
            }

            slice.AddLast(num);
            if (slice.Count == k)
            {
                result.Add(slice.First.Value);
                slice.RemoveFirst();
            }
        }

        return result.ToArray();
    }

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

            var diff = nums[end] - nums[end - 1];
            while (diff > 1 && start < end)
            {
                if (nums[start] != 0)
                {
                    return false;
                }

                start++;
                diff--;
            }

            end--;
        }

        return true;
    }

    public int LastRemaining(int n, int m)
    {
        var list = new List<int>();
        for (int i = 0; i < n; i++)
        {
            list.Add(i);
        }

        var rmIndex = (m - 1) % list.Count;
        while (list.Count > 1)
        {
            list.RemoveAt(rmIndex);
            rmIndex = (rmIndex + m - 1) % list.Count;
        }

        return list[0];
    }

    public int MaxProfit(int[] prices)
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

    public int[] ConstructArr(int[] a)
    {
        if (a.Length <= 0)
        {
            return new int[0];
        }

        var result = new int[a.Length];
        int[] prev = new int[a.Length], next = new int[a.Length];
        prev[0] = 1;
        next[a.Length - 1] = 1;
        for (int i = 1, j = a.Length - 2; i < a.Length; i++, j--)
        {
            prev[i] = prev[i - 1] * a[i - 1];
            next[j] = next[j + 1] * a[j + 1];
        }

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = prev[i] * next[i];
        }

        return result;
    }

    public int StrToInt(string str)
    {
        int res = 0, max = int.MaxValue / 10;
        bool flag = true, ackFlag = false;
        ;
        foreach (var ch in str)
        {
            if (char.IsDigit(ch))
            {
                if (res > max)
                {
                    return flag ? int.MaxValue : int.MinValue;
                }

                var newVal = res * 10 + ch - '0';
                if (newVal < res)
                {
                    return flag ? int.MaxValue : int.MinValue;
                }

                res = newVal;
                ackFlag = true;
            }
            else if (!ackFlag)
            {
                if (ch == '+' || ch == '-')
                {
                    ackFlag = true;
                    flag = ch == '+';
                }
                else if (ch == ' ')
                {
                    continue;
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

        return flag ? res : -res;
    }

    public int CuttingRope(int n)
    {
        if (n <= 2)
        {
            return 1;
        }

        var max = 0;
        for (int i = 1; i < n; i++)
        {
            var num = Math.Max(CuttingRope(n - i) * i, (n - i) * i);
            max = Math.Max(num, max);
        }

        return max;
    }

    // int FindTopK(int[] nums1, int[] nums2, int k)
    // {
    //     var index = (k + 1) / 2;
    //     int i1Start = 0,
    //         i1End = Math.Min(nums1.Length - 1, (k + 1) / 2),
    //         i2Start = 0,
    //         i2End = Math.Min(nums2.Length - 1, (k + 1) / 2);
    //     while (true)
    //     {
    //         if (nums1[index] > nums2[index])
    //         {
    //             //nums1的范围在0-index
    //         }
    //         else
    //         {
    //             //nums2的范围在
    //         }
    //     }
    // }
    public void QuickSort(int[] nums, int start, int end)
    {
        while (true)
        {
            if (start >= end)
            {
                return;
            }

            int left = start, right = end;
            var target = nums[start];
            while (start < end)
            {
                while (start < end && target <= nums[end])
                {
                    end--;
                }

                if (start < end)
                {
                    nums[start++] = nums[end];
                }

                while (start < end && target >= nums[start])
                {
                    start++;
                }

                if (start < end)
                {
                    nums[end--] = nums[start];
                }
            }

            nums[start] = target;
            QuickSort(nums, left, start - 1);
            start = start + 1;
            end = right;
        }
    }

    public void QuickSort(int[] nums)
    {
        QuickSort(nums, 0, nums.Length - 1);
    }
}