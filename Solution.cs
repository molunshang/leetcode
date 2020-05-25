using System;
using System.Collections.Generic;
using System.Linq;

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
        ;
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
}