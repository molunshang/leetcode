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
                    start = Math.Min(start, i - 1);//此前位数-1，所以start=Math.Min(start,i-1)
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
}