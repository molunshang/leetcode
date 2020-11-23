using System.Collections.Generic;

//面试题 03.05. 栈排序
//https://leetcode-cn.com/problems/sort-of-stacks-lcci/
namespace leetcode
{
    public class SortedStack
    {
        private Stack<int> data = new Stack<int>();
        private Stack<int> min = new Stack<int>();

        public SortedStack()
        {
        }

        public void Push(int val)
        {
            if (min.Count == 0 || min.Peek() >= val)
            {
                min.Push(val);
                return;
            }

            while (min.Count > 0 && min.Peek() < val)
            {
                data.Push(min.Pop());
            }

            data.Push(val);
            while (data.Count > 0)
            {
                min.Push(data.Pop());
            }
        }

        public void Pop()
        {
            if (min.Count <= 0)
            {
                return;
            }

            min.Pop();
        }

        public int Peek()
        {
            if (min.TryPeek(out var num))
            {
                return num;
            }

            return -1;
        }

        public bool IsEmpty()
        {
            return min.Count == 0;
        }
    }
}