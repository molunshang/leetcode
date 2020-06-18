using System.Collections.Generic;

//面试题 03.03. 堆盘子
//https://leetcode-cn.com/problems/stack-of-plates-lcci/
namespace leetcode
{
    public class StackOfPlates
    {
        private IList<Stack<int>> stacks = new List<Stack<int>>();
        private int limit;

        public StackOfPlates(int cap)
        {
            limit = cap;
        }

        public void Push(int val)
        {
            if (limit <= 0)
            {
                return;
            }

            Stack<int> last;
            if (stacks.Count <= 0)
            {
                last = new Stack<int>();
                stacks.Add(last);
            }
            else
            {
                last = stacks[stacks.Count - 1];
                if (last.Count >= limit)
                {
                    last = new Stack<int>();
                    stacks.Add(last);
                }
            }

            last.Push(val);
        }

        public int Pop()
        {
            if (limit <= 0 || stacks.Count <= 0)
            {
                return -1;
            }

            var last = stacks[stacks.Count - 1];
            var res = last.Pop();
            if (last.Count <= 0)
            {
                stacks.RemoveAt(stacks.Count - 1);
            }

            return res;
        }

        public int PopAt(int index)
        {
            if (limit <= 0 || index >= stacks.Count)
            {
                return -1;
            }

            var stack = stacks[index];
            var res = stack.Pop();
            if (stack.Count <= 0)
            {
                stacks.RemoveAt(index);
            }

            return res;
        }
    }
}