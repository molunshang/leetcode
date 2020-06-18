using System.Collections.Generic;

//232. 用栈实现队列/面试题 03.04. 化栈为队
//https://leetcode-cn.com/problems/implement-queue-using-stacks/
//https://leetcode-cn.com/problems/implement-queue-using-stacks-lcci/
namespace leetcode
{
    public class MyQueue
    {
        private Stack<int> inStack = new Stack<int>();
        private Stack<int> outStack = new Stack<int>();

        /** Initialize your data structure here. */
        public MyQueue()
        {

        }

        /** Push element x to the back of queue. */
        public void Push(int x)
        {
            inStack.Push(x);
        }

        /** Removes the element from in front of queue and returns that element. */
        public int Pop()
        {
            if (outStack.Count <= 0)
            {
                while (inStack.Count > 1)
                {
                    outStack.Push(inStack.Pop());
                }
                return inStack.Pop();
            }
            return outStack.Pop();
        }

        /** Get the front element. */
        public int Peek()
        {
            if (outStack.Count <= 0)
            {
                while (inStack.Count > 0)
                {
                    outStack.Push(inStack.Pop());
                }
            }
            return outStack.Peek();
        }

        /** Returns whether the queue is empty. */
        public bool Empty()
        {
            return inStack.Count == 0 && outStack.Count == 0;
        }
    }
}