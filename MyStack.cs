using System.Collections.Generic;

//https://leetcode-cn.com/problems/implement-stack-using-queues/
//225. 用队列实现栈
public class MyStack
{
    private Queue<int> inQueue = new Queue<int>();
    private Queue<int> outQueue = new Queue<int>();
    /** Initialize your data structure here. */
    public MyStack()
    {

    }

    /** Push element x onto stack. */
    public void Push(int x)
    {
        inQueue.Enqueue(x);
    }

    /** Removes the element on top of the stack and returns that element. */
    public int Pop()
    {
        if (inQueue.Count <= 0)
        {
            while (outQueue.Count > 1)
            {
                inQueue.Enqueue(outQueue.Dequeue());
            }
            return outQueue.Dequeue();
        }
        while (inQueue.Count > 1)
        {
            outQueue.Enqueue(inQueue.Dequeue());
        }
        return inQueue.Dequeue();
    }

    /** Get the top element. */
    public int Top()
    {
        if (inQueue.Count <= 0)
        {
            while (outQueue.Count > 0)
            {
                inQueue.Enqueue(outQueue.Dequeue());
            }
        }
        while (inQueue.Count > 1)
        {
            outQueue.Enqueue(inQueue.Dequeue());
        }
        return inQueue.Peek();
    }

    /** Returns whether the stack is empty. */
    public bool Empty()
    {
        return inQueue.Count <= 0 && outQueue.Count <= 0;
    }
}