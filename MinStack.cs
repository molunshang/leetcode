using System.Collections.Generic;

//155. 最小栈
//https://leetcode-cn.com/problems/min-stack/
public class MinStack
{
    /** 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。 */
    private Stack<int> _stack = new Stack<int>();

    private Stack<int> _min = new Stack<int>();

    public MinStack()
    {
    }

    public void Push(int x)
    {
        _stack.Push(x);
        if (_min.Count > 0 && x >= _min.Peek())
        {
            return;
        }
        _min.Push(x);
    }

    public void Pop()
    {
        if (_stack.Pop() <= _min.Peek())
        {
            _min.Pop();
        }
    }

    public int Top()
    {
        return _stack.Peek();
    }

    public int GetMin()
    {
        return _min.Peek();
    }
}