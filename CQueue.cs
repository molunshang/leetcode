using System.Collections.Generic;

//https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/
//面试题09. 用两个栈实现队列
//用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
namespace leetcode
{
    public class CQueue
    {
        private Stack<int> items = new Stack<int>();
        private Stack<int> temp = new Stack<int>();

        public CQueue()
        {
        }

        public void AppendTail(int value)
        {
            items.Push(value);
        }

        public int DeleteHead()
        {
            if (items.Count <= 0)
            {
                return -1;
            }

            while (items.Count > 1)
            {
                temp.Push(items.Pop());
            }

            var result = items.Pop();
            while (temp.Count > 0)
            {
                items.Push(temp.Pop());
            }

            return result;
        }
    }
}