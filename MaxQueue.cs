using System.Collections.Generic;

//https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/
//面试题59 - II. 队列的最大值
namespace leetcode
{
    public class MaxQueue
    {
        private Queue<int> data = new Queue<int>();
        private LinkedList<int> max = new LinkedList<int>();

        public MaxQueue()
        {
        }

        public int Max_value()
        {
            return data.Count <= 0 ? -1 : max.First.Value;
        }

        public void Push_back(int value)
        {
            data.Enqueue(value);
            while (max.Count > 0 && max.Last.Value < value)
            {
                max.RemoveLast();
            }

            max.AddLast(value);
        }

        public int Pop_front()
        {
            if (data.Count <= 0)
            {
                return -1;
            }

            var res = data.Dequeue();
            if (res == max.First.Value)
            {
                max.RemoveFirst();
            }

            return res;
        }
    }
}