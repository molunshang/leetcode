using System.Collections.Generic;

//面试题 03.06. 动物收容所
//https://leetcode-cn.com/problems/animal-shelter-lcci/
namespace leetcode
{
    public class AnimalShelf
    {
        private LinkedList<int[]> queueList = new LinkedList<int[]>();

        public AnimalShelf()
        {
        }

        public void Enqueue(int[] animal)
        {
            queueList.AddLast(animal);
        }

        public int[] DequeueAny()
        {
            if (queueList.Count <= 0)
            {
                return new[] {-1, -1};
            }

            var res = queueList.First;
            queueList.RemoveFirst();
            return res.Value;
        }

        int[] DequeueAny(int type)
        {
            if (queueList.Count <= 0)
            {
                return new[] {-1, -1};
            }

            var res = queueList.First;
            while (res != null && res.Value[0] != type)
            {
                res = res.Next;
            }

            if (res == null)
            {
                return new[] {-1, -1};
            }

            queueList.Remove(res);
            return res.Value;
        }

        public int[] DequeueDog()
        {
            return DequeueAny(1);
        }

        public int[] DequeueCat()
        {
            return DequeueAny(0);
        }
    }
}