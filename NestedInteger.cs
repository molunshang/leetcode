using System.Collections.Generic;

namespace leetcode
{
    public class NestedInteger
    {
        private int val;

        private List<NestedInteger> list;

        // Constructor initializes an empty nested list.
        public NestedInteger()
        {
        }

        // Constructor initializes a single integer.
        public NestedInteger(int value)
        {
            val = value;
        }

        // @return true if this NestedInteger holds a single integer, rather than a nested list.
        public bool IsInteger()
        {
            return list == null;
        }

        // @return the single integer that this NestedInteger holds, if it holds a single integer
        // Return null if this NestedInteger holds a nested list
        public int GetInteger()
        {
            return val;
        }

        // Set this NestedInteger to hold a single integer.
        public void SetInteger(int value)
        {
            val = value;
        }

        // Set this NestedInteger to hold a nested list and adds a nested integer to it.
        public void Add(NestedInteger ni)
        {
            if (list == null)
            {
                list = new List<NestedInteger>();
            }

            list.Add(ni);
        }

        // @return the nested list that this NestedInteger holds, if it holds a nested list
        // Return null if this NestedInteger holds a single integer
        public IList<NestedInteger> GetList()
        {
            return list;
        }
    }
}