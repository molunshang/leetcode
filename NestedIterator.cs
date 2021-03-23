using System.Collections.Generic;

namespace leetcode
{
    #region 341. 扁平化嵌套列表迭代器

    //https://leetcode-cn.com/problems/flatten-nested-list-iterator/

    public class NestedIterator
    {
        private Queue<int> data = new Queue<int>();

        void Dfs(IList<NestedInteger> nestedList)
        {
            foreach (var nested in nestedList)
            {
                if (nested.IsInteger())
                {
                    data.Enqueue(nested.GetInteger());
                }
                else
                {
                    Dfs(nested.GetList());
                }
            }
        }

        public NestedIterator(IList<NestedInteger> nestedList)
        {
            Dfs(nestedList);
        }

        public bool HasNext()
        {
            return data.Count > 0;
        }

        public int Next()
        {
            return data.Dequeue();
        }
    }

    #endregion
}