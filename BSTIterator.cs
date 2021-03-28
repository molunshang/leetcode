using System.Collections.Generic;

namespace leetcode
{

    #region 173. ¶þ²æËÑË÷Ê÷µü´úÆ÷

    //https://leetcode-cn.com/problems/binary-search-tree-iterator/
    public class BSTIterator
    {
        private Stack<TreeNode> stack = new Stack<TreeNode>();
        private TreeNode current;

        public BSTIterator(TreeNode root)
        {
            current = root;
        }

        /** @return the next smallest number */
        public int Next()
        {
            var num = -1;
            if (stack.Count > 0 || current != null)
            {
                while (current != null)
                {
                    stack.Push(current);
                    current = current.left;
                }

                current = stack.Pop();
                num = current.val;
                current = current.right;
            }

            return num;
        }

        /** @return whether we have a next smallest number */
        public bool HasNext()
        {
            return stack.Count > 0 || current != null;
        }
    }

    #endregion

}