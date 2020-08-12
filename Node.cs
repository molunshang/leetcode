using System.Collections.Generic;

namespace leetcode
{
    public class Node
    {
        public int val;
        public Node next;
        public Node random;
        public Node left;
        public Node right;
        public IList<Node> neighbors;
        public Node(int _val)
        {
            val = _val;
            next = null;
            random = null;
        }
    }
}