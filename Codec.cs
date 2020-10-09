using System.Collections.Generic;
using System.Text;
using System.Xml;

//https://leetcode-cn.com/problems/serialize-and-deserialize-bst/
//449. 序列化和反序列化二叉搜索树
namespace leetcode
{
    public class Codec
    {
        // Encodes a tree to a single string.
        public string serialize(TreeNode root)
        {
            if (root == null)
            {
                return string.Empty;
            }

            var str = new StringBuilder();
            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                var noChild = true;
                for (int i = 0, j = queue.Count; i < j; i++)
                {
                    root = queue.Dequeue();
                    if (root == null)
                    {
                        str.Append(string.Empty).Append(',');
                    }
                    else
                    {
                        str.Append(root.val).Append(',');
                        queue.Enqueue(root.left);
                        queue.Enqueue(root.right);
                        noChild = noChild && root.left == null && root.right == null;
                    }
                }

                if (noChild)
                {
                    break;
                }
            }

            return str.Remove(str.Length - 1, 1).ToString();
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(string data)
        {
            if (string.IsNullOrEmpty(data))
            {
                return null;
            }

            var eles = data.Split(',');
            var root = new TreeNode(int.Parse(eles[0]));
            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            for (int i = 1; i < eles.Length; i += 2)
            {
                var node = queue.Dequeue();
                if (!string.IsNullOrEmpty(eles[i]))
                {
                    node.left = new TreeNode(int.Parse(eles[i]));
                    queue.Enqueue(node.left);
                }

                if (!string.IsNullOrEmpty(eles[i + 1]))
                {
                    node.right = new TreeNode(int.Parse(eles[i + 1]));
                    queue.Enqueue(node.right);
                }
            }

            return root;
        }
    }
}