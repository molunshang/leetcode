using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;

namespace leetcode
{
    partial class Program
    {
        #region 834. 树中距离之和

        //https://leetcode-cn.com/problems/sum-of-distances-in-tree/
        public int[] SumOfDistancesInTree(int N, int[][] edges)
        {
            #region 暴力解

            int[] Force()
            {
                var graph = new List<int>[N];
                var cache = new int[N, N];
                var visited = new bool[N];
                foreach (var edge in edges)
                {
                    int n1 = edge[0], n2 = edge[1];
                    if (graph[n1] == null)
                    {
                        graph[n1] = new List<int>();
                    }

                    graph[n1].Add(n2);

                    if (graph[n2] == null)
                    {
                        graph[n2] = new List<int>();
                    }

                    graph[n2].Add(n1);
                }

                int Distance(int start, int end)
                {
                    if (start == end)
                    {
                        return 0;
                    }

                    if (cache[start, end] != 0)
                    {
                        return cache[start, end];
                    }

                    var distance = -1;
                    var next = graph[start];
                    visited[start] = true;
                    foreach (var n in next.Where(n => !visited[n]))
                    {
                        distance = Distance(n, end);
                        if (distance == -1)
                        {
                            continue;
                        }

                        distance++;
                        break;
                    }

                    visited[start] = false;
                    if (distance != -1)
                    {
                        cache[start, end] = cache[end, start] = distance;
                    }

                    return distance;
                }

                var result = new int[N];
                for (int i = 0; i < result.Length; i++)
                {
                    var sum = 0;
                    for (int j = 0; j < N; j++)
                    {
                        sum += Distance(i, j);
                    }

                    result[i] = sum;
                }

                return result;
            }

            #endregion

            var treeGraph = new List<int>[N];
            for (var i = 0; i < treeGraph.Length; i++)
            {
                treeGraph[i] = new List<int>();
            }

            foreach (var edge in edges)
            {
                int n1 = edge[0], n2 = edge[1];
                treeGraph[n1].Add(n2);
                treeGraph[n2].Add(n1);
            }

            int[] distanceSum = new int[N], childNum = new int[N];
            Array.Fill(childNum, 1);

            void PostDfs(int root, int parent)
            {
                foreach (var n in treeGraph[root])
                {
                    if (n == parent)
                    {
                        continue;
                    }

                    PostDfs(n, root);
                    childNum[root] += childNum[n];
                    distanceSum[root] = distanceSum[root] + childNum[n] + distanceSum[n];
                }
            }

            void PreDfs(int root, int parent)
            {
                foreach (var n in treeGraph[root])
                {
                    if (n == parent)
                    {
                        continue;
                    }

                    distanceSum[n] = distanceSum[root] - childNum[n] + (N - childNum[n]);
                    PreDfs(n, root);
                }
            }

            PostDfs(0, -1);
            PreDfs(0, -1);
            return distanceSum;
        }

        #endregion

        #region 面试题 17.12. BiNode

        //https://leetcode-cn.com/problems/binode-lcci/
        public TreeNode ConvertBiNode(TreeNode root)
        {
            if (root == null)
            {
                return null;
            }

            var stack = new Stack<TreeNode>();
            var result = new TreeNode(-1);
            var head = result;
            while (stack.Count > 0 || root != null)
            {
                while (root != null)
                {
                    stack.Push(root);
                    root = root.left;
                }

                root = stack.Pop();
                root.left = null;
                head.right = root;
                head = head.right;
                root = root.right;
            }

            return result.right;
        }

        #endregion

        #region 508. 出现次数最多的子树元素和

        //https://leetcode-cn.com/problems/most-frequent-subtree-sum/
        public int[] FindFrequentTreeSum(TreeNode root)
        {
            var counter = new Dictionary<int, int>();
            var max = 0;

            int TreeSum(TreeNode node)
            {
                if (node == null)
                {
                    return 0;
                }

                var sum = node.val + TreeSum(node.left) + TreeSum(node.right);
                if (counter.TryGetValue(sum, out var count))
                {
                    count++;
                }
                else
                {
                    count = 1;
                }

                max = Math.Max(count, max);
                counter[sum] = count;
                return sum;
            }

            TreeSum(root);
            return counter.Where(kv => kv.Value == max).Select(kv => kv.Key).ToArray();
        }

        #endregion

        #region 589. N叉树的前序遍历

        //https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/
        public IList<int> Preorder(Node root)
        {
            var result = new List<int>();

            void Dfs(Node node)
            {
                if (node == null)
                {
                    return;
                }

                result.Add(node.val);
                if (node.children != null && node.children.Count > 0)
                {
                    foreach (var child in node.children)
                    {
                        Dfs(child);
                    }
                }
            }

            Dfs(root);
            return result;
        }

        #endregion

        #region 590. N叉树的后序遍历

        //https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/
        public IList<int> Postorder(Node root)
        {
            var result = new List<int>();

            void Dfs(Node node)
            {
                if (node == null)
                {
                    return;
                }

                if (node.children != null && node.children.Count > 0)
                {
                    foreach (var child in node.children)
                    {
                        Dfs(child);
                    }
                }

                result.Add(node.val);
            }

            Dfs(root);
            return result;
        }

        #endregion

        #region 513. 找树左下角的值

        //https://leetcode-cn.com/problems/find-bottom-left-tree-value/
        public int FindBottomLeftValue(TreeNode root)
        {
            var res = root.val;
            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                res = queue.Peek().val;
                for (int i = 0, j = queue.Count; i < j; i++)
                {
                    root = queue.Dequeue();
                    if (root.left != null)
                    {
                        queue.Enqueue(root.left);
                    }

                    if (root.right != null)
                    {
                        queue.Enqueue(root.right);
                    }
                }
            }

            return res;
        }

        #endregion
    }
}