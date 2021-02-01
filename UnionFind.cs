using System;
using System.Collections.Generic;

namespace leetcode
{
    //并查集
    public class UnionFind
    {
        private int[] nodes;
        private int[] rank;

        public int Count { get; private set; }
        public UnionFind(int n)
        {
            Count = n;
            nodes = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++)
            {
                nodes[i] = i;
                rank[i] = 1;
            }
        }

        public int Find(int n)
        {
            return nodes[n] == n ? n : (nodes[n] = Find(nodes[n]));
        }

        public bool Union(int x, int y)
        {
            int fx = Find(x), fy = Find(y);
            if (fx == fy)
            {
                return false;
            }

            if (rank[fx] == rank[fy])
            {
                nodes[fx] = fy;
                rank[fy]++;
            }
            else if (rank[fx] < rank[fy])
            {
                nodes[fx] = fy;
            }
            else
            {
                nodes[fy] = fx;
            }
            Count--;
            return true;
        }

        public bool IsConnect(int x, int y)
        {
            return Find(x) == Find(y);
        }
    }
}