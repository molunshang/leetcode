using System.Collections.Generic;

namespace leetcode
{
    #region 146. LRU缓存机制

    //146. LRU缓存机制
    //https://leetcode-cn.com/problems/lru-cache/
    class LRUCache
    {
        class CacheNode
        {
            public int val;
            public int key;
            public CacheNode prev;
            public CacheNode next;
        }

        private int capacity;
        private Dictionary<int, CacheNode> dic = new Dictionary<int, CacheNode>();
        private CacheNode head;
        private CacheNode tail;

        public LRUCache(int capacity)
        {
            this.capacity = capacity;
        }

        private void MoveToTail(CacheNode node)
        {
            if (node == tail)
            {
                //尾节点
                return;
            }

            if (node == head)
            {
                //头节点
                head = node.next;
                head.prev = null;
            }
            else
            {
                //非头尾节点，连接前后节点
                CacheNode prev = node.prev, next = node.next;
                prev.next = next;
                next.prev = prev;
            }

            //将节点移动到尾节点
            node.prev = tail;
            tail.next = node;
            tail = node;
            tail.next = null;
        }

        public int Get(int key)
        {
            if (!dic.TryGetValue(key, out var node))
            {
                return -1;
            }

            MoveToTail(node);
            return node.val;
        }

        public void Put(int key, int value)
        {
            if (dic.TryGetValue(key, out var node))
            {
                node.val = value;
                MoveToTail(node);
            }
            else
            {
                dic[key] = node = new CacheNode {key = key, val = value};
                if (tail == null)
                {
                    head = tail = node;
                }
                else
                {
                    tail.next = node;
                    node.prev = tail;
                    tail = node;
                }
            }

            while (dic.Count > capacity)
            {
                dic.Remove(head.key);
                head = head.next;
                if (head == null)
                {
                    break;
                }

                head.prev = null;
            }
        }
    }

    #endregion
}