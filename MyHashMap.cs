using System;
using System.Collections.Generic;

//https://leetcode-cn.com/problems/design-hashmap/
//706. 设计哈希映射
namespace leetcode
{
    public class MyHashMap
    {

        class Entry
        {
            public int key;
            public int value;
            public Entry next;
        }

        private Entry[] entries;

        /** Initialize your data structure here. */
        public MyHashMap()
        {
            entries = new Entry[128];
        }

        /** value will always be non-negative. */
        public void Put(int key, int value)
        {
            var index = key % entries.Length;
            var entry = entries[index];
            if (entry == null)
            {
                entries[index] = new Entry() { key = key, value = value };
                return;
            }
            while (entry != null)
            {
                if (entry.key == key)
                {
                    entry.value = value;
                    return;
                }
                if (entry.next == null)
                {
                    entry.next = new Entry() { key = key, value = value };
                    return;
                }
                entry = entry.next;
            }
        }

        /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
        public int Get(int key)
        {
            var entry = entries[key % entries.Length];
            while (entry != null)
            {
                if (entry.key == key)
                {
                    return entry.value;
                }
                entry = entry.next;
            }
            return -1;
        }

        /** Removes the mapping of the specified value key if this map contains a mapping for the key */
        public void Remove(int key)
        {
            var index = key % entries.Length;
            var entry = entries[index];
            Entry prev = null;
            while (entry != null)
            {
                if (entry.key == key)
                {
                    if (prev == null)
                    {
                        entries[index] = entry.next;
                        entry.next = null;
                    }
                    else
                    {
                        prev.next = entry.next;
                    }
                    break;
                }
                prev = entry;
                entry = entry.next;
            }
        }
    }
}